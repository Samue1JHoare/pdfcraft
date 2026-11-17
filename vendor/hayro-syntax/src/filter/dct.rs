use crate::object::Dict;
use crate::object::dict::keys::COLOR_TRANSFORM;
use crate::object::stream::{FilterResult, ImageColorSpace, ImageData, ImageDecodeParams};
use alloc::borrow::Cow;
use core::num::NonZeroU32;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::colorspace::ColorSpace::CMYK;
use zune_jpeg::zune_core::options::DecoderOptions;

/// PdfCraft patch (48): how many times smaller, on each side, a JPEG image is decoded for
/// `params.target_dimension`: 8, 4 or 2, the largest reduction whose size (each side divided by it,
/// rounded up) is still at least the size the image is drawn at; 1 (the full size) with no target,
/// for an `/Indexed` image (averaging palette indices makes other colours) and for anything but
/// 8-bit samples of 1, 3 or 4 components. The decoded-image cache keys a JPEG image by this, so
/// draws that share a reduction share one decode, and a smaller decode never serves a larger draw.
pub fn dct_reduction(params: &ImageDecodeParams) -> u32 {
    let Some((tw, th)) = params.target_dimension else {
        return 1;
    };
    if params.is_indexed
        || params.bpc.is_some_and(|b| b != 8)
        || params.num_components.is_some_and(|n| !matches!(n, 1 | 3 | 4))
    {
        return 1;
    }
    let (w, h) = (params.width, params.height);
    if w == 0 || h == 0 || tw == 0 || th == 0 {
        return 1;
    }
    [8, 4, 2]
        .into_iter()
        .find(|&f| w.div_ceil(f) >= tw && h.div_ceil(f) >= th)
        .unwrap_or(1)
}

/// PdfCraft patch (48): a JPEG decoded at a reduced size by a host's decoder
/// ([`set_scaled_jpeg_decoder`]): 8-bit samples, `components` (1, grey, or 3, RGB) per pixel,
/// rows packed with no padding.
pub struct ScaledJpeg {
    /// `width × height × components` samples.
    pub data: Vec<u8>,
    /// Pixels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// 1 (grey) or 3 (RGB).
    pub components: u8,
}

/// PdfCraft patch (48): decodes the JPEG `data` at 1/`factor` of its size on each side (`factor`
/// is 2, 4 or 8), as raw samples: no colour management, no orientation. `None` to decode it here
/// at full size instead. Only grey and YCbCr (JFIF) JPEGs of 8-bit samples are handed to it.
pub type ScaledJpegDecoder = fn(data: &[u8], factor: u32) -> Option<ScaledJpeg>;

#[cfg(feature = "std")]
static SCALED_JPEG_DECODER: std::sync::OnceLock<ScaledJpegDecoder> = std::sync::OnceLock::new();

/// PdfCraft patch (48): installs a decoder that decodes a JPEG at a reduced size for the process
/// (on Apple platforms, ImageIO's subsampled decode, which never makes the full-size image).
/// Without one, an image drawn much smaller than it is decoded at full size and then averaged
/// down ([`dct_reduction`]). The first decoder installed stays; `false` if one already was.
#[cfg(feature = "std")]
pub fn set_scaled_jpeg_decoder(decoder: ScaledJpegDecoder) -> bool {
    SCALED_JPEG_DECODER.set(decoder).is_ok()
}

#[cfg(feature = "std")]
fn scaled_jpeg_decoder() -> Option<ScaledJpegDecoder> {
    SCALED_JPEG_DECODER.get().copied()
}

#[cfg(not(feature = "std"))]
fn scaled_jpeg_decoder() -> Option<ScaledJpegDecoder> {
    None
}

pub(crate) fn decode(
    data: &[u8],
    params: &Dict<'_>,
    image_params: &ImageDecodeParams,
) -> Option<FilterResult<'static>> {
    if image_params.width > u16::MAX as u32 || image_params.height > u16::MAX as u32 {
        return None;
    }

    // Some PDFs have weird JPEGs where the JPEG metadata is completely wrong
    // (for example indicating that one of the dimensions is u16::MAX), but the
    // metadata in the PDF image dictionary is correct. Therefore, we first
    // validate the JPEG metadata and patch the data if any of the dimensions
    // are too large (if they are too small, they will just be padded later on).
    let data = maybe_patch_jpeg_dimensions(data, image_params)?;

    let options = DecoderOptions::default()
        .set_max_width(u16::MAX as usize)
        .set_max_height(u16::MAX as usize);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&*data), options);
    decoder.decode_headers().ok()?;

    let color_transform = params.get::<u8>(COLOR_TRANSFORM);
    let input_color_space = decoder.input_colorspace().unwrap();

    let mut out_colorspace = if let Some(num_components) = image_params.num_components
        && !matches!(num_components, 1 | 3 | 4)
    {
        ColorSpace::MultiBand(NonZeroU32::new(num_components as u32)?)
    } else {
        match input_color_space {
            ColorSpace::YCbCr => {
                if color_transform.is_none_or(|c| c == 1) {
                    ColorSpace::RGB
                } else {
                    ColorSpace::YCbCr
                }
            }
            ColorSpace::RGB | ColorSpace::RGBA => ColorSpace::RGB,
            ColorSpace::Luma | ColorSpace::LumaA => ColorSpace::Luma,
            // TODO: Find test case with color transform on cmyk
            CMYK => CMYK,
            ColorSpace::YCCK => ColorSpace::YCCK,
            _ => ColorSpace::RGB,
        }
    };

    // In case image had APP14 marker, we might have to override the colorspace.
    if input_color_space == CMYK && decoder.info().unwrap().components == 3 {
        out_colorspace = ColorSpace::RGB;
    }

    // PdfCraft patch (48): drawn much smaller than it is, a grey or JFIF colour JPEG is decoded
    // at that reduction by the host's decoder, if there is one and it can.
    let factor = dct_reduction(image_params);
    if factor > 1
        && matches!(data, Cow::Borrowed(_))
        && let Some(decoder_fn) = scaled_jpeg_decoder()
        && let Some(info) = decoder.info()
        && let Some(components) = host_components(
            &data,
            image_params,
            (u32::from(info.width), u32::from(info.height)),
            info.components,
            input_color_space,
            out_colorspace,
            color_transform,
        )
        && let Some(result) = scaled_by_host(decoder_fn, &data, factor, components, image_params)
    {
        return Some(result);
    }

    decoder.set_options(DecoderOptions::default().jpeg_set_out_colorspace(out_colorspace));
    let mut decoded = decoder.decode().ok()?;

    if out_colorspace == ColorSpace::YCCK {
        // See <https://github.com/mozilla/pdf.js/blob/69595a29192b7704733404a42a2ebb537601117b/src/core/jpg.js#L1331>
        for c in decoded.chunks_mut(4) {
            let y = c[0] as f32;
            let cb = c[1] as f32;
            let cr = c[2] as f32;
            c[0] = (434.456 - y - 1.402 * cr) as u8;
            c[1] = (119.541 - y + 0.344 * cb + 0.714 * cr) as u8;
            c[2] = (481.816 - y - 1.772 * cb) as u8;
        }
    }

    let mut width = decoder.dimensions().unwrap().0 as u32;
    let mut height = decoder.dimensions().unwrap().1 as u32;

    // PdfCraft patch (48): decoded in full here (no host decoder, or a JPEG it isn't handed:
    // CMYK, YCCK, Adobe, `/ColorTransform 0`, …), and averaged down by the same reduction, in the
    // decoded samples, so each output sample is the mean of full-decode samples and only the
    // reduced image is kept. Samples that don't fill the decoded size are kept as they are.
    if factor > 1
        && let Some(channels) = (width as usize)
            .checked_mul(height as usize)
            .filter(|&n| n > 0 && decoded.len() % n == 0)
            .map(|n| decoded.len() / n)
        && let Some((shrunk, w, h)) = shrink(&decoded, width, height, channels, factor)
    {
        decoded = shrunk;
        width = w;
        height = h;
    }

    let image_data = ImageData {
        alpha: None,
        color_space: match out_colorspace {
            ColorSpace::RGB | ColorSpace::YCbCr => Some(ImageColorSpace::Rgb),
            ColorSpace::Luma => Some(ImageColorSpace::Gray),
            ColorSpace::YCCK | CMYK => Some(ImageColorSpace::Cmyk),
            ColorSpace::MultiBand(_) => None,
            _ => None,
        },
        bits_per_component: 8,
        width,
        height,
    };

    Some(FilterResult {
        data: Cow::Owned(decoded),
        image_data: Some(image_data),
    })
}

/// PdfCraft patch (48): the components per pixel (1 grey, 3 RGB) the host's decoder is to return
/// for this JPEG, or `None` when it is not handed to the host. Handed over only when the host
/// decodes it to the same samples the full decode here makes: one 8-bit Huffman-coded frame
/// (baseline, extended or progressive) of the size the image dictionary gives, grey or YCbCr
/// turned to RGB, no Adobe (APP14) marker, no `/ColorTransform` but 1 and a colour space, if
/// any, with the same number of components.
fn host_components(
    data: &[u8],
    image_params: &ImageDecodeParams,
    jpeg_size: (u32, u32),
    jpeg_components: u8,
    input: ColorSpace,
    output: ColorSpace,
    color_transform: Option<u8>,
) -> Option<u8> {
    if image_params.is_indexed || jpeg_size != (image_params.width, image_params.height) {
        return None;
    }
    let components = match (input, output, jpeg_components) {
        (ColorSpace::Luma, ColorSpace::Luma, 1) => 1,
        (ColorSpace::YCbCr, ColorSpace::RGB, 3) if color_transform.is_none_or(|c| c == 1) => 3,
        _ => return None,
    };
    if image_params.num_components.is_some_and(|n| n != components) {
        return None;
    }
    let header = read_header(data)?;
    (header.precision == 8 && matches!(header.sof, 0xC0..=0xC2) && !header.adobe).then_some(components)
}

/// PdfCraft patch (48): `data` decoded by `decoder_fn` at 1/`factor`, if what it returned is
/// what it was asked for: `components` samples per pixel, each side the image's divided by
/// `factor` (rounded either way), the data exactly that long.
fn scaled_by_host(
    decoder_fn: ScaledJpegDecoder,
    data: &[u8],
    factor: u32,
    components: u8,
    image_params: &ImageDecodeParams,
) -> Option<FilterResult<'static>> {
    let scaled = decoder_fn(data, factor)?;
    let near = |got: u32, full: u32| got > 0 && (got == full / factor || got == full.div_ceil(factor));
    let len = (scaled.width as usize)
        .checked_mul(scaled.height as usize)?
        .checked_mul(usize::from(components))?;
    if scaled.components != components
        || !near(scaled.width, image_params.width)
        || !near(scaled.height, image_params.height)
        || scaled.data.len() != len
    {
        return None;
    }
    Some(FilterResult {
        data: Cow::Owned(scaled.data),
        image_data: Some(ImageData {
            alpha: None,
            color_space: Some(if components == 1 {
                ImageColorSpace::Gray
            } else {
                ImageColorSpace::Rgb
            }),
            bits_per_component: 8,
            width: scaled.width,
            height: scaled.height,
        }),
    })
}

/// PdfCraft patch (48): `data`, `width × height` pixels of `channels` samples, averaged down by
/// `factor` on each side: each output sample is the rounded mean of the samples in its
/// `factor × factor` cell (fewer at the right and bottom edges, so the size is rounded up).
/// Buffers are reserved fallibly; `None` when `data` is not that long.
fn shrink(
    data: &[u8],
    width: u32,
    height: u32,
    channels: usize,
    factor: u32,
) -> Option<(Vec<u8>, u32, u32)> {
    let (w, h, f) = (width as usize, height as usize, factor as usize);
    let row = w.checked_mul(channels)?;
    if f < 2 || w == 0 || h == 0 || channels == 0 || data.len() != row.checked_mul(h)? {
        return None;
    }
    let (ow, oh) = (w.div_ceil(f), h.div_ceil(f));
    let out_row = ow.checked_mul(channels)?;
    let mut out = Vec::new();
    out.try_reserve_exact(out_row.checked_mul(oh)?).ok()?;
    let mut sums: Vec<u32> = Vec::new();
    sums.try_reserve_exact(out_row).ok()?;
    sums.resize(out_row, 0);
    for rows in data.chunks(row.checked_mul(f)?) {
        sums.fill(0);
        for line in rows.chunks_exact(row) {
            for (x, pixel) in line.chunks_exact(channels).enumerate() {
                let base = (x / f) * channels;
                for (sum, &v) in sums.get_mut(base..base + channels)?.iter_mut().zip(pixel) {
                    *sum += u32::from(v);
                }
            }
        }
        let n_rows = (rows.len() / row) as u32;
        for (ox, cell) in sums.chunks_exact(channels).enumerate() {
            let n_cols = (w.min(ox * f + f) - ox * f) as u32;
            let n = n_rows * n_cols;
            if n == 0 {
                return None;
            }
            out.extend(cell.iter().map(|&s| ((s + n / 2) / n) as u8));
        }
    }
    Some((out, ow as u32, oh as u32))
}

/// PdfCraft patch (48): what [`host_components`] reads from a JPEG's markers up to its first scan.
struct JpegHeader {
    /// The frame marker's second byte (0xC0 baseline, 0xC1 extended, 0xC2 progressive, …).
    sof: u8,
    /// Bits per sample.
    precision: u8,
    /// Whether an Adobe APP14 segment (its own colour transform) comes before the scan.
    adobe: bool,
}

/// PdfCraft patch (48): the frame and Adobe markers of `data`, read up to the first scan (SOS);
/// `None` when no frame comes before it or a segment runs past the data.
fn read_header(data: &[u8]) -> Option<JpegHeader> {
    let mut i = 0_usize;
    let (mut frame, mut adobe) = (None, false);
    while i.checked_add(1).is_some_and(|next| next < data.len()) {
        if *data.get(i)? != 0xFF {
            i += 1;
            continue;
        }
        let marker = *data.get(i + 1)?;
        match marker {
            0xFF => {
                i += 1;
                continue;
            }
            0xD8 | 0xD9 | 0x01 | 0x00 | 0xD0..=0xD7 => {
                i += 2;
                continue;
            }
            // Start of scan: the header is over.
            0xDA => break,
            _ => {}
        }
        let seg_len = u16::from_be_bytes([*data.get(i + 2)?, *data.get(i + 3)?]) as usize;
        let segment = data.get(i + 4..i.checked_add(2)?.checked_add(seg_len)?)?;
        match marker {
            0xC0..=0xCF if marker != 0xC4 && marker != 0xC8 && marker != 0xCC => {
                frame.get_or_insert((marker, *segment.first()?));
            }
            0xEE if segment.starts_with(b"Adobe") => adobe = true,
            _ => {}
        }
        i = i.checked_add(2)?.checked_add(seg_len)?;
    }
    let (sof, precision) = frame?;
    Some(JpegHeader { sof, precision, adobe })
}

fn maybe_patch_jpeg_dimensions<'a>(
    data: &'a [u8],
    image_params: &ImageDecodeParams,
) -> Option<Cow<'a, [u8]>> {
    let sof_offset = find_sof_marker(data)?;

    let height_offset = sof_offset.checked_add(5)?;
    let width_offset = sof_offset.checked_add(7)?;

    let jpeg_height = u16::from_be_bytes([
        *data.get(height_offset)?,
        *data.get(height_offset.checked_add(1)?)?,
    ]);
    let jpeg_width = u16::from_be_bytes([
        *data.get(width_offset)?,
        *data.get(width_offset.checked_add(1)?)?,
    ]);

    let jpeg_area = (jpeg_width as usize).checked_mul(jpeg_height as usize)?;
    let image_area = (image_params.width as usize).checked_mul(image_params.height as usize)?;
    let need_patch = jpeg_area > image_area;

    if !need_patch {
        return Some(Cow::Borrowed(data));
    }

    let target_w = (image_params.width as u16).to_be_bytes();
    let target_h = (image_params.height as u16).to_be_bytes();

    let mut patched = data.to_vec();
    patched[height_offset..height_offset.checked_add(2)?].copy_from_slice(&target_h);
    patched[width_offset..width_offset.checked_add(2)?].copy_from_slice(&target_w);

    Some(Cow::Owned(patched))
}

fn find_sof_marker(data: &[u8]) -> Option<usize> {
    let mut i = 0_usize;

    while i.checked_add(1).is_some_and(|next| next < data.len()) {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }

        let marker = data[i + 1];

        // Note: Not sure if 100% correct/robust, is AI-generated.
        match marker {
            // All SOF markers carry dimensions: SOF0–SOF15, excluding
            // 0xC4 (DHT), 0xC8 (JPG), 0xCC (DAC) which are not frame markers.
            0xC0..=0xCF if marker != 0xC4 && marker != 0xC8 && marker != 0xCC => {
                return Some(i);
            }
            // Skip padding bytes (0xFF followed by 0xFF).
            0xFF => {
                i += 1;

                continue;
            }
            // SOI (0xD8), EOI (0xD9), TEM (0x01) and stuffed byte (0x00)
            // are standalone markers with no payload.
            0xD8 | 0xD9 | 0x01 | 0x00 => {
                i += 2;

                continue;
            }
            // All other markers have a 2-byte length field — skip over them.
            _ => {
                let len_start = i.checked_add(2)?;
                let len_end = i.checked_add(3)?;
                let seg_len =
                    u16::from_be_bytes([*data.get(len_start)?, *data.get(len_end)?]) as usize;

                i = i.checked_add(2)?.checked_add(seg_len)?;
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(width: u32, height: u32, target: Option<(u32, u32)>) -> ImageDecodeParams {
        ImageDecodeParams {
            width,
            height,
            target_dimension: target,
            ..Default::default()
        }
    }

    #[test]
    fn the_reduction_is_the_largest_that_stays_at_or_above_the_drawn_size() {
        // An A4 page scanned at 300 dpi.
        assert_eq!(dct_reduction(&params(2480, 3508, Some((310, 439)))), 8);
        assert_eq!(dct_reduction(&params(2480, 3508, Some((311, 439)))), 4);
        assert_eq!(dct_reduction(&params(2480, 3508, Some((310, 440)))), 4);
        assert_eq!(dct_reduction(&params(2480, 3508, Some((1240, 1754)))), 2);
        assert_eq!(dct_reduction(&params(2480, 3508, Some((1241, 1754)))), 1);
        assert_eq!(dct_reduction(&params(2480, 3508, Some((4960, 7016)))), 1);
    }

    #[test]
    fn no_target_an_indexed_image_or_odd_samples_decode_in_full() {
        assert_eq!(dct_reduction(&params(2480, 3508, None)), 1);
        let small = Some((10, 10));
        assert_eq!(dct_reduction(&ImageDecodeParams { is_indexed: true, ..params(2480, 3508, small) }), 1);
        assert_eq!(dct_reduction(&ImageDecodeParams { bpc: Some(16), ..params(2480, 3508, small) }), 1);
        assert_eq!(dct_reduction(&ImageDecodeParams { num_components: Some(2), ..params(2480, 3508, small) }), 1);
        assert_eq!(dct_reduction(&ImageDecodeParams { num_components: Some(4), ..params(2480, 3508, small) }), 8);
        assert_eq!(dct_reduction(&params(0, 3508, small)), 1);
        assert_eq!(dct_reduction(&params(2480, 3508, Some((0, 10)))), 1);
    }

    #[test]
    fn shrinking_averages_each_cell_and_rounds_the_size_up() {
        // 5 × 3 grey samples by 2: 3 × 2, the right column and bottom row from part cells.
        let grey: Vec<u8> = (0..15).map(|i| i * 10).collect();
        let (out, w, h) = shrink(&grey, 5, 3, 1, 2).unwrap();
        assert_eq!((w, h), (3, 2));
        assert_eq!(out, [30, 50, 65, 105, 125, 140]);
        // RGB keeps its channels apart.
        let rgb = [10, 20, 30, 30, 40, 50, 50, 60, 70, 70, 80, 90];
        let (out, w, h) = shrink(&rgb, 2, 2, 3, 2).unwrap();
        assert_eq!((w, h, out), (1, 1, vec![40, 50, 60]));
    }

    #[test]
    fn shrinking_refuses_data_of_the_wrong_length_or_no_reduction() {
        let rgb = [0u8; 4 * 4 * 3];
        assert!(shrink(&rgb, 4, 4, 3, 1).is_none());
        assert!(shrink(&rgb[1..], 4, 4, 3, 2).is_none());
        assert!(shrink(&rgb, 0, 4, 3, 2).is_none());
        assert!(shrink(&rgb, 4, 4, 0, 2).is_none());
    }

    fn jfif(sof: u8, precision: u8) -> Vec<u8> {
        let mut data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 16];
        data.extend(b"JFIF\0");
        data.extend([1, 1, 0, 0, 1, 0, 1, 0, 0]);
        data.extend([0xFF, sof, 0, 11, precision, 0, 4, 0, 4, 1, 1, 0x11, 0]);
        data.extend([0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 63, 0]);
        data
    }

    #[test]
    fn the_header_names_the_frame_its_precision_and_an_adobe_marker() {
        let plain = read_header(&jfif(0xC0, 8)).unwrap();
        assert_eq!((plain.sof, plain.precision, plain.adobe), (0xC0, 8, false));
        let progressive = read_header(&jfif(0xC2, 12)).unwrap();
        assert_eq!((progressive.sof, progressive.precision), (0xC2, 12));
        let mut adobe = jfif(0xC0, 8);
        adobe.splice(2..2, [0xFF, 0xEE, 0, 14, b'A', b'd', b'o', b'b', b'e', 0, 100, 0, 0, 0, 0, 1]);
        assert!(read_header(&adobe).unwrap().adobe);
    }

    #[test]
    fn a_header_cut_short_or_without_a_frame_is_none() {
        let data = jfif(0xC0, 8);
        assert!(read_header(&data[..12]).is_none());
        assert!(read_header(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2]).is_none());
        assert!(read_header(&[0xFF, 0xD8, 0xFF, 0xE0, 0xFF, 0xFF]).is_none());
    }
}
