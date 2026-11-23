//! PdfCraft patch (34): a page drawn once and replayed for each of its GPU tiles.
//!
//! Interpreting a page (parsing its content, resolving fonts and glyph outlines, decoding and
//! resampling its images) is most of what a tile costs, and every tile of a page used to pay it
//! again. [`record_page`] draws the whole page once, as `render_scene` would (the background,
//! crop box and page group included), onto a canvas that keeps the drawing calls the renderer
//! makes. [`render_recorded`] plays those calls onto one tile's canvas with every transform
//! moved by the tile's origin, skipping fills that land off the tile.
//!
//! A tile is then exactly the matching part of the whole page: the calls are the same, and an
//! integer move of the device leaves whatever the renderer rounds to whole pixels (snapped
//! images, the page's own edges) where it was. A tile interpreted on its own applies its origin
//! before the page's transforms instead, so on a page whose transforms cancel at magnitudes past
//! f64's precision (translate by 1e19 and back) such tiles disagree with the whole page and with
//! each other; replayed ones do not.
//!
//! An image is decoded and resampled once, and every tile draws the same pixmap (`Arc`), so a
//! GPU target can upload it once. A page is not recorded, and its tiles are drawn one by one as
//! before, when it draws what a recording can't hold exactly or within its budget: masks, blends
//! but Normal, more than [`MAX_OPS`] calls or [`MAX_ELEMENTS`] path elements, or more
//! than [`MAX_PIXMAP_BYTES`] of images.

use crate::canvas::Canvas;
use crate::renderer::Renderer;
use crate::*;
use kurbo::{BezPath, Stroke};
use std::sync::Arc;
use vello_common::filter_effects::Filter;
use vello_cpu::peniko::{BlendMode as PenikoBlend, Fill};
use vello_cpu::{Mask, PaintType};

/// The most drawing calls, path elements and image bytes a recording holds. A dense page of text
/// is some 20,000 calls (a transform, a paint and a fill a glyph), its outlines held once each.
const MAX_OPS: usize = 1 << 17;
const MAX_ELEMENTS: usize = 1 << 18;
const MAX_PIXMAP_BYTES: usize = 64 << 20;

/// Device pixels a fill may paint past its control box: anti-aliasing and the slack the
/// renderer's own culling allows.
const FILL_SLACK: f64 = 2.0;

#[derive(Clone)]
enum Op {
    Transform(Affine),
    Paint(PaintType),
    PaintTransform(Affine),
    FillRule(Fill),
    Stroke(Stroke),
    Aliasing(Option<u8>),
    /// A path filled, and the device bounds (before the tile's move) it can paint.
    Fill(Arc<BezPath>, Rect),
    StrokePath(Arc<BezPath>),
    /// A rectangle filled (an image's, or the background), and its device bounds.
    Rect(Rect, Rect),
    PushClip(Arc<BezPath>),
    PopClip,
    PushLayer(Option<Arc<BezPath>>, Option<PenikoBlend>, Option<f32>),
    Blend(PenikoBlend),
    PushClipLayer(Arc<BezPath>),
    PopLayer,
}

/// A page drawn at one scale, by [`record_page`].
pub struct PageRecording {
    ops: Vec<Op>,
    bytes: usize,
}

impl PageRecording {
    /// Roughly what the recording holds in memory: its calls, paths and images.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// The canvas [`record_page`] draws on: it keeps the calls, and the state the renderer reads
/// back.
struct Recorder {
    width: u16,
    height: u16,
    settings: vello_cpu::RenderSettings,
    transform: Affine,
    fill_rule: Fill,
    stroke: Stroke,
    blend_mode: PenikoBlend,
    ops: Vec<Op>,
    /// Paths by the address they were drawn from (a glyph's cached outline), each kept once.
    paths: HashMap<usize, Arc<BezPath>>,
    pixmaps: Vec<*const Pixmap>,
    elements: usize,
    pixmap_bytes: usize,
    declined: Option<&'static str>,
}

impl Recorder {
    fn refuse(&mut self, why: &'static str) {
        self.declined.get_or_insert(why);
    }

    fn push(&mut self, op: Op) {
        if self.ops.len() >= MAX_OPS {
            self.refuse("more drawing than a recording holds");
        }
        if self.declined.is_none() {
            self.ops.push(op);
        }
    }

    /// `path`, kept once however often it is drawn.
    fn path(&mut self, path: &BezPath) -> Arc<BezPath> {
        let key = std::ptr::from_ref(path) as usize;
        if let Some(kept) = self.paths.get(&key)
            && **kept == *path
        {
            return kept.clone();
        }
        self.elements += path.elements().len();
        if self.elements > MAX_ELEMENTS {
            self.refuse("more paths than a recording holds");
        }
        let kept = Arc::new(path.clone());
        self.paths.insert(key, kept.clone());
        kept
    }

    fn bounds(&self, rect: Rect) -> Rect {
        self.transform.transform_rect_bbox(rect).inflate(FILL_SLACK, FILL_SLACK)
    }
}

impl Canvas for Recorder {
    fn width(&self) -> u16 {
        self.width
    }
    fn height(&self) -> u16 {
        self.height
    }
    fn render_settings(&self) -> &vello_cpu::RenderSettings {
        &self.settings
    }
    fn transform(&self) -> &Affine {
        &self.transform
    }
    fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
        self.push(Op::Transform(transform));
    }
    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        let paint = paint.into();
        match &paint {
            PaintType::Solid(_) => {}
            PaintType::Image(image) => {
                if let vello_cpu::ImageSource::Pixmap(pixmap) = &image.image {
                    let ptr = Arc::as_ptr(pixmap);
                    if !self.pixmaps.contains(&ptr) {
                        self.pixmaps.push(ptr);
                        self.pixmap_bytes += pixmap.data_as_u8_slice().len();
                        if self.pixmap_bytes > MAX_PIXMAP_BYTES {
                            self.refuse("more images than a recording holds");
                        }
                    }
                }
            }
            // A gradient is placed by the transform it is drawn with and its paint transform,
            // composed (`vello_hybrid::Scene::set_paint_transform`), so it moves with the tile.
            _ => {}
        }
        self.push(Op::Paint(paint));
    }
    fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.push(Op::PaintTransform(paint_transform));
    }
    fn fill_rule(&self) -> &Fill {
        &self.fill_rule
    }
    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
        self.push(Op::FillRule(fill_rule));
    }
    fn blend_mode(&self) -> PenikoBlend {
        self.blend_mode
    }
    fn set_blend_mode(&mut self, blend_mode: PenikoBlend) {
        self.blend_mode = blend_mode;
        self.push(Op::Blend(blend_mode));
    }
    fn stroke(&self) -> &Stroke {
        &self.stroke
    }
    fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke.clone();
        self.push(Op::Stroke(stroke));
    }
    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.push(Op::Aliasing(aliasing_threshold));
    }
    fn set_mask(&mut self, _mask: Mask) {
        self.refuse("mask");
    }
    fn reset_mask(&mut self) {}
    fn fill_path(&mut self, path: &BezPath) {
        let bounds = self.bounds(path.control_box());
        let path = self.path(path);
        self.push(Op::Fill(path, bounds));
    }
    fn stroke_path(&mut self, path: &BezPath) {
        let path = self.path(path);
        self.push(Op::StrokePath(path));
    }
    fn fill_rect(&mut self, rect: &Rect) {
        let bounds = self.bounds(*rect);
        self.push(Op::Rect(*rect, bounds));
    }
    fn push_clip_path(&mut self, path: &BezPath) {
        let path = self.path(path);
        self.push(Op::PushClip(path));
    }
    fn pop_clip_path(&mut self) {
        self.push(Op::PopClip);
    }
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<PenikoBlend>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        if mask.is_some() {
            self.refuse("mask");
        } else if filter.is_some() {
            self.refuse("filter");
        }
        let clip = clip_path.map(|p| self.path(p));
        self.push(Op::PushLayer(clip, blend_mode, opacity));
    }
    fn push_clip_layer(&mut self, path: &BezPath) {
        let path = self.path(path);
        self.push(Op::PushClipLayer(path));
    }
    fn push_blend_layer(&mut self, blend_mode: PenikoBlend) {
        self.push_layer(None, Some(blend_mode), None, None, None);
    }
    fn push_mask_layer(&mut self, _mask: Mask) {
        self.refuse("mask");
    }
    fn pop_layer(&mut self) {
        self.push(Op::PopLayer);
    }
    fn decline(&mut self, feature: &'static str) -> bool {
        self.refuse(feature);
        true
    }
    fn live(&self) -> bool {
        self.declined.is_none()
    }
}

/// `page` at `render_settings`' scale drawn once, for [`render_recorded`] to draw its tiles, or
/// `Err` naming why it can't be (the module notes). The offset, width and height are ignored:
/// the whole page is recorded.
pub fn record_page<'a>(
    page: &'a Page<'a>,
    cache: &RenderCache<'a>,
    interpreter_settings: &InterpreterSettings,
    render_settings: &RenderSettings,
) -> Result<PageRecording, &'static str> {
    let whole = RenderSettings { x_offset: 0.0, y_offset: 0.0, width: None, height: None, ..render_settings.clone() };
    let (width, height) = render_size(page, &whole);
    if width == 0 || height == 0 {
        return Err("empty page");
    }
    // A pixel past the page each way, so the renderer culls nothing an edge tile would draw.
    let (width, height) = (width.saturating_add(2), height.saturating_add(2));
    let whole = RenderSettings { width: Some(width), height: Some(height), ..whole };
    let recorder = Recorder {
        width,
        height,
        settings: cpu_settings(),
        transform: Affine::IDENTITY,
        fill_rule: Fill::NonZero,
        stroke: Stroke::default(),
        blend_mode: PenikoBlend::default(),
        ops: Vec::new(),
        paths: HashMap::new(),
        pixmaps: Vec::new(),
        elements: 0,
        pixmap_bytes: 0,
        declined: None,
    };
    let mut device = Renderer::new(recorder, cache, whole.color_replacement, interpreter_settings.preview);
    draw_page(page, cache, interpreter_settings, &whole, &mut device);
    let recorder = device.ctx;
    if let Some(why) = recorder.declined {
        return Err(why);
    }
    let bytes = recorder.ops.len() * size_of::<Op>() + recorder.elements * size_of::<kurbo::PathEl>() + recorder.pixmap_bytes;
    Ok(PageRecording { ops: recorder.ops, bytes })
}

/// One tile of a recorded page, the region `render_settings` names, as [`render_scene`] would
/// record it.
pub fn render_recorded(
    recording: &PageRecording,
    page: &Page<'_>,
    render_settings: &RenderSettings,
    max_image_side: u16,
) -> Result<HybridPage, &'static str> {
    let (width, height) = render_size(page, render_settings);
    if width == 0 || height == 0 {
        return Err("empty page");
    }
    let mut canvas = hybrid::HybridCanvas::new(width, height, cpu_settings(), max_image_side);
    let (x, y) = (f64::from(render_settings.x_offset), f64::from(render_settings.y_offset));
    let tile = Rect::new(x, y, x + f64::from(width), y + f64::from(height));
    let shift = Affine::translate((-x, -y));
    // The recorder's canvas started at the identity, as this one does, before the move.
    canvas.set_transform(shift);
    for op in &recording.ops {
        match op {
            Op::Transform(t) => canvas.set_transform(shift * *t),
            Op::Paint(p) => canvas.set_paint(p.clone()),
            Op::PaintTransform(t) => canvas.set_paint_transform(*t),
            Op::FillRule(f) => canvas.set_fill_rule(*f),
            Op::Stroke(s) => canvas.set_stroke(s.clone()),
            Op::Aliasing(a) => canvas.set_aliasing_threshold(*a),
            Op::Fill(path, bounds) if bounds.overlaps(tile) => canvas.fill_path(path),
            Op::Rect(rect, bounds) if bounds.overlaps(tile) => canvas.fill_rect(rect),
            Op::Fill(..) | Op::Rect(..) => {}
            Op::StrokePath(path) => canvas.stroke_path(path),
            Op::PushClip(path) => canvas.push_clip_path(path),
            Op::PopClip => canvas.pop_clip_path(),
            Op::PushLayer(clip, blend, opacity) => canvas.push_layer(clip.as_deref(), *blend, *opacity, None, None),
            Op::Blend(b) => canvas.set_blend_mode(*b),
            Op::PushClipLayer(path) => canvas.push_clip_layer(path),
            Op::PopLayer => canvas.pop_layer(),
        }
    }
    canvas.finish()
}
