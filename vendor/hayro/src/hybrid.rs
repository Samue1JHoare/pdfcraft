//! PdfCraft patch (33): pages recorded for `vello_hybrid`, which rasterizes on the GPU.
//!
//! [`HybridCanvas`] is a [`Canvas`] over a `vello_hybrid::Scene`. The renderer drives it exactly
//! as it drives `vello_cpu` (the same paints, paths, clips, image resampling and stroke widths),
//! so the two differ only in the last step, coverage to pixels. What `vello_hybrid` 0.0.8 cannot
//! draw exactly is declined, and the page is then drawn with `vello_cpu` instead:
//!
//! - masks (soft masks, image soft masks of another size than the image);
//! - every blend but Normal source-over, so knockout groups (drawn with `Copy` and `DestOut`)
//!   and blend modes;
//! - CMYK groups (drawn in ink passes of their own);
//! - an image used as the paint of a path rather than drawn as its rectangle (tiling patterns,
//!   shading textures), and images wider or taller than the target allows;
//! - filters, and pages with more paths than [`MAX_DRAWS`].
//!
//! Images are not uploaded while the page is recorded: each is drawn as an external texture,
//! `TextureId(i)` for the `i`-th of [`HybridPage::images`], which the caller binds when it
//! renders the scene.

use crate::canvas::Canvas;
use kurbo::{Affine, BezPath, Rect, Stroke};
use std::sync::Arc;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_cpu::peniko::{BlendMode, Extend, Fill, ImageQuality};
use vello_cpu::{ImageSource, Mask, PaintType, Pixmap, RenderSettings};
use vello_hybrid::{SampleRect, Scene, TextureId};

/// Paths (fills and strokes) a recorded page may have. `vello_cpu` keeps a page within its
/// coverage budget (patch 31); a scene has no such budget, so a page drawing more paths than this
/// is left to `vello_cpu`.
pub const MAX_DRAWS: usize = 400_000;

/// A page recorded for `vello_hybrid` by [`render_scene`](crate::render_scene).
pub struct HybridPage {
    /// The page, `width` × `height` pixels.
    pub scene: Scene,
    /// The images the scene draws: the `i`-th is `TextureId(i)`, premultiplied RGBA8, to be bound
    /// as a non-filterable float texture when the scene is rendered.
    pub images: Vec<Arc<Pixmap>>,
    /// Width of the scene in pixels.
    pub width: u16,
    /// Height of the scene in pixels.
    pub height: u16,
}

/// The pixmap image set as the current paint, drawn when its rectangle is filled.
struct PendingImage {
    pixmap: Arc<Pixmap>,
    quality: ImageQuality,
}

pub(crate) struct HybridCanvas {
    scene: Scene,
    /// What nested `vello_cpu` renders are made with, as for `RenderContext`.
    settings: RenderSettings,
    transform: Affine,
    fill_rule: Fill,
    blend_mode: BlendMode,
    paint_transform: Affine,
    image: Option<PendingImage>,
    images: Vec<Arc<Pixmap>>,
    max_image_side: u16,
    draws: usize,
    declined: Option<&'static str>,
}

impl HybridCanvas {
    pub(crate) fn new(
        width: u16,
        height: u16,
        settings: RenderSettings,
        max_image_side: u16,
    ) -> Self {
        let scene_settings = vello_hybrid::RenderSettings {
            level: settings.level,
            ..vello_hybrid::RenderSettings::default()
        };
        Self {
            scene: Scene::new_with(width, height, scene_settings),
            settings,
            transform: Affine::IDENTITY,
            fill_rule: Fill::NonZero,
            blend_mode: BlendMode::default(),
            paint_transform: Affine::IDENTITY,
            image: None,
            images: Vec::new(),
            max_image_side,
            draws: 0,
            declined: None,
        }
    }

    /// The recorded page, or what was declined.
    pub(crate) fn finish(self) -> Result<HybridPage, &'static str> {
        if let Some(why) = self.declined {
            return Err(why);
        }
        let (width, height) = (self.scene.width(), self.scene.height());
        Ok(HybridPage {
            scene: self.scene,
            images: self.images,
            width,
            height,
        })
    }

    fn refuse(&mut self, why: &'static str) {
        self.declined.get_or_insert(why);
    }

    fn dead(&self) -> bool {
        self.declined.is_some()
    }

    /// Counts a fill or stroke; past [`MAX_DRAWS`] the page is declined.
    fn draw(&mut self) -> bool {
        self.draws += 1;
        if self.draws > MAX_DRAWS {
            self.refuse("more paths than a scene takes");
        }
        !self.dead()
    }
}

impl Canvas for HybridCanvas {
    fn width(&self) -> u16 {
        self.scene.width()
    }
    fn height(&self) -> u16 {
        self.scene.height()
    }
    fn render_settings(&self) -> &RenderSettings {
        &self.settings
    }
    fn transform(&self) -> &Affine {
        &self.transform
    }
    fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
        self.scene.set_transform(transform);
    }
    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.image = None;
        match paint.into() {
            PaintType::Image(image) => match image.image {
                // Drawn when its rectangle is filled (`Renderer::draw_pixmap`); a scene takes no
                // pixmap paints. Padded and opaque, as `draw_pixmap` sets it, or declined there.
                ImageSource::Pixmap(pixmap)
                    if image.sampler.alpha == 1.0
                        && image.sampler.x_extend == Extend::Pad
                        && image.sampler.y_extend == Extend::Pad =>
                {
                    self.image = Some(PendingImage {
                        pixmap,
                        quality: image.sampler.quality,
                    });
                }
                _ => self.refuse("image paint"),
            },
            paint => self.scene.set_paint(paint),
        }
    }
    fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.paint_transform = paint_transform;
        self.scene.set_paint_transform(paint_transform);
    }
    fn fill_rule(&self) -> &Fill {
        &self.fill_rule
    }
    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
        self.scene.set_fill_rule(fill_rule);
    }
    fn blend_mode(&self) -> BlendMode {
        self.blend_mode
    }
    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
        if blend_mode != BlendMode::default() {
            self.refuse("blend mode");
        }
    }
    fn stroke(&self) -> &Stroke {
        self.scene.stroke()
    }
    fn set_stroke(&mut self, stroke: Stroke) {
        self.scene.set_stroke(stroke);
    }
    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.scene.set_aliasing_threshold(aliasing_threshold);
    }
    fn set_mask(&mut self, _mask: Mask) {
        self.refuse("mask");
    }
    fn reset_mask(&mut self) {}
    fn fill_path(&mut self, path: &BezPath) {
        if self.image.is_some() {
            self.refuse("image paint on a path");
        }
        if self.draw() {
            self.scene.fill_path(path);
        }
    }
    fn stroke_path(&mut self, path: &BezPath) {
        if self.image.is_some() {
            self.refuse("image paint on a path");
        }
        if self.draw() {
            self.scene.stroke_path(path);
        }
    }
    fn fill_rect(&mut self, rect: &Rect) {
        if !self.draw() {
            return;
        }
        let Some((pixmap, quality)) = self.image.as_ref().map(|i| (i.pixmap.clone(), i.quality)) else {
            self.scene.fill_rect(rect);
            return;
        };
        let (width, height) = (pixmap.width(), pixmap.height());
        // The image's own rectangle, as `draw_pixmap` fills it; anything else is a pattern.
        if *rect != Rect::new(0.0, 0.0, f64::from(width), f64::from(height))
            || self.paint_transform != Affine::IDENTITY
        {
            self.refuse("image paint on a rectangle");
            return;
        }
        if width > self.max_image_side || height > self.max_image_side {
            self.refuse("image larger than a GPU texture");
            return;
        }
        let id =TextureId(self.images.len() as u64);
        self.images.push(pixmap);
        self.scene.draw_texture_rects(
            id,
            quality,
            [SampleRect {
                source_region: RectU16::new(0, 0, width, height),
                transform: Affine::IDENTITY,
            }],
        );
    }
    fn push_clip_path(&mut self, path: &BezPath) {
        if !self.dead() {
            self.scene.push_clip_path(path);
        }
    }
    fn pop_clip_path(&mut self) {
        if !self.dead() {
            self.scene.pop_clip_path();
        }
    }
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        if mask.is_some() {
            self.refuse("mask");
        } else if filter.is_some() {
            self.refuse("filter");
        } else if blend_mode.is_some_and(|b| b != BlendMode::default()) {
            self.refuse("blend mode");
        }
        if !self.dead() {
            self.scene.push_layer(clip_path, None, opacity, None, None);
        }
    }
    fn push_clip_layer(&mut self, path: &BezPath) {
        if !self.dead() {
            self.scene.push_clip_layer(path);
        }
    }
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(None, Some(blend_mode), None, None, None);
    }
    fn push_mask_layer(&mut self, _mask: Mask) {
        self.refuse("mask");
    }
    fn pop_layer(&mut self) {
        if !self.dead() {
            self.scene.pop_layer();
        }
    }
    fn decline(&mut self, feature: &'static str) -> bool {
        self.refuse(feature);
        true
    }
    fn live(&self) -> bool {
        !self.dead()
    }
}
