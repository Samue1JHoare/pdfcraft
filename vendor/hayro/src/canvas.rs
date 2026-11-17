//! PdfCraft patch (32): the surface the renderer draws on.
//!
//! [`Renderer`](crate::renderer::Renderer) works out everything a page draws (paints, image
//! resampling, stroke widths, knockout and CMYK passes) and hands the result to a [`Canvas`]:
//! `vello_cpu`'s `RenderContext` by default. The trait is exactly the part of `RenderContext`
//! the renderer calls, with its signatures, so another rasterizer that consumes the same
//! `vello_common` paints and paths can take its place and draw the same page, and the
//! renderer's own code is shared rather than copied.
//!
//! A canvas that cannot draw something exactly says so through [`Canvas::decline`] (asked
//! before costly work such as rendering a soft mask) or by refusing the call itself; the caller
//! of such a canvas then draws the page with `RenderContext` instead. `RenderContext` declines
//! nothing.

use kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::filter_effects::Filter;
use vello_cpu::peniko::{BlendMode, Fill};
use vello_cpu::{Mask, PaintType, RenderContext, RenderSettings};

/// What [`Renderer`](crate::renderer::Renderer) draws on. See the module notes.
pub(crate) trait Canvas {
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    /// The CPU settings nested renders (soft masks, pattern cells, ink passes) are made with.
    fn render_settings(&self) -> &RenderSettings;
    fn transform(&self) -> &Affine;
    fn set_transform(&mut self, transform: Affine);
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    fn set_paint_transform(&mut self, paint_transform: Affine);
    fn fill_rule(&self) -> &Fill;
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn blend_mode(&self) -> BlendMode;
    fn set_blend_mode(&mut self, blend_mode: BlendMode);
    fn stroke(&self) -> &Stroke;
    fn set_stroke(&mut self, stroke: Stroke);
    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>);
    fn set_mask(&mut self, mask: Mask);
    fn reset_mask(&mut self);
    fn fill_path(&mut self, path: &BezPath);
    fn stroke_path(&mut self, path: &BezPath);
    fn fill_rect(&mut self, rect: &Rect);
    fn push_clip_path(&mut self, path: &BezPath);
    fn pop_clip_path(&mut self);
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    );
    fn push_clip_layer(&mut self, path: &BezPath);
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    fn push_mask_layer(&mut self, mask: Mask);
    fn pop_layer(&mut self);

    /// Whether this canvas leaves `feature` (a soft mask, a knockout group, …) to another
    /// rasterizer. Once it has, it draws nothing more of the page, which is then drawn again
    /// with `RenderContext`.
    fn decline(&mut self, _feature: &'static str) -> bool {
        false
    }

    /// Whether drawing on this canvas still counts. `false` once it has declined something:
    /// the renderer may then skip work (decoding an image) whose result would be thrown away.
    fn live(&self) -> bool {
        true
    }
}

impl Canvas for RenderContext {
    fn width(&self) -> u16 {
        Self::width(self)
    }
    fn height(&self) -> u16 {
        Self::height(self)
    }
    fn render_settings(&self) -> &RenderSettings {
        Self::render_settings(self)
    }
    fn transform(&self) -> &Affine {
        Self::transform(self)
    }
    fn set_transform(&mut self, transform: Affine) {
        Self::set_transform(self, transform);
    }
    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        Self::set_paint(self, paint);
    }
    fn set_paint_transform(&mut self, paint_transform: Affine) {
        Self::set_paint_transform(self, paint_transform);
    }
    fn fill_rule(&self) -> &Fill {
        Self::fill_rule(self)
    }
    fn set_fill_rule(&mut self, fill_rule: Fill) {
        Self::set_fill_rule(self, fill_rule);
    }
    fn blend_mode(&self) -> BlendMode {
        Self::blend_mode(self)
    }
    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        Self::set_blend_mode(self, blend_mode);
    }
    fn stroke(&self) -> &Stroke {
        Self::stroke(self)
    }
    fn set_stroke(&mut self, stroke: Stroke) {
        Self::set_stroke(self, stroke);
    }
    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        Self::set_aliasing_threshold(self, aliasing_threshold);
    }
    fn set_mask(&mut self, mask: Mask) {
        Self::set_mask(self, mask);
    }
    fn reset_mask(&mut self) {
        Self::reset_mask(self);
    }
    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }
    fn stroke_path(&mut self, path: &BezPath) {
        Self::stroke_path(self, path);
    }
    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }
    fn push_clip_path(&mut self, path: &BezPath) {
        Self::push_clip_path(self, path);
    }
    fn pop_clip_path(&mut self) {
        Self::pop_clip_path(self);
    }
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        Self::push_layer(self, clip_path, blend_mode, opacity, mask, filter);
    }
    fn push_clip_layer(&mut self, path: &BezPath) {
        Self::push_clip_layer(self, path);
    }
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        Self::push_blend_layer(self, blend_mode);
    }
    fn push_mask_layer(&mut self, mask: Mask) {
        Self::push_mask_layer(self, mask);
    }
    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }
}
