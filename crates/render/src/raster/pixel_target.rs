//! Pages the CPU draws into memory of a caller's (feature `scenes`).
//!
//! A [`RenderConfig::cpu_target`](super::RenderConfig::cpu_target) lends the memory a page's
//! raster is drawn into, in place of the renderer's own [`Pixels`](crate::Pixels): on macOS, the
//! IOSurface Core Animation shows, so the raster is never copied. The target returns what it lent
//! as the page's [`Native`]. A target that declines, or fails, leaves the page to be drawn into
//! pixels of its own as before.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::Native;

/// Lends the memory a page's raster is drawn into: see the module notes.
pub trait PixelTarget: Send + Sync {
    /// Lends memory for a `width` × `height` raster: calls `draw` at most once with exactly
    /// `width` × `height` × 4 bytes, which it fills with premultiplied RGBA8 rows whatever they
    /// held, and returns what was drawn. `None`, before or after calling `draw`, to have the page
    /// drawn into pixels of its own instead. Called on a render worker.
    fn draw(&self, width: u32, height: u32, draw: &mut dyn FnMut(&mut [u8])) -> Option<Native>;
}

/// A [`PixelTarget`] for one document's renders.
#[derive(Clone)]
pub struct CpuTarget {
    target: Arc<dyn PixelTarget>,
    /// The target panicked: every page is drawn into pixels of its own from then on.
    broken: Arc<AtomicBool>,
}

impl fmt::Debug for CpuTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuTarget").field("broken", &self.broken.load(Ordering::Relaxed)).finish_non_exhaustive()
    }
}

impl CpuTarget {
    /// Pages drawn into memory `target` lends, for one document's renders.
    pub fn new(target: Arc<dyn PixelTarget>) -> Self {
        Self { target, broken: Arc::default() }
    }

    /// A `width` × `height` raster drawn by `render` into the target's memory, or `None` for the
    /// renderer to draw it into pixels of its own. Only a raster `render` drew in full, into a
    /// buffer of the right length, is returned.
    pub(crate) fn draw(&self, width: u32, height: u32, render: &mut dyn FnMut(&mut [u8])) -> Option<Native> {
        if self.broken.load(Ordering::Relaxed) {
            return None;
        }
        let len = (width as usize).checked_mul(height as usize)?.checked_mul(4)?;
        let mut drew = false;
        let lent = catch_unwind(AssertUnwindSafe(|| {
            self.target.draw(width, height, &mut |buf: &mut [u8]| {
                if !drew && buf.len() == len {
                    render(buf);
                    drew = true;
                }
            })
        }));
        match lent {
            Ok(native) => native.filter(|_| drew),
            // The renderer draws it into its own pixels, under the worker's own guard, should it
            // panic there too.
            Err(_) => {
                self.broken.store(true, Ordering::Relaxed);
                None
            }
        }
    }
}
