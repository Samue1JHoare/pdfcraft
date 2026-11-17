//! Pages drawn by a scene target (a GPU) instead of the CPU rasterizer (feature `scenes`).
//!
//! A [`RenderConfig::scenes`](super::RenderConfig::scenes) has each page recorded as a
//! `vello_hybrid` scene (`hayro::render_scene`) and handed to its [`SceneTarget`], which
//! rasterizes it into something of its own (on macOS, an IOSurface Core Animation shows as is)
//! and returns it as the page's [`Native`]. A page the scene can't draw exactly (`render_scene`
//! names why), or one the target couldn't draw, is drawn by the CPU as before, and a page once
//! declined goes straight to the CPU afterwards. The target decides what it can do; this module
//! only routes pages and remembers.

use std::collections::HashMap;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::page::Page;
use hayro::{HybridPage, RenderCache, RenderSettings};

use super::Native;

/// Rasterizes recorded pages: see the module notes.
pub trait SceneTarget: Send + Sync {
    /// The widest or tallest image, in pixels, the target can bind as a texture; a page drawing a
    /// larger one is drawn by the CPU.
    fn max_image_side(&self) -> u16;

    /// Whether the target can draw now. While it can't (a device still starting, or one that
    /// failed), pages go straight to the CPU without being recorded.
    fn ready(&self) -> bool {
        true
    }

    /// Draws `page`, `page.width` × `page.height` pixels, and returns what it drew, or `None` to
    /// have the page drawn by the CPU instead. Called on a render worker; it may block until the
    /// drawing is complete, and must not return before it is.
    fn draw(&self, page: HybridPage) -> Option<Native>;
}

/// A [`SceneTarget`] for one document's renders, and the pages it left to the CPU.
#[derive(Clone)]
pub struct Scenes {
    target: Arc<dyn SceneTarget>,
    cpu_pages: Arc<Mutex<HashMap<usize, &'static str>>>,
    /// The target panicked: every page is drawn by the CPU from then on.
    broken: Arc<AtomicBool>,
}

impl fmt::Debug for Scenes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scenes")
            .field("cpu_pages", &self.cpu_pages.lock().map(|p| p.len()).unwrap_or(0))
            .field("broken", &self.broken.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Scenes {
    /// Pages drawn by `target`, for one document's renders.
    pub fn new(target: Arc<dyn SceneTarget>) -> Self {
        Self { target, cpu_pages: Arc::default(), broken: Arc::default() }
    }

    /// The pages (by index) the CPU has drawn because the scene couldn't, each with why.
    pub fn cpu_pages(&self) -> Vec<(usize, &'static str)> {
        let mut pages: Vec<_> = self.cpu_pages.lock().unwrap_or_else(PoisonError::into_inner).iter().map(|(p, w)| (*p, *w)).collect();
        pages.sort_unstable();
        pages
    }

    /// Page `index` drawn by the target, or `None` for the CPU to draw it.
    pub(crate) fn draw<'a>(
        &self,
        index: usize,
        page: &'a Page<'a>,
        cache: &RenderCache<'a>,
        settings: &InterpreterSettings,
        render_settings: &RenderSettings,
    ) -> Option<Native> {
        if self.broken.load(Ordering::Relaxed) || !self.target.ready() || self.cpu_pages.lock().unwrap_or_else(PoisonError::into_inner).contains_key(&index) {
            return None;
        }
        let recorded = catch_unwind(AssertUnwindSafe(|| hayro::render_scene(page, cache, settings, render_settings, self.target.max_image_side())));
        let scene = match recorded {
            Ok(Ok(scene)) => scene,
            Ok(Err(why)) => return self.leave(index, why),
            // The CPU draws it, under the worker's own guard, should it panic there too.
            Err(_) => return self.leave(index, "recording the scene panicked"),
        };
        match catch_unwind(AssertUnwindSafe(|| self.target.draw(scene))) {
            Ok(native) => native,
            Err(_) => {
                self.broken.store(true, Ordering::Relaxed);
                None
            }
        }
    }

    fn leave(&self, index: usize, why: &'static str) -> Option<Native> {
        self.cpu_pages.lock().unwrap_or_else(PoisonError::into_inner).insert(index, why);
        None
    }
}
