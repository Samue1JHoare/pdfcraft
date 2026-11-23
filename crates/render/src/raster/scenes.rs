//! Pages drawn by a scene target (a GPU) instead of the CPU rasterizer (feature `scenes`).
//!
//! A [`RenderConfig::scenes`](super::RenderConfig::scenes) has each page recorded as a
//! `vello_hybrid` scene (`hayro::render_scene`) and handed to its [`SceneTarget`], which
//! rasterizes it into something of its own (on macOS, an IOSurface Core Animation shows as is)
//! and returns it as the page's [`Native`]. A page the scene can't draw exactly (`render_scene`
//! names why), or one the target couldn't draw, is drawn by the CPU as before, and a page once
//! declined goes straight to the CPU afterwards. The target decides what it can do; this module
//! only routes pages and remembers.
//!
//! A page asked for in tiles is read once (`hayro::record_page`) and each tile replayed from the
//! recording (`hayro::render_recorded`), for the last [`RECORDINGS`] pages and scales asked for.
//! A page that can't be recorded is remembered as such, and its tiles interpreted one by one, as
//! before.

use std::collections::HashMap;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::page::Page;
use hayro::{HybridPage, PageRecording, RenderCache, RenderSettings};

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

/// Pages whose recordings a document keeps, the ones most recently asked for: enough for the
/// pages of a screen or two (a few views of one document).
pub const RECORDINGS: usize = 8;
/// The memory a document's recordings may hold together: a page of text is a megabyte or two,
/// a scanned page tens of megabytes, its image.
pub const RECORDING_BYTES: usize = 160 << 20;

/// A page at one scale: its index and the bits of its x and y scale.
type Key = (usize, u32, u32);
/// A page's recording, made by the first tile that wants it while the others wait; `None` when it
/// couldn't be made (the page is then in `unrecorded`).
type Slot = Arc<OnceLock<Option<Arc<PageRecording>>>>;

/// A [`SceneTarget`] for one document's renders, and the pages it left to the CPU. Clones share
/// everything; a document makes a new one when what its pages show changes (its configuration),
/// so no recording outlives what it recorded.
#[derive(Clone)]
pub struct Scenes {
    target: Arc<dyn SceneTarget>,
    cpu_pages: Arc<Mutex<HashMap<usize, &'static str>>>,
    /// The target panicked: every page is drawn by the CPU from then on.
    broken: Arc<AtomicBool>,
    /// The recordings, most recently asked for last.
    recordings: Arc<Mutex<Vec<(Key, Slot)>>>,
    /// Pages at a scale that couldn't be recorded, and why.
    unrecorded: Arc<Mutex<HashMap<Key, &'static str>>>,
    replay: bool,
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
        Self { target, cpu_pages: Arc::default(), broken: Arc::default(), recordings: Arc::default(), unrecorded: Arc::default(), replay: true }
    }

    /// Every tile interpreted on its own, as before recordings: for checking replayed tiles.
    pub fn without_replay(self) -> Self {
        Self { replay: false, ..self }
    }

    /// The recording of the page `key` names, made now if it isn't yet.
    fn recording<'a>(
        &self,
        key: Key,
        page: &'a Page<'a>,
        cache: &RenderCache<'a>,
        settings: &InterpreterSettings,
        render_settings: &RenderSettings,
    ) -> Option<Arc<PageRecording>> {
        if self.unrecorded.lock().unwrap_or_else(PoisonError::into_inner).contains_key(&key) {
            return None;
        }
        let slot = {
            let mut recordings = self.recordings.lock().unwrap_or_else(PoisonError::into_inner);
            let entry = match recordings.iter().position(|(k, _)| *k == key) {
                Some(i) => recordings.remove(i),
                None => {
                    if recordings.len() == RECORDINGS {
                        recordings.remove(0);
                    }
                    (key, Slot::default())
                }
            };
            let slot = entry.1.clone();
            recordings.push(entry);
            // The oldest go first while all of them hold more than the budget.
            let bytes = |r: &[(Key, Slot)]| r.iter().filter_map(|(_, s)| s.get()?.as_ref().map(|r| r.bytes())).sum::<usize>();
            while recordings.len() > 1 && bytes(&recordings) > RECORDING_BYTES {
                recordings.remove(0);
            }
            slot
        };
        slot.get_or_init(|| match hayro::record_page(page, cache, settings, render_settings) {
            Ok(recording) => Some(Arc::new(recording)),
            Err(why) => {
                self.unrecorded.lock().unwrap_or_else(PoisonError::into_inner).insert(key, why);
                None
            }
        })
        .clone()
    }

    /// The pages (by index) whose tiles are interpreted one by one because the page couldn't be
    /// recorded, each with why.
    pub fn unrecorded(&self) -> Vec<(usize, &'static str)> {
        let mut pages: Vec<_> = self.unrecorded.lock().unwrap_or_else(PoisonError::into_inner).iter().map(|((p, ..), w)| (*p, *w)).collect();
        pages.sort_unstable();
        pages.dedup_by_key(|(p, _)| *p);
        pages
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
        tiled: bool,
        page: &'a Page<'a>,
        cache: &RenderCache<'a>,
        settings: &InterpreterSettings,
        render_settings: &RenderSettings,
    ) -> Option<Native> {
        if self.broken.load(Ordering::Relaxed) || !self.target.ready() || self.cpu_pages.lock().unwrap_or_else(PoisonError::into_inner).contains_key(&index) {
            return None;
        }
        let max_image_side = self.target.max_image_side();
        let recorded = catch_unwind(AssertUnwindSafe(|| {
            let key = (index, render_settings.x_scale.to_bits(), render_settings.y_scale.to_bits());
            let recording = (self.replay && tiled && !settings.preview)
                .then(|| self.recording(key, page, cache, settings, render_settings))
                .flatten();
            match recording {
                Some(recording) => hayro::render_recorded(&recording, page, render_settings, max_image_side),
                None => hayro::render_scene(page, cache, settings, render_settings, max_image_side),
            }
        }));
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
