//! A pixel target lends the memory the CPU draws a page into (`RenderConfig::cpu_target`, feature
//! `scenes`): the page comes back as the target's `Native`, the same pixels the renderer draws
//! into its own; a target that declines, lends nothing or panics leaves the page to its own pixels.

use super::ONE_PAGE;
use crate::{CpuTarget, Native, PageRenderer, PixelTarget, RenderConfig, RenderRequest, RequestKind, Tile};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Lends a fresh buffer of `fill` bytes, or behaves as `how` says.
struct Lender {
    how: How,
    calls: AtomicUsize,
}

#[derive(Clone, Copy, PartialEq)]
enum How {
    Lend,
    Decline,
    /// Returns a native without having anything drawn.
    Skip,
    /// Lends a buffer a byte short.
    Short,
    Panic,
}

impl PixelTarget for Lender {
    fn draw(&self, width: u32, height: u32, draw: &mut dyn FnMut(&mut [u8])) -> Option<Native> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let len = width as usize * height as usize * 4;
        match self.how {
            How::Decline => None,
            How::Panic => panic!("the target failed"),
            How::Skip => Some(Native(Arc::new(Mutex::new(vec![0u8; len])))),
            How::Lend | How::Short => {
                let mut buf = vec![0xAA; if self.how == How::Short { len - 1 } else { len }];
                draw(&mut buf);
                Some(Native(Arc::new(Mutex::new(buf))))
            }
        }
    }
}

fn render(target: Option<&Arc<Lender>>, tile: Option<Tile>) -> crate::RenderedPage {
    let cpu_target = target.map(|t| CpuTarget::new(t.clone() as Arc<dyn PixelTarget>));
    let mut renderer = PageRenderer::new(Arc::new(ONE_PAGE.to_vec()), RenderConfig { cpu_target, ..RenderConfig::default() });
    let page = renderer.render(RenderRequest { page: 0, kind: RequestKind::Pixels, tile, scale: 1.0, tag: 0, preview: false });
    assert!(page.error.is_none(), "{:?}", page.error);
    page
}

fn lent(page: &crate::RenderedPage) -> Vec<u8> {
    let native = page.native.as_ref().expect("drawn into the target").0.clone();
    let buf = native.downcast::<Mutex<Vec<u8>>>().expect("the target's buffer");
    let pixels = buf.lock().unwrap().clone();
    pixels
}

#[test]
fn a_page_drawn_into_a_lent_buffer_is_the_page_drawn_into_its_own() {
    for tile in [None, Some(Tile { x: 3, y: 5, w: 40, h: 30 })] {
        let own = render(None, tile);
        let target = Arc::new(Lender { how: How::Lend, calls: AtomicUsize::new(0) });
        let page = render(Some(&target), tile);
        assert!(page.rgba.is_empty(), "no pixels of its own");
        assert_eq!((page.width, page.height), (own.width, own.height));
        assert!(lent(&page) == *own.rgba, "the same pixels");
        assert_eq!(target.calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn a_target_that_lends_nothing_usable_leaves_the_page_its_own_pixels() {
    let own = render(None, None);
    for how in [How::Decline, How::Skip, How::Short, How::Panic] {
        let target = Arc::new(Lender { how, calls: AtomicUsize::new(0) });
        let page = render(Some(&target), None);
        assert!(page.native.is_none());
        assert!(page.rgba == own.rgba);
    }
    // A target that panicked is not asked again.
    let target = Arc::new(Lender { how: How::Panic, calls: AtomicUsize::new(0) });
    let shared = CpuTarget::new(target.clone() as Arc<dyn PixelTarget>);
    let mut renderer = PageRenderer::new(Arc::new(ONE_PAGE.to_vec()), RenderConfig { cpu_target: Some(shared), ..RenderConfig::default() });
    for _ in 0..2 {
        let page = renderer.render(RenderRequest { page: 0, kind: RequestKind::Pixels, tile: None, scale: 1.0, tag: 0, preview: false });
        assert!(page.native.is_none() && !page.rgba.is_empty());
    }
    assert_eq!(target.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn printing_never_uses_the_target() {
    let target = Arc::new(Lender { how: How::Lend, calls: AtomicUsize::new(0) });
    let mut renderer = PageRenderer::new(Arc::new(ONE_PAGE.to_vec()), RenderConfig { cpu_target: Some(CpuTarget::new(target.clone())), ..RenderConfig::default() });
    let page = renderer.render(RenderRequest { page: 0, kind: RequestKind::Pixels, tile: None, scale: 1.0, tag: crate::PRINT_TAG, preview: false });
    assert!(page.native.is_none() && !page.rgba.is_empty());
    assert_eq!(target.calls.load(Ordering::Relaxed), 0);
}
