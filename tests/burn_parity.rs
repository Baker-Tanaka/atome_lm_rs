use atome_lm_rs::{Atome, CONTEXT, MODEL_BYTES, VOCAB};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

struct TrackingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let used = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(used, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[test]
fn published_model_matches_numpy_with_bounded_allocations() {
    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    let model = Atome::load(MODEL_BYTES).unwrap();
    let fixture = include_bytes!("fixtures/reference.bin");
    let mut data = fixture.as_slice();
    let mut cases = 0;
    let mut max_error = 0.0f32;
    while !data.is_empty() {
        let len = usize::from(data[0]);
        let tokens = &data[1..1 + len];
        let expected = &data[1 + len..1 + len + VOCAB * 4];
        data = &data[1 + len + VOCAB * 4..];
        let mut inference = model.start(tokens).unwrap();
        let mut layers = 0;
        loop {
            layers += 1;
            if inference.step() {
                break;
            }
            assert!(layers < 8);
        }
        assert_eq!(layers, 8);
        assert!(inference.step());
        let actual = inference.logits().unwrap();
        let expected: Vec<f32> = expected
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for (&a, &e) in actual.iter().zip(&expected) {
            let error = (a - e).abs();
            max_error = max_error.max(error);
            assert!(
                error <= 2.0e-4 + 2.0e-5 * e.abs(),
                "prompt={tokens:?} actual={a} expected={e} error={error}"
            );
        }
        let argmax = |values: &[f32]| {
            let mut best = 0;
            for i in 1..values.len() {
                if values[i] > values[best] {
                    best = i;
                }
            }
            best
        };
        assert_eq!(argmax(&actual), argmax(&expected));
        cases += 1;
    }
    assert_eq!(cases, 8);
    assert_eq!(CONTEXT, 16);
    drop(model);
    let peak = PEAK.load(Ordering::SeqCst) - baseline;
    eprintln!("cases={cases} max_logit_error={max_error} peak_requested_heap={peak}");
    assert!(peak < 192 * 1024, "peak requested heap: {peak}");
}
