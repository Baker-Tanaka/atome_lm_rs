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
            if inference.step().unwrap() {
                break;
            }
            assert!(layers < 8);
        }
        assert_eq!(layers, 8);
        assert!(inference.step().unwrap());
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
    assert_eq!(CONTEXT, 24);
    drop(model);
    let peak = PEAK.load(Ordering::SeqCst) - baseline;
    eprintln!("cases={cases} max_logit_error={max_error} peak_requested_heap={peak}");
    assert!(peak < 192 * 1024, "peak requested heap: {peak}");
}

#[test]
fn cached_generation_matches_full_context_recomputation() {
    let model = Atome::load(MODEL_BYTES).unwrap();
    let mut generation = model.cached_generation(CONTEXT).unwrap();
    let mut tokens = b"Once upon a".to_vec();
    let mut max_error = 0.0f32;
    let mut cached_time = std::time::Duration::ZERO;
    let mut full_time = std::time::Duration::ZERO;
    let mut cursor = 0;
    while cursor < CONTEXT {
        let token = tokens[cursor];
        generation.start_token(token).unwrap();
        let started = std::time::Instant::now();
        for _ in 0..8 {
            if generation.step().unwrap() {
                break;
            }
        }
        cached_time += started.elapsed();
        let cached = generation.logits().unwrap();
        let mut full = model.start(&tokens[..=cursor]).unwrap();
        let started = std::time::Instant::now();
        while !full.step().unwrap() {}
        full_time += started.elapsed();
        let full = full.logits().unwrap();
        for (&actual, &expected) in cached.iter().zip(&full) {
            max_error = max_error.max((actual - expected).abs());
        }
        let cached_next = argmax(cached);
        let full_next = argmax(&full);
        assert_eq!(cached_next, full_next, "tokens={:?}", &tokens[..=cursor]);
        assert!(
            cached
                .iter()
                .zip(&full)
                .all(|(&a, &b)| (a - b).abs() <= 2.0e-2 + 2.0e-3 * b.abs()),
            "cached logits drifted for tokens={:?}",
            &tokens[..=cursor]
        );
        if cursor + 1 == tokens.len() && cursor + 1 < CONTEXT {
            tokens.push(cached_next as u8);
        }
        cursor += 1;
    }
    assert!(generation.start_token(0).is_err());
    assert_eq!(&tokens[11..15], b" tim");
    eprintln!(
        "cached max_logit_error={max_error} cached_cpu={cached_time:?} full_context_cpu={full_time:?}"
    );
}

#[test]
fn cached_generation_fits_in_reserved_heap() {
    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    let model = Atome::load(MODEL_BYTES).unwrap();
    let mut generation = model.cached_generation(CONTEXT).unwrap();
    let mut tokens = b"Once upon a".to_vec();
    while tokens.len() < CONTEXT {
        generation.start_token(*tokens.last().unwrap()).unwrap();
        for _ in 0..8 {
            if generation.step().unwrap() {
                break;
            }
        }
        let next = argmax(generation.logits().unwrap());
        tokens.push(next as u8);
    }
    drop(generation);
    drop(model);
    let peak = PEAK.load(Ordering::SeqCst) - baseline;
    eprintln!("cached inference peak_requested_heap={peak}");
    assert!(peak < 192 * 1024, "cached peak requested heap: {peak}");
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}
