#![no_std]
#![no_main]

use atome_lm_rs::{Atome, CachedGeneration, CONTEXT, MODEL_BYTES};
use core::{
    alloc::{GlobalAlloc, Layout},
    cell::Cell,
    mem::MaybeUninit,
};
use critical_section::Mutex;
use defmt::{error, info, Debug2Format};
use defmt_rtt as _;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_time::{Instant, Timer};
use embedded_alloc::LlffHeap;
use panic_probe as _;

const HEAP_SIZE: usize = 192 * 1024;
static HEAP: LlffHeap = LlffHeap::empty();
static HEAP_PEAK: Mutex<Cell<usize>> = Mutex::new(Cell::new(0));
static mut HEAP_MEMORY: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

struct Allocator;

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { HEAP.alloc(layout) };
        if ptr.is_null() {
            error!(
                "Heap exhausted: requested={} free={}",
                layout.size(),
                HEAP.free()
            );
        } else {
            critical_section::with(|cs| {
                let peak = HEAP_PEAK.borrow(cs);
                peak.set(peak.get().max(HEAP.used()));
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { HEAP.dealloc(ptr, layout) };
    }
}

const PROMPT_MAX: usize = 32;
const MAX_NEW: usize = 48;

async fn process_token(generation: &mut CachedGeneration<'_, '_>, token: u8) {
    generation.start_token(token).unwrap_or_else(|e| {
        defmt::panic!("Invalid token: {}", Debug2Format(&e));
    });
    for _ in 0..8 {
        generation.step().unwrap_or_else(|e| {
            defmt::panic!("Cached block inference failed: {}", Debug2Format(&e));
        });
        Timer::after_millis(1).await;
    }
}

#[embassy_executor::task]
async fn inference_task() {
    let model = Atome::load(MODEL_BYTES).unwrap_or_else(|e| {
        defmt::panic!("Model load failed: {}", Debug2Format(&e));
    });
    let prompt = b"Once upon a time, in";
    let mut tokens = [0u8; PROMPT_MAX + MAX_NEW];
    tokens[..prompt.len()].copy_from_slice(prompt);
    let mut generation = model.cached_generation(CONTEXT).unwrap_or_else(|e| {
        defmt::panic!("Invalid generation window: {}", Debug2Format(&e));
    });
    info!(
        "Burn Flex / cached Atome 944K: window={} max_new={} heap={}",
        CONTEXT, MAX_NEW, HEAP_SIZE
    );
    let prefill_started = Instant::now();
    for &token in prompt {
        process_token(&mut generation, token).await;
    }
    info!(
        "Prefill: {} bytes in {} ms",
        prompt.len(),
        prefill_started.elapsed().as_millis()
    );
    let mut length = prompt.len();
    let decode_started = Instant::now();
    for index in 0..MAX_NEW {
        let logits = generation.logits().unwrap_or_else(|e| {
            defmt::panic!("Cached inference logits unavailable: {}", Debug2Format(&e));
        });
        let mut next = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[next] {
                next = i;
            }
        }
        tokens[length] = next as u8;
        length += 1;
        if matches!(next as u8, b'.' | b'!' | b'?' | b'\n') {
            info!("Stopped at sentence end after {} tokens", index + 1);
            break;
        }
        if index + 1 == MAX_NEW {
            break;
        }
        process_token(&mut generation, next as u8).await;
        if (index + 1) % 8 == 0 {
            info!(
                "{} tokens, {} ms/token",
                index + 1,
                decode_started.elapsed().as_millis() / (index as u64 + 1)
            );
        }
    }
    let peak = critical_section::with(|cs| HEAP_PEAK.borrow(cs).get());
    info!("Heap: used={} peak={}", HEAP.used(), peak);
    info!("Generation complete: bytes={=[u8]}", &tokens[..length]);
    match core::str::from_utf8(&tokens[..length]) {
        Ok(text) => info!("Generated text: {}", defmt::Display2Format(text)),
        Err(error) => error!(
            "Output is not valid UTF-8: valid_up_to={}",
            error.valid_up_to()
        ),
    }
}

#[embassy_executor::task]
async fn green_led_task(mut led: Output<'static>) {
    info!("Green LED task started (1s interval)");
    loop {
        led.set_high();
        Timer::after_millis(1000).await;
        led.set_low();
        Timer::after_millis(1000).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // The arena is static, exclusively owned by the allocator, and initialized once.
    unsafe { HEAP.init(core::ptr::addr_of_mut!(HEAP_MEMORY) as usize, HEAP_SIZE) };
    info!("Burn Atome firmware start!");

    let p = embassy_rp::init(Default::default());

    // GPIO出力の初期化
    let green_led = Output::new(p.PIN_22, Level::Low);

    // 各タスクを生成（spawn）
    spawner.spawn(green_led_task(green_led).unwrap());
    spawner.spawn(inference_task().unwrap());

    info!("All tasks spawned! Main task going idle.");

    // メインタスクは何もしない（他のタスクが動作）
    loop {
        Timer::after_secs(3600).await;
    }
}
