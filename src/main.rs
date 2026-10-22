#![no_std]
#![no_main]

use atome_lm_rs::{Atome, CONTEXT, MODEL_BYTES};
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

#[embassy_executor::task]
async fn inference_task() {
    let model = Atome::load(MODEL_BYTES).unwrap_or_else(|e| {
        defmt::panic!("Model load failed: {}", Debug2Format(&e));
    });
    let prompt = b"Once upon a time, in";
    let mut tokens = [0u8; CONTEXT];
    tokens[..prompt.len()].copy_from_slice(prompt);
    let mut generation = model.cached_generation(CONTEXT).unwrap_or_else(|e| {
        defmt::panic!("Invalid generation limit: {}", Debug2Format(&e));
    });
    info!(
        "Burn Flex / cached Atome 944K: context={} heap={}",
        CONTEXT, HEAP_SIZE
    );
    let mut last_elapsed_us = 0;
    for (position, &token) in prompt.iter().enumerate() {
        let started = Instant::now();
        generation.start_token(token).unwrap_or_else(|e| {
            defmt::panic!("Invalid prompt token: {}", Debug2Format(&e));
        });
        for _ in 0..8 {
            generation.step().unwrap_or_else(|e| {
                defmt::panic!("Cached block inference failed: {}", Debug2Format(&e));
            });
            Timer::after_millis(1).await;
        }
        last_elapsed_us = started.elapsed().as_micros();
        info!("Prompt token {} processed us={}", position, last_elapsed_us);
    }
    for index in 0..4 {
        let logits = generation.logits().unwrap_or_else(|e| {
            defmt::panic!("Cached inference logits unavailable: {}", Debug2Format(&e));
        });
        let mut next = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[next] {
                next = i;
            }
        }
        tokens[prompt.len() + index] = next as u8;
        let peak = critical_section::with(|cs| HEAP_PEAK.borrow(cs).get());
        let used = HEAP.used();
        info!(
            "TOK {} id={} us={} heap_used={} heap_peak={}",
            index, next, last_elapsed_us, used, peak
        );
        if index < 3 {
            let started = Instant::now();
            generation.start_token(next as u8).unwrap_or_else(|e| {
                defmt::panic!("Invalid generated token: {}", Debug2Format(&e));
            });
            for layer in 0..8 {
                let block_started = Instant::now();
                let complete = generation.step().unwrap_or_else(|e| {
                    defmt::panic!("Cached block inference failed: {}", Debug2Format(&e));
                });
                info!(
                    "Block {} complete={} us={}",
                    layer,
                    complete,
                    block_started.elapsed().as_micros()
                );
                Timer::after_millis(1).await;
            }
            last_elapsed_us = started.elapsed().as_micros();
        }
    }
    info!(
        "Generation complete: bytes={=[u8]}",
        &tokens[..prompt.len() + 4]
    );
    match core::str::from_utf8(&tokens[..prompt.len() + 4]) {
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
