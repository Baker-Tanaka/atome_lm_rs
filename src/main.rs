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
    let prompt = b"Once upon a";
    let mut tokens = [0u8; CONTEXT];
    tokens[..prompt.len()].copy_from_slice(prompt);
    let mut len = prompt.len();
    info!(
        "Burn Flex / Atome 944K: context={} heap={}",
        CONTEXT, HEAP_SIZE
    );
    for index in 0..4 {
        let started = Instant::now();
        let mut inference = model.start(&tokens[..len]).unwrap_or_else(|e| {
            defmt::panic!("Invalid inference input: {}", Debug2Format(&e));
        });
        for layer in 0..8 {
            let complete = inference.step();
            info!("Block {} complete={}", layer, complete);
            Timer::after_millis(1).await;
        }
        let logits = inference.logits().unwrap_or_else(|e| {
            defmt::panic!("Inference failed: {}", Debug2Format(&e));
        });
        let mut next = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[next] {
                next = i;
            }
        }
        tokens[len] = next as u8;
        len += 1;
        let peak = critical_section::with(|cs| HEAP_PEAK.borrow(cs).get());
        info!(
            "TOK {} id={} us={} heap_peak={}",
            index,
            next,
            started.elapsed().as_micros(),
            peak
        );
    }
    info!("Generation complete: bytes={=[u8]}", &tokens[..len]);
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
