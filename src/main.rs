// embassy_multitask.rs - Embassyを使った複数タスクの並行実行
//
// Embassyの非同期タスク機能を使って、複数のタスクを並行実行する例です。
// 各タスクは独立して動作し、async/awaitにより協調的にスケジューリングされます。
//
// タスク構成:
//   - green_led_task: GPIO22の緑LEDを1秒間隔で点滅
//   - red_led_task:   GPIO20の赤LEDを2秒間隔で点滅
//   - logger_task:    5秒ごとにステータスをログ出力
//
// 実行方法: cargo run --no-default-features --features embassy --example embassy_multitask

#![no_std]
#![no_main]

use defmt::*;
use defmt_rtt as _;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_time::Timer;
use panic_probe as _;

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

#[embassy_executor::task]
async fn red_led_task(mut led: Output<'static>) {
    info!("Red LED task started (2s interval)");
    loop {
        led.set_high();
        Timer::after_millis(2000).await;
        led.set_low();
        Timer::after_millis(2000).await;
    }
}

#[embassy_executor::task]
async fn logger_task() {
    info!("Logger task started (5s interval)");
    let mut count: u32 = 0;
    loop {
        Timer::after_secs(5).await;
        count += 1;
        info!("System running... uptime: {}s", count * 5);
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    info!("Embassy multitask example start!");

    let p = embassy_rp::init(Default::default());

    // GPIO出力の初期化
    let green_led = Output::new(p.PIN_22, Level::Low);
    let red_led = Output::new(p.PIN_20, Level::Low);

    // 各タスクを生成（spawn）
    spawner.spawn(green_led_task(green_led).unwrap());
    spawner.spawn(red_led_task(red_led).unwrap());
    spawner.spawn(logger_task().unwrap());

    info!("All tasks spawned! Main task going idle.");

    // メインタスクは何もしない（他のタスクが動作）
    loop {
        Timer::after_secs(3600).await;
    }
}
