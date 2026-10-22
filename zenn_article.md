---
title: "Rust組み込みで小さなLLMを動かす構成を作ってみた（RP2040 + Burn Flex）"
emoji: "🧠"
type: "tech"
topics: ["rust", "embedded", "rp2040", "burn", "llm"]
published: false
---

## はじめに

Rustの組み込み開発で、マイコン上の推論に挑戦してみました。

今回動かすのは、公開されている約94万パラメータの言語モデル「Atome LM」です。RP2040上で、Rustの機械学習フレームワークBurnのFlexバックエンドを使って推論します。

LEDを点滅させるタスクも同時に動かし、Embassyで組み込み非同期処理を行います。大きなモデルを動かす記事ではありませんが、「マイコンでモデルを動かすには、どこが大変なのか」を確かめる題材になりました。

## 今回の構成

- ボード: Baker link. Dev（RP2040）
- CPU: Cortex-M0+
- Flash: 2 MiB
- SRAM: 256 KiB
- Rust: 1.94.1
- Burn / Burn Flex: 0.21.0
- Embassy: `embassy-rp` 0.10、`embassy-executor` 0.10
- モデル: Atome LM 944,640パラメータ

Atome LMは2-bitの三値重みを使う小さなモデルです。モデルファイルはFlash上に置き、推論時に重みを読み出します。入力は最大16バイト、語彙数は256です。

このモデルは日本語の対話や汎用チャットを目的としたものではありません。今回の固定プロンプトも英語の短い文字列です。

### 全体の構成図

![RP2040上で動くAtome LM、Burn Flex、Embassyの構成](images/atome-lm-architecture.svg)

*モデルはFlashから参照し、推論中の状態はRAMに保持します。推論とLED制御は、同じコア上のEmbassyタスクとして協調的に実行します。*

図の推論タスクでは、Burn Flexの演算と専用Rustループを組み合わせています。推論ブロックの間に`await`を入れ、そのタイミングでLEDタスクにも実行を譲ります。処理時間やヒープ使用量は、`defmt` / RTT経由で確認できる構成です。

:::message
構成図はSVGファイルひとつにまとめ、draw.ioの編集データも埋め込んでいます。同梱の`images/atome-lm-architecture.svg`をdraw.io（diagrams.net）で開くと編集できます。上の相対パスはローカルプレビュー用です。Zennへ投稿する際は、画像をアップロードして発行されたURLに置き換えてください。投稿先がSVGに対応していない場合は、draw.ioからPNGなどに書き出してアップロードしてください。
:::

## Burnとは

[Burn](https://burn.dev/)は、**Rustで書かれた機械学習・深層学習フレームワーク**です。多次元配列である「テンソル」の演算やニューラルネットワークを扱う機能があり、モデルの学習と推論の両方に利用できます。

特徴の1つが、実際に演算する部分を「バックエンド」として切り替えられることです。CPUやGPUなど、実行先に合ったバックエンドを選べます。今回は、Rustで実装されたCPU向けバックエンドの**Burn Flex**を使います。

### `no_std`でも使える

Burnは、対応するバックエンドと機能を選ぶことで、Rustの標準ライブラリ`std`を使わない**`no_std`環境にも対応**しています。そのため、今回のようなOSのないマイコン上でも推論に利用できます。ただし、すべてのバックエンドや機能が`no_std`で使えるわけではありません。

このプロジェクトでは、次のようにデフォルト機能を無効にし、Flexバックエンドを有効にしています。

```toml
burn = { version = "=0.21.0", default-features = false, features = ["flex"] }
burn-flex = { version = "=0.21.0", default-features = false, features = ["critical-section"] }
```

`burn-flex`の`critical-section`機能は、Cortex-M0+でCAS（比較交換）命令を使えないことに対応するための設定です。

なお、**`no_std`は「ヒープを使わない」という意味ではありません**。今回の実装は`alloc`を使い、テンソルや状態キャッシュを動的に確保するため、アロケータとヒープ領域が必要です。この点は、後述の「ヒープの確保」で説明します。

### 今回はBurn Flexと専用Rustループを組み合わせる

ただし、テンソル演算をすべてそのまま使うと、Cortex-M0+では小さな演算の呼び出しや一時テンソルの生成が負担になります。そこで、三値重みのprojectionやAttentionの小さな行列積など、処理時間が気になる部分は専用のRustループにし、LayerNormやsoftmaxなどはBurn Flexで計算する構成にしました。

CエンジンやFFIを呼び出すのではなく、推論処理をRustのコードで実装しているのが今回のポイントです。

## Embassyで推論とLEDを並行して動かす

推論タスクでは、モデルをロードして固定プロンプトを処理します。GPIO22のLEDタスクは、1秒ごとに点灯・消灯します。

```rust
#[embassy_executor::task]
async fn green_led_task(mut led: Output<'static>) {
    loop {
        led.set_high();
        Timer::after_millis(1000).await;
        led.set_low();
        Timer::after_millis(1000).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    let green_led = Output::new(p.PIN_22, Level::Low);

    spawner.spawn(green_led_task(green_led).unwrap());
    spawner.spawn(inference_task().unwrap());

    loop {
        Timer::after_secs(3600).await;
    }
}
```

`inference_task`は、`Once upon a`をトークンごとに処理したあと、greedy方式で4トークンを生成します。推論は8層のブロックに分けて実行し、ブロックごとに`Timer::after_millis(1).await`でEmbassyへ実行を譲ります。

ここで注意したいのは、各ブロックの演算自体は同期処理だという点です。`await`の間は他のタスクが動けますが、ブロック内の演算中にLEDタスクが割り込んで処理を進めるわけではありません。そのため、LEDの周期などのリアルタイム性は保証していません。

## ヒープの確保

RP2040はメモリが限られるため、推論用のヒープを192 KiBに設定しています。モデルの重みはバイナリ全体をRAMへコピーせず、`include_bytes!`でFlash上のデータを参照します。

```rust
const HEAP_SIZE: usize = 192 * 1024;
static HEAP: LlffHeap = LlffHeap::empty();
static mut HEAP_MEMORY: [MaybeUninit<u8>; HEAP_SIZE] =
    [MaybeUninit::uninit(); HEAP_SIZE];

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    unsafe {
        HEAP.init(
            core::ptr::addr_of_mut!(HEAP_MEMORY) as usize,
            HEAP_SIZE,
        )
    };
    // ...
}
```

この192 KiBは推論中に必ずすべて使うという意味ではありません。一方で、リンカ上はヒープ領域として固定で確保するため、スタックなどに使えるRAMの余裕を圧迫します。ビルド時の静的RAM使用量は約194 KiBですが、実機でのスタック高水位やヒープ断片化はまだ測定できていません。

## トークンごとの状態をキャッシュする

プロンプト全体をトークン生成のたびに最初から計算すると、生成するたびに処理量が増えてしまいます。そこで、各層の状態を保持し、次のトークンでは新しいトークンだけを処理するキャッシュ付き生成を実装しました。

キャッシュするものは、主に次の3つです。

- SSMのhidden state
- 因果畳み込みの直近4つの入力
- AttentionのKey / Value

Key / Valueはメモリを抑えるためFP16で保持します。そのため、FP32で全系列を再計算した場合とlogitsが完全一致するわけではありません。

各層は1回の`step()`で処理します。ファームウェア側ではブロックごとの時間や、生成したtoken ID、ヒープ使用量と高水位をRTTへ出力します。

```rust
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
```

## ビルドと実行

このプロジェクトではBaker Linkの既存の`Baker Link: Debug`構成を使います。

1. Baker Link Envを起動し、ホスト側でRunします。
2. Baker link. Devを接続します。
3. VS Codeからビルド・デバッグを実行します。既存構成では、デバッグ開始前に`cargo build`が実行されます。

ファームウェアの書き込みとボードのリセットを伴います。速度を確認するときはreleaseビルドを使います。debugビルドとreleaseビルドでは実行速度が大きく異なるため、debugの計測値を性能評価に使わないようにします。

ビルドが通ることと、実機で推論が完走することは別です。実機ログを確認し、モデル生成やLEDの動作を確かめてください。

## ホストでの検証

実機だけで正しさを確かめるのは難しいため、ホスト上でもNumPyの参照実装と比較します。リポジトリでは次のコマンドで参照生成とテストを実行できます。

```sh
uv run --with numpy==2.4.6 python scripts/reference.py

cargo test --lib --test burn_parity \
  --target x86_64-unknown-linux-gnu \
  --no-default-features --features host-tests -- --nocapture
```

テストでは、1バイトの入力、通常の文字列、最大長の入力、非ASCIIバイトなど8ケースを検証し、256個すべてのlogitsとargmaxを比較します。参照logitsとの許容誤差は`2e-4 + 2e-5 * abs(reference)`です。リポジトリに記録された検証では、最大絶対誤差は約`1.4e-5`でした。

キャッシュ付き生成については、長さ11から16までのprefixで全系列再計算と比較し、argmax一致と最大logits誤差約`1.8e-3`を確認しています。キャッシュのみで最大16トークンを処理したときのホスト側ピークヒープ要求量は約129 KiBでした。

ホスト側の参照生成結果は`Once upon a tim`です。これは参照実装で確認した結果であり、RP2040実機での生成結果ではありません。

これらはホスト上のテスト結果です。RP2040実機で同じRAM使用量や処理速度になることを示すものではありません。

## RAMと速度について

比較対象となるCエンジン版は、`max_seq=16`で固定推論配列を約110 KiBと見積もっています。Burn版の約129 KiBは、キャッシュのみを使ったホスト側のピーク要求量です。測定条件が異なるため、これだけでどちらが省メモリかは結論できません。

さらに、Burn版では192 KiBのヒープ領域を静的に予約しており、ファームウェア全体のRAM余裕は小さくなります。実機のピークヒープ、スタック、処理速度、tokens/secは今後確認する項目です。

以前の全系列再計算方式で取得した実機ログでは、入力長11からの最初のtokenに約317秒かかっていました。現在は状態キャッシュを導入していますが、この古い数値を現在の実装の性能として扱うことはできません。キャッシュ導入後の実機速度は、releaseビルドで改めて計測する必要があります。

## 最後に

今回は、Burn Flexを使った`no_std`推論、Embassyによるタスク実行、限られたRAM上での状態キャッシュを組み合わせて、RP2040で小さな言語モデルを動かす構成を作りました。

モデル自体は小さく、入力長も最大16バイトです。また、現在の実装は固定プロンプトからの生成までで、対話入力やUSB CDCは実装していません。実機での速度やスタック使用量も未確認です。

一方で、ホストでの参照実装との比較や状態キャッシュのテストを用意したことで、実機に載せる前に演算の正しさを確認できるようになりました。今後はrelease版を実機で動かし、推論時間とRAM使用量を測定していきたいと思います。

## 参考

- [Atome LM](https://github.com/TilelliLab/atome-lm)（モデルとライセンスの詳細は配布元および同梱ライセンスを参照）
- [Burn](https://burn.dev/)
- [Embassy](https://embassy.dev/)
- [Rust✖️組み込みで非同期処理（async）をしてみた（Embassy）](https://zenn.dev/baker_link_lab/articles/705de547cc7077)
