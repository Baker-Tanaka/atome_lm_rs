# Atome LM on RP2040 with Burn

公開済みのAtome 944,640パラメータモデルを、CエンジンやFFIを使わず、
Burn Flexの`no_std`演算で実行するファームウェアです。

## 構成

- RP2040 / `thumbv6m-none-eabi` / Flash 2 MiB / リンカ上のRAM 256 KiB。
- Rust 1.94.1に対応するBurn **0.21.0**を固定。
  Burn Bookの現在の0.22系はRust 1.95以降が必要です。
- Cortex-M0+のCAS非対応には`burn-flex/critical-section`を使用。
- `models/atome_944k.bin`をFlashから参照し、各行の三値重みだけをFP32へ展開。
- 行列積はBurnのブロードキャスト積と`sum_dim`で計算。
  Flexの通常の`matmul`はGEMM作業領域が大きく、RP2040には不適合でした。
- LayerNorm、因果畳み込み、SSM、top-k因果Attention、Router、残差接続もBurn演算。
- コンテキストは最大**16バイト**、バッチは1。各推論で全系列を再計算します。
  SSM状態を別の入力へ持ち越しません。
- ヒープは192 KiB。各ブロック間でEmbassyへ実行を譲ります。
  ブロック内は同期演算なので、LED周期などのリアルタイム性は保証しません。

## ビルド・実行

```sh
cargo build
cargo build --release
```

Baker Linkの既存の`Baker Link: Debug`構成からdebug版を実行できます。
ホスト側でBaker Link EnvをRunし、対象ボードを接続してください。
書き込み・リセットを伴います。速度計測にはrelease版を使用してください。

GPIO22の緑LEDタスクを並行して動かします。
新しい配線は不要です。入力は初期検証用の固定プロンプト`Once upon a`で、
greedy生成を4ステップ実行して停止します。LEDは継続します。
RTTにブロック完了、token ID、推論時間、ヒープ高水位、最終バイト列を出力します。
対話入力やUSB CDCはまだ実装していません。

## 検証

```sh
uv run --with numpy==2.4.6 python scripts/reference.py
cargo test --lib --test burn_parity \
  --target x86_64-unknown-linux-gnu --no-default-features --features host-tests -- --nocapture
cargo fmt --all -- --check
cargo size --release --bin atome_lm_rs -- -A
```

従来のHALベースの組み込みテストは、Embassyと割り込みベクタを混在させずにビルドします。
以下はビルドのみで、実機テストの実行ではありません。

```sh
cargo test --no-default-features --features hal-rt --test example_test --no-run
```

NumPyスクリプトは同梱blobを独立にデコードし、参照logitsを生成します。
Pythonの元実装と同じ3経路の数式を使用しますが、元のPyTorchチェックポイント自体
との比較ではありません。Cエンジンは参照結果生成にも使いません。

テストは1バイト、通常の文字列、最大16バイト、非ASCIIバイト、
4ステップ生成中の入力を含む8ケースで256個すべてのlogitsとargmaxを比較します。
許容誤差は`2e-4 + 2e-5 * abs(reference)`。確認時の最大絶対誤差は約`1.4e-5`でした。
参照の生成結果は`Once upon a tim`です。

ホストのアロケータ計測では要求サイズのピークが約92 KiBで、
192 KiB未満をテストします。ただし、これは実機のアロケータ管理領域、
断片化、スタック高水位を保証する測定ではありません。

RP2040 releaseビルドではFlash約698 KiB、静的RAM約194 KiB
（192 KiBのヒープを含む）です。デバッグ情報はFlash使用量に含めません。
実機での生成結果、スタック使用量、tokens/sec、LED周期は未確認です。
ビルド成功を実機動作の証明として扱わないでください。

## モデル出典

- [TilelliLab/atome-lm](https://github.com/TilelliLab/atome-lm)
- 固定リビジョン: `be1218672e9adb5656c877c2dbf7b05e55d96d10`
- 元ファイル: `checkpoints/atome_944k.bin`（276,655 bytes、`ATOME01`、2-bit三値形式）
- SHA-256: `fdf8a6b69eacc5e4834e488759593198e482399887fce2c5b048a599844ae2f5`
- モデル設定: `d_model=256`, `n_layers=8`, `d_head=64`, `kernel_size=5`,
  `top_k=4`, `vocab_size=256`
- モデルのライセンス: Apache-2.0。[同梱ライセンス](models/LICENSE)を参照。
  モデルの権利はAtome LM contributorsに帰属します。

`ATOME01`には構成メタデータがないため、この実装は上記構成専用です。
別モデルの差し替えにはコードと参照fixtureの更新が必要です。
日本語対応や汎用チャット品質を保証するモデルではありません。
