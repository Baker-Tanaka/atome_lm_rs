# Atome LM on RP2040 with Burn

公開済みのAtome 944,640パラメータモデルを、CエンジンやFFIを使わず、
Burn Flexの`no_std`演算で実行するファームウェアです。

## 構成

- RP2040 / `thumbv6m-none-eabi` / Flash 2 MiB / リンカ上のRAM 256 KiB。
- Rust 1.94.1に対応するBurn **0.21.0**を固定。
  Burn Bookの現在の0.22系はRust 1.95以降が必要です。
- Cortex-M0+のCAS非対応には`burn-flex/critical-section`を使用。
- `models/atome_944k.bin`をFlashから参照し、各行の三値重みだけをFP32へ展開。
- packed ternary projectionとAttentionの小行列積は、テンソルごとのdispatchを避ける
  専用Rustループで計算し、結果をBurn Flex tensorへ戻します。
  キャッシュ付き生成では因果畳み込み・SSM・top-k Attentionも専用Rustループを使い、
  LayerNormとrouter/attentionのsoftmaxをBurn Flexで計算します。
- コンテキストは最大**24バイト**、バッチは1。生成中はSSM状態、因果畳み込み履歴、
  AttentionのK/Vをトークン間でキャッシュし、新しいトークンだけを各層へ通します。
  K/VキャッシュはRAM削減のためFP16で保持します。
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
新しい配線は不要です。固定プロンプト`Once upon a time, in`（20バイト）を
トークンごとにprefillし、その後greedyで4トークン生成して停止します。
プロンプトと生成分の合計上限は24バイトです。LEDは継続します。
RTTにブロック別時間、token ID、処理時間、現在のヒープ使用量・高水位、
最終バイト列を出力します。結果が有効なUTF-8なら`Generated text:`として文字列も出力し、
不正なUTF-8の場合は文字列化せず、エラー位置をログに出します。
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

キャッシュ使用時、最大24トークンを処理するホストのピークヒープ要求量は約169 KiBで、
192 KiB未満をテストします。ただし、これは実機のアロケータ管理領域、
断片化、スタック高水位を保証する測定ではありません。

RP2040 releaseビルドではFlash約698 KiB、静的RAM約194 KiB
（192 KiBのヒープを含む）です。デバッグ情報はFlash使用量に含めません。
実機での生成結果、スタック使用量、tokens/sec、LED周期は未確認です。
ビルド成功を実機動作の証明として扱わないでください。

## Cエンジン版とのRAM比較

現時点では、**Burn版のRAM消費がC版より少ないとは確認できていません**。
固定的に確保する領域と、推論中に実際に使う領域を区別する必要があります。

| 比較項目 | Cエンジン版 | Burn版 |
|---|---:|---:|
| 推論バッファの概算 | 約160 KiB | キャッシュのみのホスト計測 約169 KiB |
| RP2040用の静的RAM | 約160 KiBの固定バッファが目安 | 約194 KiB（192 KiBのヒープを含む） |
| 実機でのピークRAM | 未測定 | 未測定 |

C版の約160 KiBは、AtomeのCヘッダに定義される固定推論配列を
`max_seq=24`で見積もった値です。公式のCortex-M3実測表にある
`max_seq=64`の約412 KiBからの実測値ではなく、RP2040上での比較測定でもありません。
Burn版の約169 KiB（173,392 bytes）は専用Rust projectionとFP16状態キャッシュ導入後に、
最大24トークンをキャッシュのみで処理した際のホストアロケータ要求ピークです。
実機でのアロケータ管理領域、メモリ断片化、スタックを含まず、
RP2040で同じ値になることは保証しません。

## 推論速度の改善と実機計測

初期のBurnテンソル実装では、出力1行ごとにFlex tensorの要素積と縮約を
実行していました。行列全体の処理としては小さい操作が多く、特にCortex-M0+では
フレームワークdispatchと一時tensor生成のコストが支配的になります。
現在はpacked三値重みの行列ベクトル積とAttentionの小行列積をRustループにまとめ、
演算ごとに一度だけBurn tensorへ変換します。モデルの他の演算はBurn Flexを維持します。

変更前のRP2040ログでは、入力長11からの最初のtokenに約317秒、
その後は約7秒ずつ増加し、ヒープ高水位は約69 KiBから約4.6 KiBずつ増えていました。
各生成で全入力系列を先頭から再計算するため、系列が伸びるほど処理時間と一時領域が
増加します。このヒープ増加は系列長による一時領域の増加と整合しますが、
それだけでリークの有無を断定できないため、現在値`heap_used`と高水位を合わせて確認します。

Rustループへの変更後、full-context方式のホスト上の8ケース検証は約0.72秒で完了し、
logits最大誤差は約`9.1e-6`、ピーク要求量は約89 KiBでした。
変更前のホスト比較は約1.5秒でしたが、これはRP2040上の速度比を示しません。
**変更後の実機速度はまだ測定していません。** 新しいblock別時間と`heap_used`を
使ってRP2040上で再計測し、改善を確認してください。

### トークン単位の状態キャッシュ

full-context方式ではトークンを1つ生成するたびに、プロンプトの全トークンを
全層で再計算していました。現在は各層のSSM hidden stateと直近4個の畳み込み入力、
Attentionの過去K/Vを保持します。次トークンでは現在トークンだけを処理します。
K/VはFP16保存のため、状態を保持しないFP32全系列実行と数値的に完全一致ではありません。

ホスト検証では長さ11から24までの全prefixについて、cached logitsを全系列再計算と
比較し、argmax一致・logits最大誤差約`1.8e-3`を確認しました。
K/Vキャッシュのため使用メモリは増えます。キャッシュだけのピーク要求量が
192 KiB未満であることをホストテストで確認します。
キャッシュと全系列再計算を同時に行う比較テストは、両方の作業領域を使います。
releaseホストでの今回の検証では、長さ11から24までのcached処理合計が約111 ms、
同じprefixを全系列再計算した合計が約297 msでした。
これはホストでの参考値で、RP2040での速度比を示しません。

初期実機ログの約317秒はキャッシュ導入前の数値です。新方式の実機速度とヒープ高水位、
最初のprefill時間および2トークン目以降の時間は、release版で改めて測定してください。
K/V領域はgeneration開始時に最大コンテキスト分を一括予約するため、
生成途中でヒープ高水位がtokenごとに増え続けないことも確認してください。

したがって、現状はBurn版の推論中ヒープ使用量がC版の推論バッファ見積もりを
下回る可能性はありますが、リンカ上では192 KiBのヒープを固定予約するため、
**ファームウェア全体のRAM余裕はC版より小さくなっています**。
実機でヒープ・スタックの高水位を確認し、問題がなければヒープ予約を縮小するのが
次の見直し候補です。

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
