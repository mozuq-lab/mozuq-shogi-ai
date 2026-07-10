# 将棋AI開発ガイド（蒸留Transformer）

## プロジェクト概要

水匠5の評価値を教師として蒸留した軽量Transformer評価ネットワークを構築する。
Value Network（局面→評価値）のみを実装対象とし、ポリシーネットは将来の拡張とする。

## 開発環境

| 環境 | 用途 | 備考 |
|------|------|------|
| Mac (Apple Silicon) | 開発、デバッグ、データ生成 | 現在の開発環境 |
| Windows (RTX 4070 Super) | 本格学習 (Phase 3) | CUDA使用 |

データの受け渡しはクラウドストレージを使用する。

## 開発フェーズと進捗

- [x] Phase 0: 環境構築
- [x] Phase 1: 教師データ生成
- [x] Phase 2: モデル設計・実装
- [x] Phase 3: 学習スクリプト実装（本格学習はWindows環境で実施）
- [x] Phase 4: エンジン組み込み（1手読み）
- [ ] Phase 5: モデル改良（詳細は末尾のタスクリストを参照）

## ディレクトリ構成

```
shogi-ai/
├── external/shogi-cli/     # 将棋エンジン (git submodule)
│   ├── suisho5/            # 水匠5
│   │   ├── YaneuraOu-mac   # Mac用エンジン
│   │   ├── YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe  # Windows用エンジン
│   │   └── eval/nn.bin     # 評価関数
│   └── hao/                # Hao
│       ├── YaneuraOu-mac   # Mac用エンジン
│       ├── YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe  # Windows用エンジン
│       └── eval/nn.bin     # 評価関数
├── shogi_utils/            # 将棋関連ユーティリティ
│   ├── __init__.py
│   └── usi_engine.py       # USIエンジンラッパー ✓
├── models/                 # モデル定義 ✓
│   ├── __init__.py
│   ├── value_transformer.py  # Transformerモデル
│   ├── sfen_parser.py        # SFENパーサー
│   ├── dataset.py            # PyTorch Dataset
│   └── features.py           # 拡張特徴量（利き、玉距離等）
├── tests/                  # テスト ✓
│   └── test_models.py
├── data/raw/               # 生成データ (.gitignore対象)
├── tools/
│   └── gen_dataset.py      # データ生成スクリプト ✓
├── train/                  # 学習スクリプト ✓
│   └── train.py            # Value Network学習
├── engine/                 # USIエンジン ✓
│   ├── evaluator.py        # モデル評価器
│   └── usi_server.py       # USIプロトコルサーバー
├── scripts/
│   ├── usi_test.py         # USI疎通確認スクリプト ✓
│   ├── to_kif.py           # KIF形式変換スクリプト ✓
│   ├── move_agreement.py   # 指し手一致率測定 ✓
│   └── selfplay_match.py   # モデル同士の自己対局 ✓
├── env/
│   └── README.md           # セットアップ手順
└── reports/                # 評価レポート
```

## 実装済み機能

### USIエンジンラッパー (`shogi_utils/usi_engine.py`)

```python
from shogi_utils import USIEngine, get_engine_path, get_default_engine_path

# 水匠5を使用（デフォルト）
with USIEngine(get_default_engine_path()) as engine:
    engine.init_usi()
    engine.set_option("USI_OwnBook", False)
    engine.is_ready()
    engine.set_position(moves=["7g7f", "3c3d"])
    result = engine.go(depth=10)  # or movetime=500
    print(result.bestmove, result.score_cp)

# Haoを使用
with USIEngine(get_engine_path("hao")) as engine:
    engine.init_usi()
    # ...
```

### データ生成スクリプト (`tools/gen_dataset.py`)

将棋エンジン同士の自己対局で教師データを生成する。

```bash
# 深さ10で1対局（デフォルト、水匠5使用）
python tools/gen_dataset.py -n 1

# 深さ15で10対局
python tools/gen_dataset.py -n 10 --depth 15 -o data/raw/depth15.jsonl

# 時間指定（500ms）で生成
python tools/gen_dataset.py -n 10 --movetime 500

# 並列実行（4ワーカー）
python tools/gen_dataset.py -n 100 --depth 10 --workers 4

# Haoエンジンを使用
python tools/gen_dataset.py -n 10 --engine-type hao
python tools/gen_dataset.py -n 10 --depth 15 --engine-type hao -o data/raw/hao_depth15.jsonl

# 弱いAIとの対局データ生成
python tools/gen_dataset.py -n 100 --weak-side white --weak-prob 0.5
python tools/gen_dataset.py -n 100 --weak-side alternate --weak-prob 0.3 --workers 4
```

#### 弱いAI対局オプション

強いAI vs 弱いAI の対局データを生成。弱い側がランダム手を指すことで、評価値に差がつく局面を収集できる。

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--weak-side` | なし | 弱い側 (`black`=先手, `white`=後手, `alternate`=交互, `both`=両方) |
| `--weak-prob` | 0.5 | 弱い側がランダム手を指す確率 (0.0〜1.0) |
| `--random-opening` | 32 | 序盤のランダム手数（多様性のため） |
| `--random-type` | `full` | ランダム手の生成方式 (`engine`=go random, `full`=完全ランダム) |
| `--record-pv-leaf` | 無効 | PV末端局面（静止局面）のレコードも記録 |
| `--multipv` | 1 | 2以上で上位N手の子局面レコードも記録 |

#### データ密度を上げるオプション

- `--record-pv-leaf`: 探索評価値は読み筋（PV）末端の静的評価に対応するため、
  PVを適用した静止局面にもラベルを付ける。1手読みで使う評価関数の学習に有効。
  レコードには `"source": "pv_leaf"` が付く。
- `--multipv N`: 1回の探索で上位N手を取得し、順位2以下の手の子局面に評価値
  （符号反転済み）を付与。「良い手と悪い手の評価差」を効率的に収集できる。
  レコードには `"source": "multipv"` が付く。
  さらに親局面レコードに候補手リスト `candidates`
  （`[{"move", "score_cp", "rank"}, ...]`、手番側視点）が付与される。
  これはranking損失（`train.py --ranking-weight`）とオフライン一致率
  （`move_agreement.py --offline`）の教師として使う。
- 両方有効にすると1対局あたりのレコード数が約4倍になる（depth 5, MultiPV 3実測）。

**ランダム手生成方式の違い：**

| 方式 | コマンド | 評価値 | 特徴 |
|------|---------|--------|------|
| engine | `go random` | なし（0になる） | エンジン内蔵のランダム、高速 |
| full | 通常探索後に手を差し替え | あり | 全ての局面で評価値を記録 |

※ `engine`方式で生成した既存データはランダム手の局面に評価値0のダミーラベルが
記録されている（学習に有害）。学習時に`--drop-zero-cp`で除外するか、
`--random-type full`でデータを再生成すること。

**動作の違い：**

- `--weak-side`なし: 両側とも`--random-opening`手数だけランダム、以降は通常探索
- `--weak-side black/white/alternate`:
  - 強い側: `--random-opening`手数だけランダム（多様性）、以降は通常探索
  - 弱い側: 全局面で`--weak-prob`に従ってランダム手を選択
- `--weak-side both`: 両側とも`--weak-prob`に従ってランダム手を選択（評価値は正確に記録）

※ `--random-type full`（デフォルト）を使用すると、序盤のランダム手でも評価値が記録される。
※ 評価値は常に深い探索から取得されるため、正確なラベルが付与される。

#### エンジン選択

| オプション | エンジン | 説明 |
|-----------|---------|------|
| `--engine-type suisho5` | 水匠5 | デフォルト |
| `--engine-type hao` | Hao | 別の評価関数 |
| `--engine /path/to/engine` | 任意 | パス直接指定 |

#### 出力形式 (JSONL)

```json
{"sfen": "startpos", "score_cp": 37, "ply": 0, "game_id": 0, "result": "white_win"}
{"sfen": "startpos moves 2g2f", "score_cp": -136, "ply": 1, "game_id": 0, "result": "white_win"}
```

| フィールド | 説明 |
|-----------|------|
| `sfen` | 局面（startpos + 指し手列） |
| `score_cp` | 評価値（centipawn、手番側視点） |
| `ply` | 手数 |
| `game_id` | 対局ID |
| `result` | 対局結果 (`black_win`, `white_win`, `draw`) |
| `source` | レコード由来（省略=対局局面, `pv_leaf`, `multipv`） |
| `candidates` | MultiPV候補手リスト（`--multipv` 2以上時、親局面のみ） |

### KIF形式変換スクリプト (`scripts/to_kif.py`)

JSONLファイルからKIF形式の棋譜ファイルを生成する。評価値をShogiGUI互換形式でコメントとして埋め込む。

```bash
# 棋譜をKIF形式で表示
python scripts/to_kif.py data/raw/dataset.jsonl 0

# ファイルに出力
python scripts/to_kif.py data/raw/dataset.jsonl 0 -o game0.kif

# 評価値なしで出力
python scripts/to_kif.py data/raw/dataset.jsonl 0 --no-scores

# 対局一覧を表示
python scripts/to_kif.py data/raw/dataset.jsonl --list
```

#### 出力例

```
手数----指手----
   1 ▲２六歩
**解析  評価値 112
   2 △３四歩
**解析  評価値 127
```

評価値はShogiGUI形式（`**解析  評価値 数値`）で出力され、ShogiGUIでグラフ表示が可能。

## 水匠5の使い方

```bash
cd external/shogi-cli/suisho5
echo -e "usi\nisready\nposition startpos\ngo depth 10\nquit" | ./YaneuraOu-mac
```

### パフォーマンス

| 環境 | スレッド数 | nps |
|------|-----------|-----|
| Mac Apple Silicon | 1 | 約 1.0M |
| Mac Apple Silicon | 8 | 約 4.9M |
| Windows Ryzen 7 9700X (AVX2) | 1 | 約 2.4M |
| Windows Ryzen 7 9700X (AVX2) | 8 | 約 15.7M |
| Windows Ryzen 7 9700X (AVX2) | 16 | 約 21.4M |

※ Windows環境ではAVX2版を使用（AVX512VNNI版はRyzen 9000シリーズ非対応）

### Value Network (`models/`)

```python
from models import ValueTransformer, ShogiValueDataset, collate_fn
from torch.utils.data import DataLoader

# モデル作成
model = ValueTransformer(
    d_model=256,
    n_heads=4,
    n_layers=4,
    ffn_dim=512,
    dropout=0.1,
)
# パラメータ数: 約215万

# データセット読み込み
dataset = ShogiValueDataset("data/raw/hao_trial_500_depth10.jsonl")
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# 推論
batch = next(iter(loader))
value, outcome = model(batch["board"], batch["hand"], batch["turn"])
# value: (batch, 1) 評価値 [-1, 1]
# outcome: (batch, 1) 勝率予測 [0, 1]
```

#### 入力表現

- 81マス（盤面）+ 14トークン（持ち駒）= 95トークン
- 駒種埋め込み（29種: 空、先手駒14種、後手駒14種）
- 持ち駒は駒種埋め込み + 枚数埋め込み
- 手番埋め込みを全トークンに加算

#### 拡張特徴量（オプション）

`--use-features` フラグで有効化。各マスに10次元の追加特徴量を付与：

| 特徴量 | 次元 | 説明 |
|--------|------|------|
| attack_map | 2 | 先手/後手の利き（0/1） |
| king_distance | 2 | 先手玉/後手玉からのチェビシェフ距離 |
| piece_value | 1 | 駒価値（先手+、後手-） |
| control | 1 | 支配度（利きの差をtanh正規化） |
| king_safety | 4 | 玉安全度（味方駒数、敵利き数 × 先後） |

```python
# 拡張特徴量を使用したモデル
model = ValueTransformer(use_features=True)

# 拡張特徴量を使用した学習
PYTHONPATH=. python train/train.py --data data.jsonl --use-features
```

### 勝敗補助損失

モデルは評価値に加えて勝率予測も出力。学習時に補助損失として使用：

```bash
# 補助損失の重みを調整（デフォルト: 0.1）
PYTHONPATH=. python train/train.py --data data.jsonl --aux-loss-weight 0.2
```

損失 = MSE(評価値) + weight × BCE(勝率)

※ pv_leaf/multipv分岐局面には本譜の勝敗が当てはまらない（実際に進行した
本譜の勝敗が一律コピーされている）ため、`source`フィールド付きレコードは
勝敗損失から自動で除外される（outcome_weight=0）。`--target-mode wdl`でも
分岐局面は勝敗をブレンドせず評価値勝率のみを使う。

### Pairwise ranking損失

candidatesフィールド付きデータ（`gen_dataset.py --multipv` 2以上で生成）を使い、
同一親局面の候補手ペアについて「良い手を指した子局面の評価が悪い手より高くなる」
関係を補助損失として学習する（Bonanzaの比較学習と同系統）。
1手読みは子局面評価値のargmaxで手を選ぶため、候補手間の順序学習は
手選びの精度に直結する。

```bash
PYTHONPATH=. python train/train.py --data data_mpv.jsonl --ranking-weight 0.3
```

- 損失 = softplus(v_better − v_worse)（RankNet形式。vは子局面の手番側視点なので
  良い手側が小さくなるべき）
- 評価値差が `--ranking-min-gap`（デフォルト30cp）未満のペアはノイズとして除外
- 訓練/検証はメインデータと同じ対局単位分割を共有（リーク防止）
- 検証時はペア正答率（pair_accuracy）をエポックごとにログ出力

### オフライン指し手一致率の自動計測

`--agreement-data` にcandidatesフィールド付きJSONLを指定すると、
定期checkpoint保存時と学習終了時に、rank 1の手との一致率を
エンジン起動なしで計測してログJSONに記録する（val_lossと強さの乖離を早期検出）。

```bash
PYTHONPATH=. python train/train.py \
    --data data_mpv.jsonl \
    --agreement-data data_mpv.jsonl --agreement-limit 200
```

計測はEvaluator（推論と同じ経路）を通すため、学習時と推論時の
前処理の食い違いも検出できる。

### 推奨オプションを有効にした学習

```bash
PYTHONPATH=. python train/train.py \
    --data data/raw/large_dataset.jsonl \
    --use-features \
    --aux-loss-weight 0.1 \
    --normalize-turn \
    --augment-flip \
    --cp-noise 7.5 \
    --cp-clamp 2000 \
    --value-loss huber \
    --ema-decay 0.999 \
    --num-workers 4 \
    --epochs 100 \
    --batch-size 512
```

- `--drop-zero-cp` は旧方式（`--random-type engine`）で生成したデータを使う場合のみ追加。
- `--cp-clamp` は `--cp-filter-threshold`（除外方式）の上位互換。併用時はfilter→clampの順に適用。
- モデル構造オプション（`--use-king-relative` / `--use-2d-pos` / `--use-discrete-hand`）と
  `--target-mode wdl` は、`scripts/move_agreement.py` で効果を測りながら1つずつ追加を推奨。

#### 出力

- スカラー値 [-1, 1]（勝率近似、tanh正規化）

#### 評価値の正規化

```python
from models import normalize_cp, denormalize_cp

# centipawn → [-1, 1]
value = normalize_cp(500)   # → 0.395

# [-1, 1] → centipawn
cp = denormalize_cp(0.395)  # → 500
```

## Phase 3: 学習

### 学習スクリプト (`train/train.py`)

```bash
# Mac環境での動作確認（小規模）
PYTHONPATH=. python train/train.py \
    --data data/raw/hao_trial_500_depth10.jsonl \
    --epochs 5 --batch-size 64 --device auto

# Windows環境での本格学習
PYTHONPATH=. python train/train.py \
    --data data/raw/large_dataset.jsonl \
    --epochs 100 --batch-size 512 --device cuda

# 学習再開
PYTHONPATH=. python train/train.py \
    --data data/raw/large_dataset.jsonl \
    --resume checkpoints/epoch_0050.pt
```

#### 主なオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--data` | 必須 | データファイルパス |
| `--epochs` | 100 | エポック数 |
| `--batch-size` | 512 | バッチサイズ |
| `--lr` | 3e-4 | 学習率 |
| `--device` | auto | デバイス（auto/cuda/mps/cpu） |
| `--output-dir` | checkpoints | 出力ディレクトリ |
| `--resume` | - | 再開するチェックポイント |
| `--val-split` | 0.1 | 検証データ割合（対局単位で分割） |
| `--use-features` | - | 拡張特徴量を使用 |
| `--aux-loss-weight` | 0.1 | 勝敗補助損失の重み |
| `--drop-zero-cp` | - | score_cp==0の局面を除外（旧方式データのダミーラベル対策） |
| `--cp-scale` | 1200 | 評価値正規化のスケール |
| `--variance-reg-weight` | 0 | 分散正則化の重み（通常は不要） |
| `--ema-decay` | 0 | EMA減衰率（0=無効、推奨: 0.999）。有効時はEMA重みで検証・保存 |
| `--value-loss` | mse | 評価値損失（`mse` / `huber`） |
| `--huber-delta` | 0.5 | Huber lossの遷移点 |
| `--cp-clamp` | なし | 評価値を±この値に丸める（除外せず学習に残す） |
| `--target-mode` | cp | ターゲット空間（`cp`=tanh正規化, `wdl`=勝率でelmoブレンド） |
| `--wdl-scale` | 600 | cp→勝率変換のシグモイドスケール |
| `--wdl-lambda` | 0.5 | elmoブレンドの教師評価値の重み（0〜1） |
| `--use-king-relative` | - | 玉相対位置埋め込み（HalfKP相当の帰納バイアス） |
| `--use-2d-pos` | - | 2D位置埋め込み（学習可能な段・筋埋め込み） |
| `--use-discrete-hand` | - | 持ち駒枚数を離散埋め込みで表現 |
| `--ranking-weight` | 0 | pairwise ranking損失の重み（0=無効、candidatesフィールド必須） |
| `--ranking-min-gap` | 30 | rankingペアの最小評価値差（cp、これ未満はノイズとして除外） |
| `--agreement-data` | なし | オフライン一致率計測用のcandidatesフィールド付きJSONL |
| `--agreement-limit` | 200 | 一致率計測の局面数上限 |

#### 評価値ターゲットの選択

- `--target-mode cp`（デフォルト）: `tanh(score_cp / cp_scale)` を回帰
- `--target-mode wdl`: 教師評価値を勝率 `sigmoid(score_cp / wdl_scale)` に変換し、
  実際の勝敗結果と `--wdl-lambda` でブレンド（elmo式）。
  `target = λ × 評価値勝率 + (1−λ) × 勝敗` を [-1, 1] にマップして回帰。
  推論時はcheckpointの `target_mode` から自動で逆変換される。

#### 出力ファイル

```text
checkpoints/
├── best.pt           # ベストモデル（最小val_loss）
├── epoch_XXXX.pt     # 定期保存（--save-every間隔）
├── final.pt          # 最終モデル
└── log_YYYYMMDD_HHMMSS.json  # 学習ログ
```

### ハイパーパラメータ

- データセット: 10万〜100万局面
- 訓練:バリデーション = 9:1（対局単位で分割し、同一対局の局面がtrain/valに跨るリークを防止）
- Optimizer: AdamW
- 学習率: 3e-4（5エポックwarmup後、コサインアニーリング）
- バッチサイズ: 512〜1024
- 損失関数: MSE(評価値) + 0.1 × BCE(勝率)

## Phase 4: USIエンジン

### USIエンジンの起動

```bash
# ラッパースクリプトで起動（将棋GUIから登録する場合はこちら）
./shogi-ai-engine.sh

# 直接起動
PYTHONPATH=. python engine/usi_server.py --model checkpoints/best.pt

# デバイス指定
PYTHONPATH=. python engine/usi_server.py --model checkpoints/best.pt --device cpu
```

#### 将棋GUIでの使用

将棋所やShogiGUIなどのUSI対応GUIで `shogi-ai-engine.sh` をエンジンとして登録すると対局可能。

#### 対応USIコマンド

| コマンド | 説明 |
|---------|------|
| `usi` | エンジン情報を返す |
| `isready` | モデルを読み込み、`readyok`を返す |
| `position startpos [moves ...]` | 初期局面から指定の手を適用 |
| `position sfen <sfen> [moves ...]` | SFEN局面から指定の手を適用 |
| `go` | 最善手を探索（1手読み） |
| `quit` | 終了 |

#### 評価器 (`engine/evaluator.py`)

```python
from engine.evaluator import Evaluator

# 評価器を初期化
evaluator = Evaluator("checkpoints/best.pt", device="auto")

# SFEN文字列で評価
score = evaluator.evaluate_sfen("startpos moves 7g7f 3c3d")

# python-shogiのBoardで評価
import shogi
board = shogi.Board()
score = evaluator.evaluate_board(board)

# 最善手を探索（1手読み）
best_move, score = evaluator.find_best_move(board)
```

### 依存ライブラリ

- `python-shogi`: 合法手生成に使用

### 制限事項

- 現在は1手読みのみ（各合法手の評価値を比較）
- 1手詰めはルールで確定評価（NN評価に依らず+30000で即選択）
- 千日手・連続王手・入玉宣言の確定評価は未対応
- 探索アルゴリズム（αβ等）は未実装

## 評価関数の計測

val_lossより直接的に「強さ」を測る2つのスクリプト。改善を入れるたびに
効果を数字で確認する。

### 指し手一致率 (`scripts/move_agreement.py`)

モデルの1手読みが選ぶ手と教師エンジンの最善手の一致率を測定。
序盤（ply<30）/中盤（30-79）/終盤（80+）別の集計付き。

```bash
PYTHONPATH=. python scripts/move_agreement.py \
    --model checkpoints/best.pt \
    --data data/raw/dataset.jsonl \
    --depth 10 --limit 200 --output reports/agreement.json

# オフラインモード: candidatesフィールド（MultiPV記録）のrank 1を教師にする。
# エンジン不要で高速。MultiPV上位手に入った率（MultiPV hit）も出力。
PYTHONPATH=. python scripts/move_agreement.py \
    --model checkpoints/best.pt \
    --data data/raw/dataset_mpv.jsonl \
    --offline --limit 200
```

学習中の自動計測は `train.py --agreement-data` を参照。

### モデル同士の自己対局 (`scripts/selfplay_match.py`)

2つのcheckpointを1手読み同士で対局させ、勝敗と推定レート差を出す。
ランダム開始局面から先後入替のペア対局で公平性を確保。

```bash
PYTHONPATH=. python scripts/selfplay_match.py \
    --model-a checkpoints/new.pt \
    --model-b checkpoints/old.pt \
    --games 20 --output reports/match.json
```

## コーディング規約

- 型ヒント必須
- docstring: Google style
- テスト: pytest
- Python 3.10+互換（`from __future__ import annotations`使用）

---

## Phase 5: モデル改良タスク

### 優先度A: 即効性が高く低リスク（1-2時間）

- [x] **Gradient Clipping** - 勾配爆発防止
  - `train.py`: `--grad-clip-norm`オプション（デフォルト: 1.0）

- [x] **Label Smoothing** - 勝敗予測の過信防止
  - `train.py`: `--label-smoothing`オプション（デフォルト: 0.05）

- [x] **評価値ノイズ付与** - 過学習抑制
  - `train.py`: `--cp-noise`オプション（デフォルト: 0、推奨: 5-10）

- [x] **簡単局面フィルタ** - 極端な評価値を間引き
  - `train.py`: `--cp-filter-threshold`オプション（推奨: 1500）

### 優先度B: データ効率の最大化（半日〜1日）

- [x] **盤面正規化（手番反転）** - 常に先手視点に統一
  - `train.py`: `--normalize-turn`オプション
  - 効果: 後手番の局面を先手視点に変換し、モデルの学習を効率化

- [x] **左右反転データ拡張** - 対称性を活用
  - `train.py`: `--augment-flip`オプション
  - 効果: データを2倍に（normalize-turnと合わせて使用可能）

- [x] **MultiPV候補手のcandidates記録** - ranking学習・オフライン一致率の教師
  - `gen_dataset.py`: `--multipv` 2以上で親局面レコードに候補手リストを付与

- [x] **Pairwise ranking損失** - 候補手間の優劣を直接学習（Bonanza比較学習と同系統）
  - `train.py`: `--ranking-weight` / `--ranking-min-gap`オプション
  - `models/dataset.py`: `ShogiRankingPairDataset`（対局単位分割をメインと共有）

- [x] **オフライン指し手一致率の自動計測** - 「val_lossは下がったが弱い」の早期検出
  - `scripts/move_agreement.py`: `--offline`モード（candidates教師、エンジン不要）
  - `train.py`: `--agreement-data`で定期保存時と学習終了時に自動計測・ログ記録

- [ ] **重み付き損失（nodes基準）** - 難しい局面を重視
  - `gen_dataset.py`: 探索ノード数をJSONLに保存（SearchResultに既存）
  - `dataset.py`, `train.py`: ノード数が多い局面の損失を重くする

### 優先度C: 学習の安定化・汎化性能向上（半日〜1日）

- [ ] **SWA (Stochastic Weight Averaging)** - 汎化性能向上
  - `train.py`: 末期エポックでSWAを有効化
  - `torch.optim.swa_utils.AveragedModel`使用

- [x] **EMA (Exponential Moving Average)** - 推論用重み平滑化
  - `train.py`: `--ema-decay`オプション（推奨: 0.999）

- [ ] **検証セット分割評価** - 弱点の可視化
  - `train.py`: 序盤(ply<30)/中盤(30-80)/終盤(80+)別にval_lossを計算・ログ出力

### 優先度D: 推論改良（半日）

- [ ] **TTA (Test-Time Augmentation)** - 推論精度向上
  - `evaluator.py`: 元盤面と左右反転盤面の評価値を平均

- [ ] **Temperature付きスケーリング** - 出力調整
  - `evaluator.py`: `tanh(x / T)`でチューニングパラメータ導入

### 優先度E: 特徴量拡張（半日〜1日）

- [ ] **王手フラグ追加** - 局面状態の明示
  - `features.py`: python-shogiの`board.is_check()`を活用

- [ ] **駒得スカラー追加** - グローバル特徴
  - `features.py`: 先手の駒価値合計 - 後手の駒価値合計

### 優先度F: モデル構造の実験（数日）

- [ ] **CNN + Transformer** - 局所性を明示
  - `value_transformer.py`: 3×3近傍を扱うCNN埋め込みの上にTransformerを重ねる

- [ ] **RoPE/ALiBi位置埋め込み** - 長手数への汎化
  - `value_transformer.py`: 固定正弦波→Rotary Position Embeddingに変更

### 優先度G: 長期研究課題

- [ ] **マルチタスク（指し手優先度ヘッド）** - ポリシーの粗スーパビジョン
- [ ] **cp→勝率キャリブレーション** - 後処理補正関数
- [ ] **Curriculum Learning** - 序盤→中盤→終盤のデータ比率を段階的に変化
- [ ] **軽量ビーム探索** - αβ等の探索アルゴリズム実装
- [ ] **Endgame特化ヘッド** - 終盤検知時に別ヘッドに切り替え
