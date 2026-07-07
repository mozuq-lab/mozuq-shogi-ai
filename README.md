# 将棋AI（蒸留Transformer）

水匠5の評価値を教師として蒸留した軽量Transformer評価ネットワーク。

## 特徴

- **Transformerベースの評価関数**: 81マス + 持ち駒をトークンとして処理
- **拡張特徴量**: 利きマップ、玉の安全度、駒価値などを追加入力として使用可能
- **モデル構造オプション**: 玉相対位置埋め込み（NNUEのHalfKP相当）、2D位置埋め込み、持ち駒離散埋め込み
- **柔軟な学習ターゲット**: tanh正規化 / 勝率空間（elmo式ブレンド）、EMA・Huber loss対応
- **高密度データ生成**: PV末端局面・MultiPV候補手へのラベル付与で1対局あたり約4倍のデータ
- **計測ツール**: 教師との指し手一致率、モデル同士の自己対局によるレート差測定
- **勝敗補助損失**: 評価値予測と勝率予測のマルチタスク学習
- **USIプロトコル対応**: 将棋GUIで対局可能

## セットアップ

```bash
# 依存関係のインストール
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# サブモジュールの初期化（水匠5エンジン・評価関数）
git submodule update --init --recursive
```

## クイックスタート

### データ生成

```bash
# 水匠5で自己対局データを生成
python tools/gen_dataset.py -n 100 --depth 10 -o data/raw/dataset.jsonl

# PV末端局面 + MultiPV候補手も記録（レコード数が約4倍に）
python tools/gen_dataset.py -n 100 --depth 10 --workers 4 \
    --record-pv-leaf --multipv 3 -o data/raw/dataset.jsonl

# 弱いAIとの対局データ生成（評価値に差がつく局面を収集）
python tools/gen_dataset.py -n 100 --weak-side alternate --weak-prob 0.3
```

### 学習

```bash
# 基本的な学習
PYTHONPATH=. python train/train.py \
    --data data/raw/dataset.jsonl \
    --epochs 100 \
    --batch-size 512

# 推奨オプションを有効にした学習
PYTHONPATH=. python train/train.py \
    --data data/raw/dataset.jsonl \
    --use-features \
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

モデル構造オプション（`--use-king-relative` / `--use-2d-pos` / `--use-discrete-hand`）や
勝率空間ターゲット（`--target-mode wdl`）は、下記の計測ツールで効果を確認しながら
1つずつ追加することを推奨。全オプションの一覧は [CLAUDE.md](CLAUDE.md) を参照。

### 評価関数の計測

```bash
# 教師エンジンとの指し手一致率（序盤/中盤/終盤別）
PYTHONPATH=. python scripts/move_agreement.py \
    --model checkpoints/best.pt --data data/raw/dataset.jsonl \
    --depth 10 --limit 300

# 新旧モデルの自己対局（勝率と推定レート差）
PYTHONPATH=. python scripts/selfplay_match.py \
    --model-a checkpoints/new.pt --model-b checkpoints/old.pt --games 30
```

### 棋譜確認（KIF形式変換）

```bash
# 対局をKIF形式で出力（ShogiGUIで評価値グラフ表示可能）
python scripts/to_kif.py data/raw/dataset.jsonl 0 -o game0.kif
```

### USIエンジンとして使用

```bash
# 起動
./shogi-ai-engine.sh

# または直接実行
PYTHONPATH=. python engine/usi_server.py --model checkpoints/best.pt
```

将棋所やShogiGUIなどのUSI対応GUIで `shogi-ai-engine.sh` をエンジンとして登録すると対局できます。

## 主なオプション

代表的なもののみ。全一覧と詳細は [CLAUDE.md](CLAUDE.md) を参照。

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--data` | 必須 | データファイルパス |
| `--epochs` | 100 | エポック数 |
| `--batch-size` | 512 | バッチサイズ |
| `--device` | auto | デバイス（auto/cuda/mps/cpu） |
| `--use-features` | - | 拡張特徴量を使用 |
| `--normalize-turn` | - | 後手番を先手視点に正規化 |
| `--augment-flip` | - | 左右反転でデータ2倍化 |
| `--cp-clamp` | - | 評価値を±この値に丸める（推奨: 2000） |
| `--value-loss` | mse | 評価値損失（mse / huber） |
| `--ema-decay` | 0 | EMA減衰率（推奨: 0.999） |
| `--target-mode` | cp | ターゲット空間（cp / wdl=勝率elmoブレンド） |
| `--use-king-relative` | - | 玉相対位置埋め込み |

## ディレクトリ構成

```
shogi-ai/
├── models/           # モデル定義
├── train/            # 学習スクリプト
├── engine/           # USIエンジン
├── tools/            # データ生成ツール
├── scripts/          # 計測・変換スクリプト
├── tests/            # テスト（pytest）
├── external/         # 外部エンジン（git submodule）
├── reports/          # 評価レポート
└── checkpoints/      # 学習済みモデル
```

## 開発ドキュメント

詳細な開発ガイドは [CLAUDE.md](CLAUDE.md) を参照してください。

## ライセンス

MIT License
