# 評価関数改善の実装サマリー（2026-07-07 自動実装）

夜間の自動実装セッションで、評価関数を強くするための改善（探索強化以外）を
実装した。全機能テスト済み（82テストパス）、機能ごとにmainへコミット済み。

## 実装した機能

### 1. 計測基盤（コミット c51864c, 5f7f38c）

| スクリプト | 用途 |
|-----------|------|
| `scripts/move_agreement.py` | モデル1手読みと教師エンジン最善手の一致率（序盤/中盤/終盤別） |
| `scripts/selfplay_match.py` | 2つのcheckpointの対局による勝率・推定レート差 |

- 現行best.ptのベースライン: **一致率 30%**（depth 5教師、10局面の小サンプル）
- 今後の改善はこの2指標で効果測定する

### 2. 学習の安定化（コミット 86293f3, e4a163e）

- **EMA** `--ema-decay 0.999`: 検証・保存にEMA重みを使用。checkpointには
  推論用（EMA）と再開用（raw）の両方を保存
- **Huber loss** `--value-loss huber --huber-delta 0.5`: 教師の探索の揺れにロバスト

### 3. ターゲット設計（コミット dfcd87d）

- **cp clamp** `--cp-clamp 2000`: 大差局面を除外せず丸めて学習に残す
  （従来の`--cp-filter-threshold`は終盤情報を捨てていた）
- **勝率空間ターゲット** `--target-mode wdl --wdl-lambda 0.5`: elmo式。
  教師評価値の勝率と実際の勝敗をブレンド。推論時は自動で逆変換

### 4. モデル構造オプション（コミット 6ad48ff）

- **玉相対位置埋め込み** `--use-king-relative`: 各マスに先手玉・後手玉からの
  相対位置(17×17)埋め込みを加算。NNUEのHalfKPに相当する帰納バイアスで、
  最も効果が期待される構造変更
- **2D位置埋め込み** `--use-2d-pos`: 固定正弦波→学習可能な段・筋埋め込み
- **持ち駒離散埋め込み** `--use-discrete-hand`: 枚数の非線形価値を表現

いずれもデフォルト無効・既存checkpointと後方互換（best.ptのロード確認済み）。

### 5. データ生成の拡張（コミット 01d6414）

- **PV末端記録** `--record-pv-leaf`: 探索評価値をPV末端の静止局面に付与。
  「探索後の評価値が取り合い途中の局面に付く」ミスマッチを解消する、
  1手読みエンジンにとって最重要のデータ改善
- **MultiPV** `--multipv 3`: 上位N手の子局面（順位2以下）に評価値を付与。
  「良い手と悪い手の評価差」を1回の探索から収集
- 実測: depth 5・2対局で 165→654レコード（約4倍）
- 副産物: MultiPV時にbestmoveの評価値が最下位候補の値になるバグを修正

## 検証内容

- ユニットテスト82件パス（新規56件追加）
- 実エンジン（水匠5）でのデータ生成動作確認（PV末端・MultiPVレコードの内訳・符号を検証）
- 新形式データを`ShogiValueDataset`で読み込めることを確認
- 既存checkpoint（checkpoints/best.pt）のロード互換を確認
- move_agreement / selfplay_match を実モデルでスモークテスト

## 未実装（提案済みだが今回のスコープ外）

- SWA、検証セットのply帯別val_loss、TTA、王手フラグ・駒得スカラー特徴量
- CNN+Transformer、αβ探索（探索強化は意図的に除外）

## 推奨する次のステップ

1. **データ再生成**（推奨コマンド例）:
   ```bash
   .venv/bin/python tools/gen_dataset.py -n 2000 --depth 10 --workers 8 \
       --record-pv-leaf --multipv 3 -o data/raw/suisho5_2000_pv_mpv.jsonl
   ```
   ※ MultiPV 3はdepth 10の探索時間を約2〜3倍にする点に注意

2. **改善オプションを全部入れた学習**:
   ```bash
   PYTHONPATH=. python train/train.py \
       --data data/raw/suisho5_2000_pv_mpv.jsonl \
       --use-features --normalize-turn --augment-flip \
       --use-king-relative --use-2d-pos --use-discrete-hand \
       --target-mode wdl --wdl-lambda 0.7 \
       --cp-clamp 2000 --value-loss huber --ema-decay 0.999 \
       --epochs 30 --batch-size 512
   ```
   ※ 一度に全部入れると効果の切り分けができないため、
   ベースライン→1機能ずつ追加、をmove_agreementで測るのが理想

3. **効果測定**: 学習ごとに
   ```bash
   PYTHONPATH=. python scripts/move_agreement.py --model checkpoints/best.pt \
       --data <held-out data> --depth 10 --limit 500
   PYTHONPATH=. python scripts/selfplay_match.py \
       --model-a checkpoints/best.pt --model-b <旧モデル> --games 50
   ```
