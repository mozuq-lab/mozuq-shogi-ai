# 局面感度蒸留ロードマップ設計書（Phase 0–1 詳細 / Phase 2–4 概要）

日付: 2026-07-11
ステータス: レビュー待ち

## 背景と目的

主流の将棋AI（NNUE+αβ、Policy/Value+MCTS）と差別化しつつ棋力を上げるため、
以下のロードマップを段階的に実装する。各フェーズは独立に計測し、
効果をアブレーションで示せる形を保つ。

| Phase | 内容 | 状態 |
|-------|------|------|
| 0 | 計測基盤: cp regret計測 + 独立holdout運用 | 一部完了※ |
| 1 | 局面感度蒸留: ΔV（評価値差分）損失 | 本設計書の主対象 |
| 2 | ハードネガティブマイニング（ΔV誤差の大きいペアの重点採用） | 概要のみ |
| 3 | Qヘッド（from-toトークンペア採点）による全合法手一括評価 | 設計合意済み・概要のみ |
| 4 | 未来依存の戦術補助タスク（SEE・短手数詰み・静止後駒損益） | 概要のみ |

※ Phase 0 のうち「1手詰め確定評価」（`52d9c83`）と「分岐局面の勝敗マスク」
（`2bae5eb`）は実装済み。

### 設計原則（Codexレビューで確定した制約）

- ΔV損失は新情報ではなく「意思決定に効く差分方向の誤差への再重み付け」。
  価値の中心は摂動ペアの設計・採用基準・教師信頼性の保証にある。
- 摂動は**到達可能性の安全リスト内**に限定する（教師の水匠5も実戦分布で
  学習されたNNUEであり、異常局面ではラベルが不安定になるため）。
- 摂動は**駒素材を保存するもの優先**（駒の増減ペアのΔVは駒価値で説明されて
  しまい、位置・働きの感度という本来の信号が薄まる）。
- ペアの両局面は**手番が一致**していること（score_cpが手番側視点のため）。
- Qヘッドは Phase 3 まで導入しない（差分学習の効果と構造変更の効果を
  切り分けるため）。

## 現行コードとの整合検証（2026-07-11、HEAD=e95ee95）

直近コミット（`db522ca` best.ptへの一致率記録、`2bae5eb` 勝敗マスク、
`52d9c83` 1手詰め確定評価、`e95ee95` ドキュメント）と本計画の衝突を
実コードで確認した。結果:

- **衝突なし**。勝敗マスクは `outcome_weight` 経由で完結しており、
  ΔVペア学習は outcome を使わないため相互作用しない。
- 1手詰め確定評価は `find_best_move` の NN 前のルール判定であり、
  一致率・regret 計測、将来のQヘッドいずれとも直交する。
- checkpoint への一致率記録（`db522ca`）は `state` への追記のみで、
  `Evaluator` は `config` と `model_state_dict` しか読まないため影響なし。
- **好材料**: `_ranking_forward`（train.py）が ΔV に必要な Siamese forward
  （encoder共有で2局面をcatして1回のforward）を既に実装している。
  `ShogiRankingPairDataset` / `select_val_games` による対局単位分割の共有、
  ペアバッチサイズの半減調整（1ステップの局面数をメインと揃える）も
  流用できるため、Phase 1a は既存機構の拡張だけで実装できる。

## Phase 0: 計測基盤

### 0-1. cp regret 計測（`scripts/move_agreement.py`）

一致率は同等手が複数ある局面で原理的にノイズが入る。教師視点での
「選んだ手と最善手の評価差」（regret）を主指標に追加する。

- `measure_agreement_offline` を拡張:
  - モデルの選択手が candidates に含まれる場合:
    `regret = score_cp(rank1) − score_cp(選択手)`（cp、手番側視点、≧0）
  - 含まれない場合: regret は下限しか分からない（censored）。
    件数を `regret_censored` として別集計し、平均には含めない
    （カバレッジは既存の `multipv_hit_rate` で監視）。
  - 詰みスコア（±30000）で平均が壊れるため、per-position regret を
    `--regret-clamp`（デフォルト 1000cp）で丸めてから集計する。
  - 出力に `regret_mean_cp` / `regret_median_cp` / `regret_censored` を追加。
    `summarize()` を拡張し ply帯別（序盤/中盤/終盤）の regret も出す。
- `train.py` の自動計測（`run_offline_agreement` → `state.agreement`）は
  結果 dict のマージなので**無変更で新フィールドが記録される**。
  エポックログの表示行に regret を追加する。
- 既存の JSON 出力・checkpoint への記録形式と後方互換
  （キー追加のみ、既存キーは不変）。

### 0-2. 独立 holdout の運用

- 学習に一切使わない専用 JSONL（例: `data/raw/holdout_mpv.jsonl`）を
  `gen_dataset.py` で別途生成し（MultiPV付き、固定シード）、
  `--agreement-data` とオフライン計測はこのファイルに統一する。
- 対局単位分割（`split_by_game`）はファイル内リークを防ぐが、
  生成条件レベルの過適合は検出できない。holdout はこれを補う。
- コード変更: `train.py` に `agreement_data == data_path` の場合の
  警告ログを追加（エラーにはしない）。
- 運用ルールを CLAUDE.md に記載する。
- game_id はファイルごとに 0 起番なので、ファイルを跨いだ game_id の
  照合はしない（ファイル分離のみで担保する）。

### 0-3. 計測プロトコルの固定

フェーズ間の比較条件を揃えるため、以下を固定する:

- holdout ファイル、`--limit`、シード、regret-clamp 値
- 教師探索条件（MultiPV数・depth）。Phase 3 で Q top-k recall を測る際も
  同一条件を使う。

## Phase 1: 局面感度蒸留（ΔV損失）

### Phase 1a: 既存 candidates ペアで ΔV 回帰（データ再生成不要）

現在の ranking 損失は兄弟手ペアの「順序」のみ学習している。
これを「差分量」の回帰に拡張する。

- `ShogiRankingPairDataset` を拡張し、ペアごとに教師差分ターゲットを返す:
  - 良い手 a（親視点 s_a）、悪い手 b（s_b、s_a > s_b）に対し、
    子局面の値ターゲットは手番反転で `y = n(−s)`（n は target_mode に従う
    正規化: cp なら tanh(cp/cp_scale)、wdl なら勝率マップ）。
  - `delta_target = n(−s_a) − n(−s_b)`（< 0 になるのが正しい向き）。
  - このためデータセットに `cp_scale` / `target_mode` / `wdl_scale` を追加
    （メイン設定と同値を渡す）。
- `train_epoch`: `_ranking_forward` の返す `(value_better, value_worse)` に
  対し、既存 RankNet 損失に加えて
  `delta_loss = huber((v_a − v_b) − delta_target)` を計算。
  **1回の forward を2つの損失で共有**するため追加計算はほぼゼロ。
- 新オプション: `--delta-weight`（デフォルト 0 = 無効）。
  検証時は `validate_ranking` を拡張して delta 誤差もログする。
- normalize_turn / augment_flip との整合は既存のペア経路をそのまま通る
  （両局面は同手番なので同一変換、ΔV は左右反転で不変）。

### Phase 1b: 摂動ペア生成（`tools/gen_perturb_pairs.py` 新規）

安全リスト内の摂動でペアを増やす。優先順:

1. **巻き戻し分岐**: 本譜の局面から k 手（k=1..3）巻き戻し、教師の
   MultiPV 候補から本譜と異なる合法手を1手進めた局面と、本譜側の
   同 ply 局面のペア。到達可能・同手番。
2. **同一駒の移動先違い**: 同一親局面から同じ駒を異なるマスへ動かす
   合法手ペア（candidates に含まれない移動先も対象にできる）。
3. **成/不成ペア**: 同じ移動で成る/成らないだけが違うペア（素材一致）。

分岐や移動先の違いで駒の取り合いが変わるとペアの素材は一致しない。
生成時に素材差を `material_diff` として記録し、学習側で
「素材一致ペア優先」のフィルタ/重み付けができるようにする
（設計原則の素材保存優先はこのフィルタで実現する）。
4. **公式駒落ち初期局面**からの通常対局データ（ペアではなく分布拡張。
   `gen_dataset.py` の初期局面オプション追加として実装）。

仕様:

- 入力は学習に使う JSONL。**元レコードの game_id を出力ペアに引き継ぐ**
  （`select_val_games` の分割を共有し、val 局面由来のペアが訓練に
  混入するリークを防ぐ。ranking と同じ機構）。
- 両局面を水匠5で評価してラベル付け。**安定性フィルタ**: ノード数を
  変えて2回評価し、評価順序が入れ替わる・差の符号が変わるペアは破棄。
- ペアの両局面の手番一致を assert。
- 出力形式: `{"sfen_a", "sfen_b", "score_cp_a", "score_cp_b",
  "pair_type", "game_id"}`（score は各局面の手番側視点）。
- 学習側: 新 `ShogiDeltaPairDataset`（`ShogiRankingPairDataset` の
  `_prepare_position` を共通化して流用）+ `--delta-data` オプション。
  ローダーは ranking と同様に1メインバッチにつき1ペアバッチ消費
  （尽きたら周回）、バッチサイズ半減で局面数を揃える。
- 消費が1:1固定のため、摂動学習の実効的な強さは `--delta-weight` で
  制御する。摂動ペアの局面数は全学習局面の10〜20%を目安に生成する。

### Phase 1 の判定基準

holdout での cp regret（主指標）・一致率・ranking pair accuracy を
Phase 0 プロトコルで比較。ベースライン（ΔVなし・同一データ・同一エポック）
との A/B で改善を確認してから Phase 2 に進む。

## Phase 2–4 概要（別設計書で詳細化）

- **Phase 2**: 現行モデルでペアファイルを一括推論し、ΔV 誤差上位の
  ペアに重みを付けた再学習。ペアレコードに任意フィールド `weight` を
  追加し、損失で乗算する（データセット側は読むだけ）。
- **Phase 3**: Qヘッド（from-toトークンペア採点、value/outcomeヘッド維持、
  `use_action_head` を config に保存、engine は value/q/hybrid の3モード）。
  教師Qの探索条件を統一し、Q top-k recall を計測。
  合意済み設計は会話ログ参照、実装前に本書と同形式で詳細化する。
- **Phase 4**: 手単位の検証可能な戦術補助タスク（SEE、短手数詰みの有無、
  静止後駒損益）。入力特徴（features.py）と重複しない
  「現在の盤面から直接見えない事実」に限定する。Phase 3 の
  手トークン化基盤を流用する。

## テスト計画

- regret 集計: 既知の candidates を持つ合成レコードで
  mean/median/censored/clamp を検証（pytest）。
- delta_target の符号と正規化: cp/wdl 両モードで手計算値と一致すること。
- 摂動ペア生成: 手番一致 assert、game_id 引き継ぎ、安定性フィルタの
  破棄動作をモックエンジンで検証。
- normalize_turn / augment_flip 下で ΔV ターゲットが不変であること。
- 既存テスト（test_models.py / test_evaluator.py）の無退行。

## リスク

- 摂動局面の教師ラベル品質: 安全リスト + 安定性フィルタで緩和。
  それでも劣化が見えたら pair_type 別に損失重みを下げて切り分ける。
- ΔV が効かない可能性: Phase 1a は実装コストが小さいので、効果ゼロでも
  損失は限定的。判定は regret で行い、val_loss では判断しない。
- 学習時間の増加（ペア分の forward）: ranking と同じ混合方式のため
  1ステップの局面数は不変。エポック時間の増加は許容範囲を実測で確認。
