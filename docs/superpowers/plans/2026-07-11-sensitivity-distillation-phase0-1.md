# 局面感度蒸留 Phase 0–1 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cp regret 計測と独立 holdout 運用（Phase 0）、既存 candidates ペアの ΔV 差分回帰（Phase 1a）、摂動ペア生成と学習統合（Phase 1b）を実装する。

**Architecture:** 既存の ranking 学習基盤（`ShogiRankingPairDataset` + `_ranking_forward` の Siamese forward + 対局単位分割共有）を拡張し、順序学習に差分量回帰を追加する。摂動ペアは新スクリプト `tools/gen_perturb_pairs.py` で安全リスト内の摂動（巻き戻し分岐・移動先違い・成/不成）を生成し、エンジンの2段階評価による安定性フィルタを通す。

**Tech Stack:** Python 3.10+ / PyTorch / python-shogi / pytest。教師エンジンは水匠5（`shogi_utils.USIEngine`、`go(nodes=N)` 対応済み）。

**Spec:** `docs/superpowers/specs/2026-07-11-sensitivity-distillation-design.md`

## Global Constraints

- 型ヒント必須、docstring は Google style、`from __future__ import annotations` を使用（コーディング規約）
- regret のデフォルト clamp は **1000cp**、regret はオフラインモード（candidates 教師）のみ
- ΔV ターゲット規約（符号バグ防止のため厳守）:
  - candidates ペア（Phase 1a）: score は**親の手番側視点** → 子局面ターゲットは `n(−s)`。`delta_target = n(−s_better) − n(−s_worse)`（**負値**になる）
  - 摂動ペア（Phase 1b）: score は**各局面自身の手番側視点**（モデル出力と同じ） → `delta_target = n(s_a) − n(s_b)`（符号反転なし）
  - `n(s)` = cp モード: `tanh(s / cp_scale)`、wdl モード: `2·sigmoid(s / wdl_scale) − 1`
- ペアローダーはメインバッチ1回につき1バッチ消費（尽きたら周回）、バッチサイズは `batch_size // 2`
- ペアの両局面は手番一致（生成側で ValueError）
- テスト実行はリポジトリルートで `python -m pytest`
- コミットメッセージは日本語（既存リポジトリの慣習）で、以下のトレーラを付ける:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012rhFZh9d6FQjKbDtRNhTsN
  ```

## ファイル構成

| ファイル | 変更 | 責務 |
|---------|------|------|
| `scripts/move_agreement.py` | 変更 | `summarize_regret()` 追加、`measure_agreement_offline()` に regret 統合、CLI `--regret-clamp` |
| `train/train.py` | 変更 | regret ログ、holdout 警告、`compute_delta_loss()`、`--delta-weight` / `--delta-data` 統合、`validate_delta()` |
| `models/dataset.py` | 変更 | `value_target_from_cp()`、`prepare_pair_position()`、`ShogiRankingPairDataset` に delta_target、`ShogiDeltaPairDataset` 新設 |
| `models/__init__.py` | 変更 | 新シンボルの export |
| `tools/gen_perturb_pairs.py` | 新規 | 摂動ペア生成（純粋関数群 + エンジンラベル付け + CLI） |
| `CLAUDE.md` | 変更 | 計測プロトコル・新オプションのドキュメント |
| `tests/test_scripts.py` | 変更 | `summarize_regret` のテスト |
| `tests/test_ranking.py` | 変更 | regret 統合・delta_target・`ShogiDeltaPairDataset` のテスト |
| `tests/test_train.py` | 変更 | `compute_delta_loss`・holdout 警告・E2E 学習テスト |
| `tests/test_gen_perturb_pairs.py` | 新規 | 摂動ペア生成の純粋関数・ラベル付けのテスト |

---

### Task 1: summarize_regret 関数（Phase 0）

**Files:**
- Modify: `scripts/move_agreement.py`（`summarize()` の直後に追加）
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: `PHASE_BOUNDS`（move_agreement.py:50 既存）
- Produces: `summarize_regret(records: list[tuple[int, float | None]], clamp: float = 1000.0) -> dict`
  — 返り値キー: `regret_mean_cp`, `regret_median_cp`, `regret_samples`, `regret_censored`, `regret_clamp`, `regret_by_phase`（Task 2 が使用）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_scripts.py` の import に `summarize_regret` を追加し、`TestSummarize` クラスの後に追加:

```python
from scripts.move_agreement import (
    board_from_sfen_line,
    engine_position_args,
    summarize,
    summarize_regret,
)
```

```python
class TestSummarizeRegret:
    """summarize_regret（cp regret集計）のテスト."""

    def test_mean_median_and_censored(self) -> None:
        # (ply, regret)。Noneはcensored（モデルの手が候補外で下限しか不明）
        records = [(0, 0.0), (10, 100.0), (40, 200.0), (90, None)]
        summary = summarize_regret(records, clamp=1000.0)
        assert summary["regret_mean_cp"] == pytest.approx(100.0)
        assert summary["regret_median_cp"] == pytest.approx(100.0)
        assert summary["regret_samples"] == 3
        assert summary["regret_censored"] == 1

    def test_clamp_applied(self) -> None:
        # 詰みスコア由来の巨大regretはclampで丸めてから平均する
        records = [(0, 30000.0), (0, 0.0)]
        summary = summarize_regret(records, clamp=1000.0)
        assert summary["regret_mean_cp"] == pytest.approx(500.0)

    def test_phase_breakdown(self) -> None:
        records = [(0, 100.0), (40, 300.0), (90, None)]
        summary = summarize_regret(records, clamp=1000.0)
        by_phase = summary["regret_by_phase"]
        assert by_phase["opening"]["mean_cp"] == pytest.approx(100.0)
        assert by_phase["middlegame"]["mean_cp"] == pytest.approx(300.0)
        assert by_phase["endgame"]["censored"] == 1
        assert by_phase["endgame"]["samples"] == 0

    def test_empty(self) -> None:
        summary = summarize_regret([], clamp=1000.0)
        assert summary["regret_mean_cp"] == 0.0
        assert summary["regret_samples"] == 0
        assert summary["regret_censored"] == 0
```

`tests/test_scripts.py` に `pytest` の import が無い場合は追加する（`import pytest`）。

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_scripts.py::TestSummarizeRegret -v`
Expected: FAIL（ImportError: cannot import name 'summarize_regret'）

- [ ] **Step 3: 実装**

`scripts/move_agreement.py` の import に `import statistics` を追加し、`summarize()` の直後に:

```python
def summarize_regret(
    records: list[tuple[int, float | None]],
    clamp: float = 1000.0,
) -> dict:
    """cp regret記録を集計.

    regretは「教師視点での最善手と選択手の評価差」（cp、≧0）。
    一致率と違い同等手が複数ある局面のノイズを吸収できる。
    モデルの選択手がcandidatesに無い局面は下限しか分からない（censored）
    ため件数のみ数え、平均・中央値には含めない。詰みスコア（±30000）で
    平均が壊れないよう、集計前にclampで丸める。

    Args:
        records: (ply, regret)のリスト。regret=Noneはcensored
        clamp: 集計前にregretを丸める上限（cp）

    Returns:
        regret_mean_cp / regret_median_cp / regret_samples /
        regret_censored / regret_clamp / regret_by_phase を含む辞書
    """

    def _stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        clamped = [min(v, clamp) for v in values]
        return statistics.mean(clamped), statistics.median(clamped)

    valid = [r for _, r in records if r is not None]
    mean, median = _stats(valid)
    summary: dict = {
        "regret_mean_cp": mean,
        "regret_median_cp": median,
        "regret_samples": len(valid),
        "regret_censored": sum(1 for _, r in records if r is None),
        "regret_clamp": clamp,
    }

    by_phase: dict = {}
    for phase, (lo, hi) in PHASE_BOUNDS.items():
        phase_records = [(ply, r) for ply, r in records if lo <= ply <= hi]
        phase_valid = [r for _, r in phase_records if r is not None]
        p_mean, p_median = _stats(phase_valid)
        by_phase[phase] = {
            "mean_cp": p_mean,
            "median_cp": p_median,
            "samples": len(phase_valid),
            "censored": sum(1 for _, r in phase_records if r is None),
        }
    summary["regret_by_phase"] = by_phase
    return summary
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_scripts.py -v`
Expected: 全 PASS（既存テスト含む）

- [ ] **Step 5: コミット**

```bash
git add scripts/move_agreement.py tests/test_scripts.py
git commit -m "cp regret集計関数を追加（一致率より頑健な手選び指標）"
```

---

### Task 2: measure_agreement_offline への regret 統合（Phase 0）

**Files:**
- Modify: `scripts/move_agreement.py`（`measure_agreement_offline` 本体、`main()` の CLI と表示）
- Test: `tests/test_ranking.py`（`TestOfflineAgreement` に追加）

**Interfaces:**
- Consumes: Task 1 の `summarize_regret`
- Produces: `measure_agreement_offline(evaluator, data_path, limit=None, seed=42, regret_clamp=1000.0) -> dict`
  — 返り値に Task 1 のキーが追加される。**既存キーは不変**（train.py の自動計測は dict マージなので無変更で新キーが記録される）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_ranking.py` の `TestOfflineAgreement` に追加:

```python
    def test_regret_zero_for_best_move(self, ranking_data: Path) -> None:
        # "2g2f"はstartposのrank1（regret 0）。
        # "startpos moves 2g2f"では候補外なのでcensored
        evaluator = _FakeEvaluator("2g2f")
        summary = measure_agreement_offline(evaluator, ranking_data)
        assert summary["regret_mean_cp"] == pytest.approx(0.0)
        assert summary["regret_samples"] == 1
        assert summary["regret_censored"] == 1

    def test_regret_nonzero_for_worse_move(self, ranking_data: Path) -> None:
        # "9g9f"はstartposのrank3: regret = 50 − (−80) = 130cp
        evaluator = _FakeEvaluator("9g9f")
        summary = measure_agreement_offline(evaluator, ranking_data)
        assert summary["regret_mean_cp"] == pytest.approx(130.0)
        assert summary["regret_samples"] == 1
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_ranking.py::TestOfflineAgreement -v`
Expected: 新規2件が FAIL（KeyError: 'regret_mean_cp'）

- [ ] **Step 3: 実装**

`measure_agreement_offline` のシグネチャに `regret_clamp: float = 1000.0` を追加し、docstring の Args に追記。ループ前に `regret_records: list[tuple[int, float | None]] = []` を初期化。ループ本体を次のように変更（`teacher_move` 決定部と判定部の置き換え）:

```python
    for sample in samples:
        board = board_from_sfen_line(sample["sfen"])
        legal_moves = list(board.legal_moves)
        if len(legal_moves) <= 1:
            # 選択の余地がない局面は測定対象外
            skipped += 1
            continue

        candidates = sample["candidates"]
        best = min(candidates, key=lambda c: c["rank"])
        teacher_move = best["move"]
        candidate_scores = {c["move"]: c.get("score_cp") for c in candidates}

        model_move, _ = evaluator.find_best_move(board)

        ply = sample.get("ply", 0)
        results.append((ply, model_move == teacher_move))
        if model_move in candidate_scores:
            hit_count += 1

        # cp regret: 教師視点での最善手と選択手の評価差。
        # 選択手が候補外（またはスコア欠損）の場合は下限しか分からないので
        # censored（None）として記録する
        best_score = best.get("score_cp")
        chosen_score = candidate_scores.get(model_move)
        if best_score is None or chosen_score is None:
            regret_records.append((ply, None))
        else:
            regret_records.append((ply, float(best_score - chosen_score)))
```

集計部（`summary = summarize(results)` の後）に追加:

```python
    summary.update(summarize_regret(regret_records, clamp=regret_clamp))
```

`main()` に CLI オプションを追加:

```python
    parser.add_argument("--regret-clamp", type=float, default=1000.0,
                        help="regret集計時のclamp上限（cp、詰みスコア対策）")
```

offline 分岐の呼び出しを変更:

```python
        summary = measure_agreement_offline(
            evaluator, args.data, limit=limit, seed=args.seed,
            regret_clamp=args.regret_clamp,
        )
```

表示部（`MultiPV hit` の print の後）に追加:

```python
    if "regret_mean_cp" in summary:
        print(f"cp regret:  mean={summary['regret_mean_cp']:.1f} "
              f"median={summary['regret_median_cp']:.1f} "
              f"(censored={summary['regret_censored']})")
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_ranking.py tests/test_scripts.py -v`
Expected: 全 PASS

- [ ] **Step 5: コミット**

```bash
git add scripts/move_agreement.py tests/test_ranking.py
git commit -m "オフライン一致率計測にcp regretを追加"
```

---

### Task 3: train.py の regret ログ・holdout 警告・計測プロトコルのドキュメント（Phase 0）

**Files:**
- Modify: `train/train.py`
- Modify: `CLAUDE.md`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Task 2 の regret キー（`run_offline_agreement` の返り値に自動で含まれる）
- Produces: `agreement_data_overlaps(data_path: str, agreement_data: str | None) -> bool`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_train.py` に追加:

```python
class TestAgreementOverlapWarning:
    """holdout運用の同一ファイル検出のテスト."""

    def test_same_file_detected(self, tmp_path: Path) -> None:
        from train.train import agreement_data_overlaps
        path = tmp_path / "data.jsonl"
        path.write_text("{}")
        assert agreement_data_overlaps(str(path), str(path))

    def test_different_file(self, tmp_path: Path) -> None:
        from train.train import agreement_data_overlaps
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text("{}")
        b.write_text("{}")
        assert not agreement_data_overlaps(str(a), str(b))

    def test_none_agreement_data(self) -> None:
        from train.train import agreement_data_overlaps
        assert not agreement_data_overlaps("data.jsonl", None)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_train.py::TestAgreementOverlapWarning -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 実装**

`train/train.py` の `run_offline_agreement` の前に追加:

```python
def agreement_data_overlaps(data_path: str, agreement_data: str | None) -> bool:
    """一致率計測データが学習データと同一ファイルかどうか判定.

    学習データでの計測は生成条件レベルの過適合を検出できないため、
    独立したholdoutデータの使用を促す警告に使う。

    Args:
        data_path: 学習データのパス
        agreement_data: 一致率計測データのパス（未指定はNone）

    Returns:
        同一ファイルならTrue
    """
    if not agreement_data:
        return False
    try:
        return Path(agreement_data).resolve() == Path(data_path).resolve()
    except OSError:
        return False
```

`train()` の冒頭（`device = get_device(...)` の直後）に追加:

```python
    if agreement_data_overlaps(config.data_path, config.agreement_data):
        logger.warning(
            "agreement_dataが学習データと同一ファイルです。"
            "計測は独立したholdoutデータで行うことを推奨します"
        )
```

agreement のログ3箇所に regret を追加。定期保存時（`Agreement (epoch ...)` のログ）:

```python
                logger.info(
                    f"Agreement (epoch {epoch + 1}): "
                    f"{result['agreement']:.3f} "
                    f"(multipv_hit={result['multipv_hit_rate']:.3f}, "
                    f"regret_mean={result['regret_mean_cp']:.1f}cp)"
                )
```

学習終了時（`Agreement (final)` のログ）:

```python
        logger.info(
            f"Agreement (final): {result['agreement']:.3f} "
            f"(序盤={result['opening']['agreement']:.3f}, "
            f"中盤={result['middlegame']['agreement']:.3f}, "
            f"終盤={result['endgame']['agreement']:.3f}, "
            f"multipv_hit={result['multipv_hit_rate']:.3f}, "
            f"regret_mean={result['regret_mean_cp']:.1f}cp)"
        )
```

best.pt 計測時（`Agreement (best, ...)` のログ）:

```python
            logger.info(
                f"Agreement (best, epoch {best_epoch}): "
                f"{best_result['agreement']:.3f} "
                f"(multipv_hit={best_result['multipv_hit_rate']:.3f}, "
                f"regret_mean={best_result['regret_mean_cp']:.1f}cp)"
            )
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_train.py -v`
Expected: 全 PASS（`test_train_with_agreement` が regret 付きログでも通ること）

- [ ] **Step 5: CLAUDE.md に計測プロトコルを追記**

「## 評価関数の計測」セクションの導入文の直後に追加:

```markdown
### 計測プロトコル（フェーズ間比較の固定条件）

改善の効果比較は以下の固定条件で行う:

- **独立holdout**: 学習に一切使わない専用JSONL（`gen_dataset.py`で別途生成、
  MultiPV付き・固定シード）を`--agreement-data`とオフライン計測に使う。
  学習データと同一ファイルを指定すると`train.py`が警告を出す。
- **主指標はcp regret**: 教師視点での「最善手と選択手の評価差」。
  一致率は同等手が複数ある局面でノイズが入るが、regretはそれを吸収する。
  選択手がMultiPV候補外の局面はcensoredとして件数のみ記録。
  詰みスコア対策として`--regret-clamp`（デフォルト1000cp）で丸めて集計。
- 教師探索条件（MultiPV数・depth/nodes）と`--limit`・シードを揃える。
```

`scripts/move_agreement.py` の説明にある offline の記述の後に1行追加:

```markdown
オフラインモードではcp regret（mean/median、ply帯別、censored数）も出力される。
```

- [ ] **Step 6: コミット**

```bash
git add train/train.py tests/test_train.py CLAUDE.md
git commit -m "regretログとholdout同一ファイル警告を追加、計測プロトコルを記載"
```

---

### Task 4: value_target_from_cp と ShogiRankingPairDataset の delta_target（Phase 1a）

**Files:**
- Modify: `models/dataset.py`
- Modify: `models/__init__.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Consumes: 既存 `normalize_cp` / `cp_to_wdl`（models/value_transformer.py）
- Produces:
  - `value_target_from_cp(score_cp: float, target_mode: str = "cp", cp_scale: float = 1200.0, wdl_scale: float = 600.0) -> float`
  - `ShogiRankingPairDataset.__init__(..., cp_scale: float = 1200.0, target_mode: str = "cp", wdl_scale: float = 600.0)`
  - `ShogiRankingPairDataset.pairs` の要素が5タプル `(sfen, better_move, worse_move, better_score, worse_score)` になる（**破壊的変更**、既存テスト2件を更新）
  - `__getitem__` の返り値に `delta_target`（スカラーテンソル、負値）が追加
  - `ranking_collate_fn` が `delta_target` を `(batch,)` にスタック（Task 5, 9 が使用）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_ranking.py` の既存テスト2件を5タプル対応に更新:

```python
    def test_better_move_comes_first(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        triples = {p[:3] for p in dataset.pairs}
        assert ("startpos", "2g2f", "9g9f") in triples
        assert ("startpos moves 2g2f", "8c8d", "3c3d") in triples
```

（`test_pair_count_with_gap_filter` 等は件数のみの検証なので変更不要）

新クラスを追加:

```python
class TestDeltaTarget:
    """delta_target（ΔV回帰ターゲット）のテスト."""

    def _pair_index(
        self, dataset: ShogiRankingPairDataset, triple: tuple[str, str, str]
    ) -> int:
        return next(
            i for i, p in enumerate(dataset.pairs) if p[:3] == triple
        )

    def test_cp_mode_value(self, ranking_data: Path) -> None:
        # (2g2f: +50, 9g9f: -80) → n(−50) − n(+80)、n=tanh(cp/1200)
        dataset = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, cp_scale=1200.0
        )
        idx = self._pair_index(dataset, ("startpos", "2g2f", "9g9f"))
        expected = math.tanh(-50 / 1200.0) - math.tanh(80 / 1200.0)
        assert dataset[idx]["delta_target"].item() == pytest.approx(
            expected, abs=1e-6
        )

    def test_delta_negative_for_all_pairs(self, ranking_data: Path) -> None:
        # 良い手側の子局面ターゲットは常に小さい（相手番視点）
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        for i in range(len(dataset)):
            assert dataset[i]["delta_target"].item() < 0

    def test_wdl_mode_value(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, target_mode="wdl", wdl_scale=600.0
        )
        idx = self._pair_index(dataset, ("startpos", "2g2f", "9g9f"))

        def n(cp: float) -> float:
            return 2.0 / (1.0 + math.exp(-cp / 600.0)) - 1.0

        expected = n(-50.0) - n(80.0)
        assert dataset[idx]["delta_target"].item() == pytest.approx(
            expected, abs=1e-6
        )

    def test_collate_includes_delta_target(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        batch = ranking_collate_fn([dataset[0], dataset[1]])
        assert batch["delta_target"].shape == (2,)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_ranking.py -v`
Expected: `TestDeltaTarget` 全件 FAIL、`test_better_move_comes_first` は PASS のまま（3タプルの間は `p[:3] == p`）

- [ ] **Step 3: 実装**

`models/dataset.py` の `child_sfen` の後に追加:

```python
def value_target_from_cp(
    score_cp: float,
    target_mode: str = "cp",
    cp_scale: float = 1200.0,
    wdl_scale: float = 600.0,
) -> float:
    """score_cp（手番側視点）をモデルの値ターゲット空間へ変換.

    ShogiValueDatasetのtarget_mode変換（勝敗ブレンドなしの分岐局面と
    同じ規約）をペアデータセットでも使えるよう関数化したもの。

    Args:
        score_cp: 評価値（cp、その局面の手番側視点）
        target_mode: "cp"（tanh正規化）または "wdl"（勝率を[-1,1]にマップ）
        cp_scale: cpモードの正規化スケール
        wdl_scale: wdlモードのシグモイドスケール

    Returns:
        [-1, 1]の値ターゲット
    """
    if target_mode == "wdl":
        return 2.0 * cp_to_wdl(score_cp, wdl_scale) - 1.0
    return normalize_cp(score_cp, cp_scale)
```

`ShogiRankingPairDataset.__init__` に引数を追加（`min_gap_cp` の後）:

```python
        cp_scale: float = 1200.0,
        target_mode: str = "cp",
        wdl_scale: float = 600.0,
```

冒頭で検証と保存:

```python
        if target_mode not in ("cp", "wdl"):
            raise ValueError(f"Unknown target_mode: {target_mode}")
        self.cp_scale = cp_scale
        self.target_mode = target_mode
        self.wdl_scale = wdl_scale
```

`self.pairs` の型コメントと `_load_pairs` の append を5タプルに変更:

```python
        # (親局面sfen, 良い手, 悪い手, 良い手のscore_cp, 悪い手のscore_cp)
        self.pairs: list[tuple[str, str, str, float, float]] = []
```

```python
                        self.pairs.append((
                            sfen, better["move"], worse["move"],
                            float(better["score_cp"]), float(worse["score_cp"]),
                        ))
```

`__getitem__` のアンパックと返り値を変更:

```python
        parent_sfen, move_better, move_worse, score_better, score_worse = (
            self.pairs[idx]
        )
```

`result` 構築の後（`use_features` 分岐の前）に追加:

```python
        # ΔV回帰ターゲット。scoreは親の手番側視点なので、子局面の
        # 値ターゲットは手番反転でn(−s)。良い手側が小さくなる（負値）
        delta = value_target_from_cp(
            -score_better, self.target_mode, self.cp_scale, self.wdl_scale
        ) - value_target_from_cp(
            -score_worse, self.target_mode, self.cp_scale, self.wdl_scale
        )
        result["delta_target"] = torch.tensor(delta, dtype=torch.float32)
```

docstring の Returns にも `delta_target` を追記。クラス docstring の Args に3引数を追記。

`ranking_collate_fn` に追加（`features_a` 分岐の前）:

```python
    if "delta_target" in batch[0]:
        result["delta_target"] = torch.stack(
            [s["delta_target"] for s in batch]
        )
```

`models/__init__.py` の import と `__all__` に `value_target_from_cp` を追加（`from models.dataset import` の行に追加）。

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_ranking.py tests/test_models.py -v`
Expected: 全 PASS

- [ ] **Step 5: コミット**

```bash
git add models/dataset.py models/__init__.py tests/test_ranking.py
git commit -m "candidatesペアにΔV回帰ターゲットを追加"
```

---

### Task 5: compute_delta_loss と学習統合・--delta-weight（Phase 1a）

**Files:**
- Modify: `train/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Task 4 の `delta_target`（ranking バッチに含まれる）、既存 `_ranking_forward`
- Produces:
  - `compute_delta_loss(value_a: torch.Tensor, value_b: torch.Tensor, delta_target: torch.Tensor, huber_delta: float = 0.5) -> torch.Tensor`
  - `TrainConfig.delta_weight: float = 0.0`、CLI `--delta-weight`
  - `validate_ranking(...) -> tuple[float, float, float]`（**返り値が3要素に変更**: loss, accuracy, delta_mae）
  - `_build_ranking_loaders` のゲートが `ranking_weight > 0 or delta_weight > 0` に変更

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_train.py` に追加:

```python
class TestComputeDeltaLoss:
    """compute_delta_loss（ΔV差分回帰）のテスト."""

    def test_zero_when_diff_matches_target(self) -> None:
        from train.train import compute_delta_loss
        value_a = torch.tensor([-0.3, 0.1])
        value_b = torch.tensor([0.2, 0.3])
        target = torch.tensor([-0.5, -0.2])
        loss = compute_delta_loss(value_a, value_b, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_small_error_quadratic(self) -> None:
        from train.train import compute_delta_loss
        # 誤差0.2 < huber_delta=0.5 → 二乗領域: 0.5 * 0.2^2
        value_a = torch.tensor([0.0])
        value_b = torch.tensor([0.0])
        target = torch.tensor([-0.2])
        loss = compute_delta_loss(value_a, value_b, target, huber_delta=0.5)
        assert loss.item() == pytest.approx(0.5 * 0.2**2, abs=1e-6)
```

`TestTrainRanking` に追加:

```python
    def test_train_with_delta_loss(
        self, candidates_data: Path, tmp_path: Path
    ) -> None:
        """ΔV損失のみ（ranking無効）でも学習が完了しdelta_maeが記録される."""
        config = _small_config(
            candidates_data, tmp_path / "ckpt",
            ranking_weight=0.0, delta_weight=0.3, ranking_min_gap=30.0,
        )
        train(config)

        ckpt = torch.load(
            tmp_path / "ckpt" / "final.pt", map_location="cpu",
            weights_only=False,
        )
        assert ckpt["config"]["delta_weight"] == 0.3
        ranking_val = ckpt["state"]["ranking_val"]
        assert len(ranking_val) == config.epochs
        assert "delta_mae" in ranking_val[0]
        assert ranking_val[0]["delta_mae"] >= 0.0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_train.py::TestComputeDeltaLoss tests/test_train.py::TestTrainRanking::test_train_with_delta_loss -v`
Expected: FAIL（ImportError / TypeError: unexpected keyword 'delta_weight'）

- [ ] **Step 3: 実装**

`train/train.py` の `compute_ranking_loss` の後に追加:

```python
def compute_delta_loss(
    value_a: torch.Tensor,
    value_b: torch.Tensor,
    delta_target: torch.Tensor,
    huber_delta: float = 0.5,
) -> torch.Tensor:
    """ΔV（ペア間の評価値差分）のHuber損失を計算.

    絶対値の一致ではなく「似た局面間の評価差」を教師差分に合わせる。
    1手読みの手選びはargmaxで決まるため、差分方向の誤差が着手を
    直接左右する。ラベルを完全に学習できれば差分も合うが、有限容量では
    この損失が差分方向の誤差を優先的に抑える再重み付けとして働く。

    Args:
        value_a: ペアA側の評価値 (batch,)
        value_b: ペアB側の評価値 (batch,)
        delta_target: value_a − value_b の教師ターゲット (batch,)
        huber_delta: Huber lossの遷移点

    Returns:
        損失テンソル（スカラー）
    """
    return nn.functional.huber_loss(
        value_a - value_b, delta_target, delta=huber_delta
    )
```

`TrainConfig` に追加（`ranking_min_gap` の後）:

```python
    # ΔV差分回帰損失（0=無効）
    # candidatesペア（および--delta-dataの摂動ペア）の評価値差分を
    # Huber回帰する。rankingが順序のみ学ぶのに対し差分量まで合わせる
    delta_weight: float = 0.0
```

`train_epoch` のシグネチャに `delta_weight: float = 0.0` を追加し、ranking 分岐を次に置き換え:

```python
        # Pairwise ranking損失 + ΔV差分回帰損失（同じforwardを共有）
        ranking_loss: torch.Tensor | None = None
        delta_loss: torch.Tensor | None = None
        if ranking_iter is not None:
            try:
                ranking_batch = next(ranking_iter)
            except StopIteration:
                ranking_iter = iter(ranking_loader)
                ranking_batch = next(ranking_iter)
            value_better, value_worse = _ranking_forward(
                model, ranking_batch, device
            )
            if ranking_weight > 0:
                ranking_loss = compute_ranking_loss(value_better, value_worse)
            if delta_weight > 0 and "delta_target" in ranking_batch:
                delta_loss = compute_delta_loss(
                    value_better, value_worse,
                    ranking_batch["delta_target"].to(device),
                    huber_delta,
                )
```

損失合成を変更:

```python
        loss = value_loss + aux_loss_weight * outcome_loss - variance_reg_weight * variance_reg
        if ranking_loss is not None:
            loss = loss + ranking_weight * ranking_loss
        if delta_loss is not None:
            loss = loss + delta_weight * delta_loss
```

ステップログに delta を追加:

```python
            ranking_str = (
                f", ranking={ranking_loss.item():.6f}"
                if ranking_loss is not None else ""
            )
            delta_str = (
                f", delta={delta_loss.item():.6f}"
                if delta_loss is not None else ""
            )
            logger.info(
                f"Step {state.global_step}: loss={loss.item():.6f} "
                f"(value={value_loss.item():.6f}, outcome={outcome_loss.item():.6f}, "
                f"var_reg={variance_reg.item():.4f}{ranking_str}{delta_str})"
            )
```

`validate_ranking` を3要素返しに変更:

```python
@torch.no_grad()
def validate_ranking(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """rankingペアの検証（損失・ペア正答率・ΔV平均絶対誤差）.

    ペア正答率は「良い手の子局面の評価値が悪い手より低い」割合で、
    1手読みの手選びの正しさを直接反映する指標。delta_maeは
    差分量の予測誤差（ΔV回帰の検証指標）。

    Args:
        model: 評価するモデル
        loader: rankingペアの検証ローダー
        device: デバイス

    Returns:
        (ranking損失, ペア正答率, ΔV平均絶対誤差)
    """
    model.eval()
    total_loss = 0.0
    total_delta_err = 0.0
    correct = 0
    total = 0

    for batch in loader:
        value_better, value_worse = _ranking_forward(model, batch, device)
        loss = compute_ranking_loss(value_better, value_worse)
        total_loss += loss.item() * value_better.size(0)
        correct += (value_better < value_worse).sum().item()
        total += value_better.size(0)
        if "delta_target" in batch:
            target = batch["delta_target"].to(device)
            total_delta_err += (
                (value_better - value_worse - target).abs().sum().item()
            )

    if total == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total, correct / total, total_delta_err / total
```

`_build_ranking_loaders` のゲートとエラーメッセージ、データセット構築を変更:

```python
    if config.ranking_weight <= 0 and config.delta_weight <= 0:
        return None, None
```

```python
        raise ValueError(
            "--ranking-weight/--delta-weightには"
            "game_idが2対局以上あるデータが必要です"
        )
```

両方の `ShogiRankingPairDataset(...)` 構築に引数を追加:

```python
        cp_scale=config.cp_scale,
        target_mode=config.target_mode,
        wdl_scale=config.wdl_scale,
```

`len(ranking_train_dataset) == 0` のエラーメッセージ内の `--ranking-weightには` を `--ranking-weight/--delta-weightには` に変更。

`train()` の呼び出し側を更新。`train_epoch(...)` に `delta_weight=config.delta_weight,` を追加。検証部:

```python
        if ranking_val_loader is not None:
            rank_loss, rank_acc, delta_mae = validate_ranking(
                eval_model, ranking_val_loader, device
            )
            state.ranking_val.append(
                {"epoch": epoch + 1, "loss": rank_loss,
                 "accuracy": rank_acc, "delta_mae": delta_mae}
            )
            logger.info(
                f"Ranking val: loss={rank_loss:.6f}, "
                f"pair_accuracy={rank_acc:.3f}, delta_mae={delta_mae:.4f}"
            )
```

CLI に追加（`--ranking-min-gap` の後）:

```python
    parser.add_argument("--delta-weight", type=float, default=0.0,
                        help="ΔV差分回帰損失の重み（0=無効。candidatesペアの評価値差分を回帰）")
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_train.py tests/test_ranking.py -v`
Expected: 全 PASS

- [ ] **Step 5: コミット**

```bash
git add train/train.py tests/test_train.py
git commit -m "ΔV差分回帰損失を追加（--delta-weight、rankingとforward共有）"
```

---

### Task 6: 摂動ペア生成の純粋関数群（Phase 1b）

**Files:**
- Create: `tools/gen_perturb_pairs.py`
- Test: `tests/test_gen_perturb_pairs.py`（新規）

**Interfaces:**
- Consumes: `models.dataset.child_sfen`、`scripts.move_agreement.board_from_sfen_line`
- Produces（Task 7 が使用）:
  - `material_balance(board: shogi.Board) -> int`
  - `move_destination_pairs(board: shogi.Board, max_pairs_per_group: int = 1, rng: random.Random | None = None) -> list[tuple[str, str]]`
  - `promotion_pairs(board: shogi.Board) -> list[tuple[str, str]]`
  - `rewind_branch_pairs(records: list[dict]) -> list[dict]`
  - `board_perturb_pairs(record: dict, max_pairs_per_group: int = 1, rng: random.Random | None = None) -> list[dict]`
  - `is_stable(delta_label: int, delta_stability: int) -> bool`
  - ペア dict のキー: `sfen_a`, `sfen_b`, `pair_type`, `game_id`（ラベル前）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_perturb_pairs.py` を新規作成:

```python
"""摂動ペア生成（局面感度蒸留）のテスト."""

from __future__ import annotations

import shogi

from tools.gen_perturb_pairs import (
    board_perturb_pairs,
    is_stable,
    material_balance,
    move_destination_pairs,
    promotion_pairs,
    rewind_branch_pairs,
)


class TestMaterialBalance:
    """material_balanceのテスト."""

    def test_startpos_is_zero(self) -> None:
        assert material_balance(shogi.Board()) == 0

    def test_hand_piece_counted(self) -> None:
        # 先手が歩を1枚持っている（盤上は玉のみ対称）
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        assert material_balance(board) == 90

    def test_side_to_move_view(self) -> None:
        # 同一局面でも手番側視点なので符号が反転する
        board_black = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        board_white = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 w P 1")
        assert material_balance(board_black) == -material_balance(board_white)


class TestMoveDestinationPairs:
    """move_destination_pairsのテスト."""

    def test_same_from_square(self) -> None:
        pairs = move_destination_pairs(shogi.Board(), max_pairs_per_group=2)
        assert len(pairs) > 0
        for usi_a, usi_b in pairs:
            # 同一グループ（同じ駒）のペアなので移動元表記が一致する
            assert usi_a[:2] == usi_b[:2]
            assert usi_a != usi_b

    def test_max_pairs_per_group(self) -> None:
        few = move_destination_pairs(shogi.Board(), max_pairs_per_group=1)
        many = move_destination_pairs(shogi.Board(), max_pairs_per_group=10)
        assert len(few) <= len(many)


class TestPromotionPairs:
    """promotion_pairsのテスト."""

    def test_startpos_has_none(self) -> None:
        assert promotion_pairs(shogi.Board()) == []

    def test_pawn_promotion_pair(self) -> None:
        # 5d歩が5cへ進む手は成/不成の両方が合法
        board = shogi.Board("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        pairs = promotion_pairs(board)
        assert ("5d5c+", "5d5c") in pairs


class TestRewindBranchPairs:
    """rewind_branch_pairsのテスト."""

    def test_pairs_from_candidates(self) -> None:
        records = [
            {
                "sfen": "startpos", "ply": 0, "game_id": 3,
                "candidates": [
                    {"move": "2g2f", "score_cp": 50, "rank": 1},
                    {"move": "7g7f", "score_cp": 30, "rank": 2},
                ],
            },
            {"sfen": "startpos moves 2g2f", "ply": 1, "game_id": 3},
        ]
        pairs = rewind_branch_pairs(records)
        # 本譜(2g2f)と異なる候補7g7fの分岐のみペアになる
        assert len(pairs) == 1
        assert pairs[0]["sfen_a"] == "startpos moves 2g2f"
        assert pairs[0]["sfen_b"] == "startpos moves 7g7f"
        assert pairs[0]["pair_type"] == "rewind_branch"
        assert pairs[0]["game_id"] == 3

    def test_no_next_record_no_pair(self) -> None:
        records = [
            {
                "sfen": "startpos", "ply": 0, "game_id": 0,
                "candidates": [{"move": "2g2f", "score_cp": 50, "rank": 1}],
            },
        ]
        assert rewind_branch_pairs(records) == []


class TestBoardPerturbPairs:
    """board_perturb_pairsのテスト."""

    def test_generates_move_dest_pairs(self) -> None:
        record = {"sfen": "startpos", "game_id": 7}
        pairs = board_perturb_pairs(record, max_pairs_per_group=1)
        assert len(pairs) > 0
        for pair in pairs:
            assert pair["game_id"] == 7
            assert pair["pair_type"] in ("move_dest", "promotion")
            assert pair["sfen_a"].startswith("startpos moves ")


class TestIsStable:
    """is_stable（安定性フィルタ）のテスト."""

    def test_same_sign_stable(self) -> None:
        assert is_stable(100, 50)
        assert is_stable(-100, -20)

    def test_sign_flip_unstable(self) -> None:
        assert not is_stable(100, -50)

    def test_zero_is_stable(self) -> None:
        assert is_stable(0, 100)
        assert is_stable(100, 0)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_gen_perturb_pairs.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'tools.gen_perturb_pairs'）

- [ ] **Step 3: 実装**

`tools/gen_perturb_pairs.py` を新規作成:

```python
"""局面感度蒸留用の摂動ペア生成スクリプト.

学習JSONLの本譜局面から、到達可能性の安全リスト内の摂動ペアを作り、
エンジンで両局面をラベル付けしてΔV学習用のペアJSONLを出力する。

ペア種別:
    rewind_branch: 本譜の次局面 vs 同一親局面から別候補手で分岐した局面
    move_dest:     同一駒を異なるマスへ動かす合法手ペアの子局面
    promotion:     同じ移動で成/不成だけが異なる合法手ペアの子局面（素材一致）

使用例:
    python tools/gen_perturb_pairs.py \\
        --data data/raw/dataset_mpv.jsonl \\
        -o data/raw/perturb_pairs.jsonl \\
        --label-nodes 200000 --stability-nodes 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import shogi

from models.dataset import child_sfen
from scripts.move_agreement import board_from_sfen_line, engine_position_args
from shogi_utils import USIEngine, get_engine_path
from tools.gen_dataset import mate_to_cp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 駒価値（駒割の一致判定に使う。厳密なチューニング値である必要はない）
PIECE_VALUES: dict[int, int] = {
    shogi.PAWN: 90,
    shogi.LANCE: 315,
    shogi.KNIGHT: 405,
    shogi.SILVER: 495,
    shogi.GOLD: 540,
    shogi.BISHOP: 855,
    shogi.ROOK: 990,
    shogi.KING: 0,
    shogi.PROM_PAWN: 540,
    shogi.PROM_LANCE: 540,
    shogi.PROM_KNIGHT: 540,
    shogi.PROM_SILVER: 540,
    shogi.PROM_BISHOP: 945,
    shogi.PROM_ROOK: 1080,
}


def material_balance(board: shogi.Board) -> int:
    """手番側から見た駒割（盤上+持ち駒、玉除く）を計算.

    Args:
        board: 対象局面

    Returns:
        手番側の駒価値合計 − 相手側の駒価値合計（cp相当）
    """
    balance = 0
    for square in shogi.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = PIECE_VALUES[piece.piece_type]
        balance += value if piece.color == board.turn else -value
    for color in shogi.COLORS:
        sign = 1 if color == board.turn else -1
        for piece_type, count in board.pieces_in_hand[color].items():
            balance += sign * PIECE_VALUES[piece_type] * count
    return balance


def move_destination_pairs(
    board: shogi.Board,
    max_pairs_per_group: int = 1,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    """同一駒を異なるマスへ動かす合法手ペアを列挙.

    打つ手は同じ駒種の打ち先違いを同一グループとして扱う。
    成/不成の違いはpromotion_pairsの担当なので、同じ成り状態同士のみ
    ペアにする。

    Args:
        board: 親局面
        max_pairs_per_group: 1グループ（同一駒）あたりの最大ペア数
        rng: サンプリング用乱数（Noneで先頭から選択）

    Returns:
        (USI手a, USI手b)のリスト
    """
    groups: dict[tuple, list[shogi.Move]] = defaultdict(list)
    for move in board.legal_moves:
        if move.from_square is None:
            key = ("drop", move.drop_piece_type)
        else:
            key = ("move", move.from_square, bool(move.promotion))
        groups[key].append(move)

    pairs: list[tuple[str, str]] = []
    for moves in groups.values():
        if len(moves) < 2:
            continue
        combos = [
            (a, b) for i, a in enumerate(moves) for b in moves[i + 1:]
        ]
        if rng is not None:
            rng.shuffle(combos)
        for move_a, move_b in combos[:max_pairs_per_group]:
            pairs.append((move_a.usi(), move_b.usi()))
    return pairs


def promotion_pairs(board: shogi.Board) -> list[tuple[str, str]]:
    """同じ移動で成/不成だけが異なる合法手ペアを列挙（素材一致）.

    Args:
        board: 親局面

    Returns:
        (成る手のUSI, 成らない手のUSI)のリスト
    """
    by_fromto: dict[tuple[int, int], dict[bool, shogi.Move]] = defaultdict(dict)
    for move in board.legal_moves:
        if move.from_square is None:
            continue
        by_fromto[(move.from_square, move.to_square)][
            bool(move.promotion)
        ] = move
    return [
        (variants[True].usi(), variants[False].usi())
        for variants in by_fromto.values()
        if True in variants and False in variants
    ]


def rewind_branch_pairs(records: list[dict]) -> list[dict]:
    """本譜レコード列から（本譜の次局面, 分岐局面）ペアを構築.

    candidates付きレコードr(t)と本譜の次レコードr(t+1)について、
    candidatesの本譜と異なる手で分岐した局面とr(t+1)のペアを作る。
    どのplyでも分岐させるため、設計書の「k手巻き戻し」はこの
    per-record適用で網羅される。両局面は同一ply・同一手番。

    Args:
        records: 同一対局の本譜レコード（sourceフィールド無し）のリスト

    Returns:
        sfen_a（本譜側）/sfen_b（分岐側）/pair_type/game_idを持つ
        dictのリスト（評価値ラベルは未付与）
    """
    pairs: list[dict] = []
    by_ply = {r["ply"]: r for r in records}
    for ply, record in by_ply.items():
        candidates = record.get("candidates")
        next_record = by_ply.get(ply + 1)
        if not candidates or next_record is None:
            continue
        next_sfen_parts = next_record["sfen"].split()
        if len(next_sfen_parts) < 2:
            continue
        played = next_sfen_parts[-1]
        for cand in candidates:
            if cand["move"] == played:
                continue
            pairs.append({
                "sfen_a": next_record["sfen"],
                "sfen_b": child_sfen(record["sfen"], cand["move"]),
                "pair_type": "rewind_branch",
                "game_id": record.get("game_id", 0),
            })
    return pairs


def board_perturb_pairs(
    record: dict,
    max_pairs_per_group: int = 1,
    rng: random.Random | None = None,
) -> list[dict]:
    """1つの本譜レコードからmove_dest/promotionペアを構築.

    Args:
        record: 本譜レコード（sfenとgame_idを使用）
        max_pairs_per_group: move_destの1駒あたり最大ペア数
        rng: サンプリング用乱数

    Returns:
        sfen_a/sfen_b/pair_type/game_idを持つdictのリスト（ラベル未付与）
    """
    board = board_from_sfen_line(record["sfen"])
    game_id = record.get("game_id", 0)
    pairs: list[dict] = []
    for move_a, move_b in move_destination_pairs(
        board, max_pairs_per_group, rng
    ):
        pairs.append({
            "sfen_a": child_sfen(record["sfen"], move_a),
            "sfen_b": child_sfen(record["sfen"], move_b),
            "pair_type": "move_dest",
            "game_id": game_id,
        })
    for move_a, move_b in promotion_pairs(board):
        pairs.append({
            "sfen_a": child_sfen(record["sfen"], move_a),
            "sfen_b": child_sfen(record["sfen"], move_b),
            "pair_type": "promotion",
            "game_id": game_id,
        })
    return pairs


def is_stable(delta_label: int, delta_stability: int) -> bool:
    """2回の評価で差分の符号が反転していないか判定.

    どちらかの差分が0の場合は「反転」とは言えないため安定扱いとする。

    Args:
        delta_label: ラベル探索での score_a − score_b
        delta_stability: 安定性確認探索での score_a − score_b

    Returns:
        安定ならTrue
    """
    if delta_label == 0 or delta_stability == 0:
        return True
    return (delta_label > 0) == (delta_stability > 0)
```

（`argparse` / `Callable` / `Optional` / `USIEngine` / `engine_position_args` /
`mate_to_cp` はこの時点では未使用だが Task 7 で使うため import 済みにしておく。
lint が未使用 import を検出する環境なら Task 7 まで import を遅らせてもよい）

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_gen_perturb_pairs.py -v`
Expected: 全 PASS

- [ ] **Step 5: コミット**

```bash
git add tools/gen_perturb_pairs.py tests/test_gen_perturb_pairs.py
git commit -m "摂動ペア生成の純粋関数群を追加（巻き戻し分岐・移動先違い・成不成）"
```

---

### Task 7: 摂動ペアのラベル付けと CLI（Phase 1b）

**Files:**
- Modify: `tools/gen_perturb_pairs.py`
- Test: `tests/test_gen_perturb_pairs.py`

**Interfaces:**
- Consumes: Task 6 の全関数、`USIEngine.go(nodes=N)`、`mate_to_cp(score_cp, score_mate)`
- Produces:
  - `label_pairs(pairs: list[dict], evaluate: Callable[[str, int], Optional[int]], label_nodes: int, stability_nodes: int) -> tuple[list[dict], int]`
  - `load_mainline_records(data_path: Path) -> dict[int, list[dict]]`
  - `build_pairs(games: dict[int, list[dict]], max_pairs_per_game: int, seed: int) -> list[dict]`
  - 出力 JSONL レコード: `{"sfen_a", "sfen_b", "score_cp_a", "score_cp_b", "pair_type", "game_id", "material_diff"}`（score は各局面の手番側視点。Task 8 が読む）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_perturb_pairs.py` に追加:

```python
import json
from pathlib import Path

import pytest

from tools.gen_perturb_pairs import (
    build_pairs,
    label_pairs,
    load_mainline_records,
)


class TestLabelPairs:
    """label_pairs（エンジンラベル付け+安定性フィルタ）のテスト."""

    @staticmethod
    def _make_evaluate(scores: dict[str, dict[int, int]]):
        def evaluate(sfen: str, nodes: int) -> int | None:
            return scores.get(sfen, {}).get(nodes)
        return evaluate

    def test_labels_and_material_diff(self) -> None:
        pairs = [{
            "sfen_a": "startpos moves 2g2f",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "rewind_branch",
            "game_id": 0,
        }]
        evaluate = self._make_evaluate({
            "startpos moves 2g2f": {200: -30, 50: -20},
            "startpos moves 7g7f": {200: -60, 50: -40},
        })
        labeled, dropped = label_pairs(
            pairs, evaluate, label_nodes=200, stability_nodes=50
        )
        assert dropped == 0
        assert len(labeled) == 1
        assert labeled[0]["score_cp_a"] == -30
        assert labeled[0]["score_cp_b"] == -60
        # 序盤の歩の差し替えなので素材は一致
        assert labeled[0]["material_diff"] == 0

    def test_unstable_pair_dropped(self) -> None:
        pairs = [{
            "sfen_a": "startpos moves 2g2f",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "rewind_branch",
            "game_id": 0,
        }]
        # ラベル探索と安定性探索で差分の符号が反転 → 破棄
        evaluate = self._make_evaluate({
            "startpos moves 2g2f": {200: -30, 50: -80},
            "startpos moves 7g7f": {200: -60, 50: -40},
        })
        labeled, dropped = label_pairs(
            pairs, evaluate, label_nodes=200, stability_nodes=50
        )
        assert labeled == []
        assert dropped == 1

    def test_turn_mismatch_raises(self) -> None:
        pairs = [{
            "sfen_a": "startpos",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "move_dest",
            "game_id": 0,
        }]
        with pytest.raises(ValueError, match="手番"):
            label_pairs(pairs, lambda s, n: 0, 200, 50)

    def test_none_score_dropped(self) -> None:
        pairs = [{
            "sfen_a": "startpos moves 2g2f",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "rewind_branch",
            "game_id": 0,
        }]
        labeled, dropped = label_pairs(pairs, lambda s, n: None, 200, 50)
        assert labeled == []
        assert dropped == 1


class TestLoadAndBuild:
    """load_mainline_records / build_pairsのテスト."""

    @pytest.fixture
    def data_path(self, tmp_path: Path) -> Path:
        records = [
            {
                "sfen": "startpos", "score_cp": 50, "ply": 0, "game_id": 0,
                "candidates": [
                    {"move": "2g2f", "score_cp": 50, "rank": 1},
                    {"move": "7g7f", "score_cp": 30, "rank": 2},
                ],
            },
            {"sfen": "startpos moves 2g2f", "score_cp": -40, "ply": 1,
             "game_id": 0},
            # 分岐レコード（source付き）は本譜として扱わない
            {"sfen": "startpos moves 2g2f 3c3d", "score_cp": 20, "ply": 2,
             "game_id": 0, "source": "multipv"},
            {"sfen": "startpos", "score_cp": 10, "ply": 0, "game_id": 1},
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records))
        return path

    def test_load_mainline_records(self, data_path: Path) -> None:
        games = load_mainline_records(data_path)
        assert set(games.keys()) == {0, 1}
        assert len(games[0]) == 2  # source付きは除外
        assert games[0][0]["ply"] == 0

    def test_build_pairs_caps_per_game(self, data_path: Path) -> None:
        games = load_mainline_records(data_path)
        pairs = build_pairs(games, max_pairs_per_game=3, seed=42)
        per_game: dict[int, int] = {}
        for pair in pairs:
            per_game[pair["game_id"]] = per_game.get(pair["game_id"], 0) + 1
        assert all(count <= 3 for count in per_game.values())
        assert len(pairs) > 0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_gen_perturb_pairs.py -v`
Expected: 新規クラスが FAIL（ImportError: cannot import name 'label_pairs'）

- [ ] **Step 3: 実装**

`tools/gen_perturb_pairs.py` に追加:

```python
def label_pairs(
    pairs: list[dict],
    evaluate: Callable[[str, int], Optional[int]],
    label_nodes: int,
    stability_nodes: int,
) -> tuple[list[dict], int]:
    """ペア両局面をラベル付けし、安定性フィルタを通過したものを返す.

    手番一致の検証とmaterial_diffの付与もここで行う。

    Args:
        pairs: sfen_a/sfen_b/pair_type/game_idを持つペアのリスト
        evaluate: (sfen, nodes) → score_cp（手番側視点、評価不能はNone）
        label_nodes: ラベル用探索のノード数
        stability_nodes: 安定性確認用探索のノード数

    Returns:
        (ラベル付きペアのリスト, 破棄されたペア数)
    """
    labeled: list[dict] = []
    dropped = 0
    for pair in pairs:
        board_a = board_from_sfen_line(pair["sfen_a"])
        board_b = board_from_sfen_line(pair["sfen_b"])
        if board_a.turn != board_b.turn:
            raise ValueError(
                f"ペアの手番が一致しません: "
                f"{pair['sfen_a']} / {pair['sfen_b']}"
            )

        score_a = evaluate(pair["sfen_a"], label_nodes)
        score_b = evaluate(pair["sfen_b"], label_nodes)
        if score_a is None or score_b is None:
            dropped += 1
            continue

        stab_a = evaluate(pair["sfen_a"], stability_nodes)
        stab_b = evaluate(pair["sfen_b"], stability_nodes)
        if (
            stab_a is None
            or stab_b is None
            or not is_stable(score_a - score_b, stab_a - stab_b)
        ):
            dropped += 1
            continue

        labeled.append({
            **pair,
            "score_cp_a": score_a,
            "score_cp_b": score_b,
            "material_diff": (
                material_balance(board_a) - material_balance(board_b)
            ),
        })
    return labeled, dropped


def load_mainline_records(data_path: Path) -> dict[int, list[dict]]:
    """本譜レコード（sourceなし）をgame_idごとにply昇順で読み込む.

    Args:
        data_path: 学習JSONLのパス

    Returns:
        game_id → 本譜レコードのリスト（ply昇順）
    """
    games: dict[int, list[dict]] = defaultdict(list)
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("source") is not None:
                continue
            games[record.get("game_id", 0)].append(record)
    for records in games.values():
        records.sort(key=lambda r: r["ply"])
    return games


def build_pairs(
    games: dict[int, list[dict]],
    max_pairs_per_game: int,
    seed: int,
) -> list[dict]:
    """全対局から摂動ペア候補を構築し、対局ごとに上限までサンプリングする.

    Args:
        games: game_id → 本譜レコードのリスト
        max_pairs_per_game: 1対局あたりの最大ペア数
        seed: サンプリング用シード

    Returns:
        ペア候補のリスト（ラベル未付与）
    """
    rng = random.Random(seed)
    all_pairs: list[dict] = []
    for game_id, records in sorted(games.items()):
        game_pairs = rewind_branch_pairs(records)
        for record in records:
            game_pairs.extend(board_perturb_pairs(record, rng=rng))
        if len(game_pairs) > max_pairs_per_game:
            game_pairs = rng.sample(game_pairs, max_pairs_per_game)
        all_pairs.extend(game_pairs)
    return all_pairs


def make_engine_evaluator(
    engine: USIEngine,
) -> Callable[[str, int], Optional[int]]:
    """USIEngineをlabel_pairs用のevaluate関数に変換.

    Args:
        engine: 初期化済みのUSIEngine

    Returns:
        (sfen, nodes) → score_cp（詰みはmate_to_cpで±30000に変換）
    """
    def evaluate(sfen: str, nodes: int) -> Optional[int]:
        engine_sfen, moves = engine_position_args(sfen)
        engine.set_position(sfen=engine_sfen, moves=moves)
        result = engine.go(nodes=nodes)
        return mate_to_cp(result.score_cp, result.score_mate)
    return evaluate


def main() -> None:
    """エントリーポイント."""
    parser = argparse.ArgumentParser(description="摂動ペア生成（局面感度蒸留用）")
    parser.add_argument("--data", type=str, required=True,
                        help="入力JSONL（学習データ。game_idを引き継ぐ）")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="出力ペアJSONLのパス")
    parser.add_argument("--engine-type", type=str, default="suisho5",
                        choices=["suisho5", "hao"], help="ラベル付けエンジン")
    parser.add_argument("--engine", type=str, default=None,
                        help="エンジンパスの直接指定（--engine-typeより優先）")
    parser.add_argument("--label-nodes", type=int, default=200000,
                        help="ラベル用探索のノード数")
    parser.add_argument("--stability-nodes", type=int, default=50000,
                        help="安定性確認用探索のノード数")
    parser.add_argument("--max-pairs-per-game", type=int, default=40,
                        help="1対局あたりの最大ペア数")
    parser.add_argument("--seed", type=int, default=42, help="サンプリング用シード")
    args = parser.parse_args()

    games = load_mainline_records(Path(args.data))
    pairs = build_pairs(games, args.max_pairs_per_game, args.seed)
    logger.info(f"Pair candidates: {len(pairs)} from {len(games)} games")

    engine_path = (
        Path(args.engine) if args.engine else get_engine_path(args.engine_type)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with USIEngine(engine_path) as engine:
        engine.init_usi()
        engine.set_option("USI_OwnBook", False)
        engine.is_ready()
        labeled, dropped = label_pairs(
            pairs,
            make_engine_evaluator(engine),
            args.label_nodes,
            args.stability_nodes,
        )

    type_counts: dict[str, int] = {}
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in labeled:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            type_counts[pair["pair_type"]] = (
                type_counts.get(pair["pair_type"], 0) + 1
            )

    logger.info(
        f"Labeled pairs: {len(labeled)} (dropped by stability filter: "
        f"{dropped}) -> {output_path}"
    )
    logger.info(f"Pair types: {type_counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_gen_perturb_pairs.py -v`
Expected: 全 PASS

- [ ] **Step 5: 実エンジンでの疎通確認（手動スモーク）**

Run:
```bash
head -20 data/raw/*.jsonl | head -1  # 入力データがあることを確認
python tools/gen_perturb_pairs.py \
    --data $(ls data/raw/*.jsonl | head -1) \
    -o /tmp/perturb_smoke.jsonl \
    --max-pairs-per-game 4 --label-nodes 20000 --stability-nodes 5000
head -3 /tmp/perturb_smoke.jsonl
```
Expected: ペア JSONL が出力され、各行に `score_cp_a` / `score_cp_b` / `material_diff` がある。
（`data/raw/` に JSONL が無い環境ではこのステップをスキップし、その旨を報告する）

- [ ] **Step 6: コミット**

```bash
git add tools/gen_perturb_pairs.py tests/test_gen_perturb_pairs.py
git commit -m "摂動ペアのエンジンラベル付けと安定性フィルタ・CLIを追加"
```

---

### Task 8: ShogiDeltaPairDataset（Phase 1b）

**Files:**
- Modify: `models/dataset.py`
- Modify: `models/__init__.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Consumes: Task 4 の `value_target_from_cp`、Task 7 の出力 JSONL 形式
- Produces:
  - `prepare_pair_position(sfen: str, use_features: bool, normalize_turn: bool, apply_flip: bool) -> dict[str, torch.Tensor]`（モジュールレベル関数。`ShogiRankingPairDataset._prepare_position` はこれの呼び出しに置き換え）
  - `ShogiDeltaPairDataset(data_path, use_features=False, normalize_turn=False, augment_flip=False, cp_scale=1200.0, target_mode="cp", wdl_scale=600.0, same_material_only=False, include_game_ids=None, exclude_game_ids=None)`
  - `__getitem__` は `board_a/hand_a/turn_a/board_b/hand_b/turn_b/delta_target`（+ `features_a/features_b`）を返し、**`ranking_collate_fn` と `_ranking_forward` をそのまま流用できる**

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_ranking.py` の import に `ShogiDeltaPairDataset` を追加（`from models import` 行）し、末尾に追加:

```python
@pytest.fixture
def delta_pair_data(tmp_path: Path) -> Path:
    """gen_perturb_pairs.py出力形式の小規模ペアデータを生成."""
    records = [
        # 手番側視点スコア: a=-30, b=-60 → delta_target = n(-30) − n(-60) > 0
        {"sfen_a": "startpos moves 2g2f", "sfen_b": "startpos moves 7g7f",
         "score_cp_a": -30, "score_cp_b": -60,
         "pair_type": "rewind_branch", "game_id": 0, "material_diff": 0},
        {"sfen_a": "startpos moves 2g2f 8c8d",
         "sfen_b": "startpos moves 2g2f 3c3d",
         "score_cp_a": 40, "score_cp_b": 10,
         "pair_type": "move_dest", "game_id": 1, "material_diff": 90},
    ]
    path = tmp_path / "delta_pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


class TestShogiDeltaPairDataset:
    """ShogiDeltaPairDataset（摂動ペア）のテスト."""

    def test_len_and_shapes(self, delta_pair_data: Path) -> None:
        dataset = ShogiDeltaPairDataset(delta_pair_data)
        assert len(dataset) == 2
        item = dataset[0]
        assert item["board_a"].shape == (81,)
        assert item["hand_b"].shape == (14,)
        assert item["turn_a"].item() == item["turn_b"].item()

    def test_delta_target_no_sign_flip(self, delta_pair_data: Path) -> None:
        # scoreは各局面自身の手番側視点なので符号反転しない
        dataset = ShogiDeltaPairDataset(delta_pair_data, cp_scale=1200.0)
        expected = math.tanh(-30 / 1200.0) - math.tanh(-60 / 1200.0)
        assert dataset[0]["delta_target"].item() == pytest.approx(
            expected, abs=1e-6
        )

    def test_same_material_only(self, delta_pair_data: Path) -> None:
        dataset = ShogiDeltaPairDataset(
            delta_pair_data, same_material_only=True
        )
        assert len(dataset) == 1

    def test_game_id_filters(self, delta_pair_data: Path) -> None:
        train_ds = ShogiDeltaPairDataset(
            delta_pair_data, exclude_game_ids={0}
        )
        val_ds = ShogiDeltaPairDataset(
            delta_pair_data, include_game_ids={0}
        )
        assert len(train_ds) == 1
        assert len(val_ds) == 1

    def test_augment_flip_doubles(self, delta_pair_data: Path) -> None:
        dataset = ShogiDeltaPairDataset(delta_pair_data, augment_flip=True)
        assert len(dataset) == 4
        # 反転してもΔVターゲットは不変
        assert dataset[2]["delta_target"].item() == pytest.approx(
            dataset[0]["delta_target"].item()
        )

    def test_normalize_turn_forces_black_view(
        self, delta_pair_data: Path
    ) -> None:
        # 後手番ペアも先手視点に正規化され、ΔVターゲットは不変
        base = ShogiDeltaPairDataset(delta_pair_data)
        dataset = ShogiDeltaPairDataset(delta_pair_data, normalize_turn=True)
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["turn_a"].item() == 0
            assert item["turn_b"].item() == 0
            assert item["delta_target"].item() == pytest.approx(
                base[i]["delta_target"].item()
            )

    def test_collate_compatible(self, delta_pair_data: Path) -> None:
        dataset = ShogiDeltaPairDataset(delta_pair_data)
        batch = ranking_collate_fn([dataset[0], dataset[1]])
        assert batch["board_a"].shape == (2, 81)
        assert batch["delta_target"].shape == (2,)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_ranking.py::TestShogiDeltaPairDataset -v`
Expected: FAIL（ImportError: cannot import name 'ShogiDeltaPairDataset'）

- [ ] **Step 3: 実装**

`models/dataset.py`: `ShogiRankingPairDataset` の前にモジュールレベル関数を追加し、既存 `_prepare_position` の本体を移す:

```python
def prepare_pair_position(
    sfen: str,
    use_features: bool,
    normalize_turn: bool,
    apply_flip: bool,
) -> dict[str, torch.Tensor]:
    """ペアデータセット共通の1局面分モデル入力を作成.

    Args:
        sfen: 局面（"startpos [moves ...]" または "sfen ..."）
        use_features: 拡張特徴量を付与するかどうか
        normalize_turn: 後手番を先手視点に正規化するかどうか
        apply_flip: 左右反転を適用するかどうか

    Returns:
        board/hand/turn（use_features時はfeaturesも）の辞書
    """
    parsed = parse_sfen(sfen)
    board = parsed.board
    hand = parsed.hand
    turn = parsed.turn

    if normalize_turn:
        # ダミーのラベルを渡して盤面のみ正規化
        board, hand, turn, _, _ = normalize_to_black_view(
            board, hand, turn, 0.0, 0.5
        )

    if apply_flip:
        board, hand = augment_horizontal_flip(board, hand)

    result = {"board": board, "hand": hand, "turn": turn}
    if use_features:
        result["features"] = stack_features(board)
    return result
```

`ShogiRankingPairDataset._prepare_position` を削除し、`__getitem__` の呼び出しを置き換え:

```python
        pos_a = prepare_pair_position(
            child_sfen(parent_sfen, move_better),
            self.use_features, self.normalize_turn, apply_flip,
        )
        pos_b = prepare_pair_position(
            child_sfen(parent_sfen, move_worse),
            self.use_features, self.normalize_turn, apply_flip,
        )
```

`ShogiRankingPairDataset` の後に新クラスを追加:

```python
class ShogiDeltaPairDataset(Dataset):
    """摂動ペア（局面感度蒸留）のΔV回帰データセット.

    gen_perturb_pairs.pyの出力JSONLを読み込む。score_cp_a/score_cp_bは
    各局面自身の手番側視点（モデル出力と同じ視点）なので、
    delta_target = n(score_a) − n(score_b) を符号反転なしで回帰する
    （candidatesペアのdelta_targetとは規約が異なる点に注意）。

    Args:
        data_path: ペアJSONLのパス
        use_features: 拡張特徴量を使用するかどうか
        normalize_turn: 後手番を先手視点に正規化（メインデータセットと揃える）
        augment_flip: 左右反転でペアを2倍に拡張
        cp_scale: cpモードの正規化スケール
        target_mode: 値ターゲット空間（"cp" / "wdl"）
        wdl_scale: wdlモードのシグモイドスケール
        same_material_only: material_diff==0のペアのみ使用
        include_game_ids: 指定時、このgame_idのペアのみ使用（検証用）
        exclude_game_ids: 指定時、このgame_idのペアを除外（訓練用）
    """

    def __init__(
        self,
        data_path: str | Path,
        use_features: bool = False,
        normalize_turn: bool = False,
        augment_flip: bool = False,
        cp_scale: float = 1200.0,
        target_mode: str = "cp",
        wdl_scale: float = 600.0,
        same_material_only: bool = False,
        include_game_ids: set[int] | None = None,
        exclude_game_ids: set[int] | None = None,
    ) -> None:
        if target_mode not in ("cp", "wdl"):
            raise ValueError(f"Unknown target_mode: {target_mode}")

        self.data_path = Path(data_path)
        self.use_features = use_features
        self.normalize_turn = normalize_turn
        self.augment_flip = augment_flip
        self.cp_scale = cp_scale
        self.target_mode = target_mode
        self.wdl_scale = wdl_scale
        self.same_material_only = same_material_only
        # (sfen_a, sfen_b, score_cp_a, score_cp_b)
        self.pairs: list[tuple[str, str, float, float]] = []

        self._load_pairs(include_game_ids, exclude_game_ids)

    def _load_pairs(
        self,
        include_game_ids: set[int] | None,
        exclude_game_ids: set[int] | None,
    ) -> None:
        """ペアJSONLを読み込む."""
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                game_id = sample.get("game_id", 0)
                if (
                    include_game_ids is not None
                    and game_id not in include_game_ids
                ):
                    continue
                if (
                    exclude_game_ids is not None
                    and game_id in exclude_game_ids
                ):
                    continue
                if (
                    self.same_material_only
                    and sample.get("material_diff", 0) != 0
                ):
                    continue

                self.pairs.append((
                    sample["sfen_a"], sample["sfen_b"],
                    float(sample["score_cp_a"]), float(sample["score_cp_b"]),
                ))

    def __len__(self) -> int:
        base_len = len(self.pairs)
        if self.augment_flip:
            return base_len * 2
        return base_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """指定インデックスのペアを取得.

        Returns:
            board_a/hand_a/turn_a/board_b/hand_b/turn_b/delta_targetの辞書
            （use_features時はfeatures_a/features_bも）
        """
        apply_flip = False
        if self.augment_flip:
            base_len = len(self.pairs)
            if idx >= base_len:
                idx = idx - base_len
                apply_flip = True

        sfen_a, sfen_b, score_a, score_b = self.pairs[idx]

        pos_a = prepare_pair_position(
            sfen_a, self.use_features, self.normalize_turn, apply_flip
        )
        pos_b = prepare_pair_position(
            sfen_b, self.use_features, self.normalize_turn, apply_flip
        )

        delta = value_target_from_cp(
            score_a, self.target_mode, self.cp_scale, self.wdl_scale
        ) - value_target_from_cp(
            score_b, self.target_mode, self.cp_scale, self.wdl_scale
        )

        result = {
            "board_a": pos_a["board"],
            "hand_a": pos_a["hand"],
            "turn_a": pos_a["turn"],
            "board_b": pos_b["board"],
            "hand_b": pos_b["hand"],
            "turn_b": pos_b["turn"],
            "delta_target": torch.tensor(delta, dtype=torch.float32),
        }
        if self.use_features:
            result["features_a"] = pos_a["features"]
            result["features_b"] = pos_b["features"]
        return result
```

`models/__init__.py` の import と `__all__` に `ShogiDeltaPairDataset` と `prepare_pair_position` を追加。

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_ranking.py tests/test_models.py -v`
Expected: 全 PASS（`_prepare_position` 置き換え後も既存 ranking テストが通ること）

- [ ] **Step 5: コミット**

```bash
git add models/dataset.py models/__init__.py tests/test_ranking.py
git commit -m "摂動ペア用ShogiDeltaPairDatasetを追加、ペア前処理を共通化"
```

---

### Task 9: --delta-data の学習統合とドキュメント（Phase 1b）

**Files:**
- Modify: `train/train.py`
- Modify: `CLAUDE.md`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Task 8 の `ShogiDeltaPairDataset`、Task 5 の `compute_delta_loss`、既存 `_ranking_forward` / `ranking_collate_fn` / `select_val_games` / `_make_loader`
- Produces:
  - `TrainConfig.delta_data: str | None = None`、`TrainConfig.delta_same_material_only: bool = False`
  - `TrainState.delta_val: list[dict]`（`{"epoch", "mae"}`）
  - `_build_delta_loaders(config, dataset, device) -> tuple[DataLoader | None, DataLoader | None]`
  - `validate_delta(model, loader, device) -> float`
  - CLI `--delta-data` / `--delta-same-material-only`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_train.py` の `TestTrainRanking` に追加:

```python
    def test_train_with_delta_data(
        self, candidates_data: Path, tmp_path: Path
    ) -> None:
        """摂動ペアデータ付きで学習が完了し、delta検証が記録される."""
        import json

        pairs = [
            {"sfen_a": "startpos moves 7g7f",
             "sfen_b": "startpos moves 2g2f",
             "score_cp_a": -30, "score_cp_b": -50,
             "pair_type": "rewind_branch", "game_id": gid,
             "material_diff": 0}
            for gid in range(3)
        ]
        delta_path = tmp_path / "pairs.jsonl"
        delta_path.write_text("\n".join(json.dumps(p) for p in pairs))

        config = _small_config(
            candidates_data, tmp_path / "ckpt",
            ranking_weight=0.0, delta_weight=0.3,
            delta_data=str(delta_path),
        )
        train(config)

        ckpt = torch.load(
            tmp_path / "ckpt" / "final.pt", map_location="cpu",
            weights_only=False,
        )
        assert ckpt["config"]["delta_data"] == str(delta_path)
        delta_val = ckpt["state"]["delta_val"]
        assert len(delta_val) == config.epochs
        assert delta_val[0]["mae"] >= 0.0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_train.py::TestTrainRanking::test_train_with_delta_data -v`
Expected: FAIL（TypeError: unexpected keyword 'delta_data'）

- [ ] **Step 3: 実装**

`TrainConfig` に追加（`delta_weight` の後）:

```python
    # 摂動ペアデータ（gen_perturb_pairs.py出力、Noneで無効）
    # delta_weight > 0 のとき、candidatesペアに加えて摂動ペアのΔVも学習する
    delta_data: str | None = None
    # material_diff==0（素材一致）のペアのみ使用
    delta_same_material_only: bool = False
```

`TrainState` に追加:

```python
    # 摂動ペア検証メトリクス（epochごと: {"epoch", "mae"}）
    delta_val: list[dict] = field(default_factory=list)
```

`models` の import 行に `ShogiDeltaPairDataset` を追加。

`_build_ranking_loaders` の後に追加:

```python
def _build_delta_loaders(
    config: TrainConfig,
    dataset: ShogiValueDataset,
    device: torch.device,
) -> tuple[DataLoader | None, DataLoader | None]:
    """摂動ペア（--delta-data）のtrain/val用DataLoaderを構築する.

    メインデータセットと同じ対局分割を共有する。gen_perturb_pairs.pyは
    入力JSONLのgame_idをペアへ引き継ぐため、学習データと同じJSONLから
    生成したペアであれば検証対局由来のペアが訓練に混ざらない。

    Args:
        config: 学習設定
        dataset: メインのShogiValueDataset（game_id分割の共有に使用）
        device: 実行デバイス

    Returns:
        (訓練用DataLoader, 検証用DataLoader)。delta_data未指定または
        delta_weight <= 0の場合は(None, None)。

    Raises:
        ValueError: 対局単位の分割を共有できない場合、
            またはペアが1つも構築できない場合。
    """
    if not config.delta_data or config.delta_weight <= 0:
        return None, None

    game_ids = [s.get("game_id", 0) for s in dataset.samples]
    val_games = select_val_games(game_ids, config.val_split)
    if val_games is None:
        raise ValueError(
            "--delta-dataにはgame_idが2対局以上あるデータが必要です"
        )

    dataset_kwargs = dict(
        use_features=config.use_features,
        normalize_turn=config.normalize_turn,
        cp_scale=config.cp_scale,
        target_mode=config.target_mode,
        wdl_scale=config.wdl_scale,
        same_material_only=config.delta_same_material_only,
    )
    delta_train_dataset = ShogiDeltaPairDataset(
        config.delta_data,
        augment_flip=config.augment_flip,
        exclude_game_ids=val_games,
        **dataset_kwargs,
    )
    if len(delta_train_dataset) == 0:
        raise ValueError(
            f"--delta-dataからペアを構築できません: {config.delta_data}"
        )

    pair_batch_size = max(1, config.batch_size // 2)
    delta_train_loader = _make_loader(
        delta_train_dataset, pair_batch_size, True, ranking_collate_fn,
        config, device,
    )

    delta_val_dataset = ShogiDeltaPairDataset(
        config.delta_data,
        augment_flip=False,
        include_game_ids=val_games,
        **dataset_kwargs,
    )
    delta_val_loader: DataLoader | None = None
    if len(delta_val_dataset) > 0:
        delta_val_loader = _make_loader(
            delta_val_dataset, pair_batch_size, False, ranking_collate_fn,
            config, device,
        )
    logger.info(
        f"Delta pairs: train={len(delta_train_dataset)}, "
        f"val={len(delta_val_dataset)}"
    )
    return delta_train_loader, delta_val_loader
```

`validate_ranking` の後に追加:

```python
@torch.no_grad()
def validate_delta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """摂動ペアのΔV平均絶対誤差を計算.

    Args:
        model: 評価するモデル
        loader: 摂動ペアの検証ローダー
        device: デバイス

    Returns:
        ΔV平均絶対誤差
    """
    model.eval()
    total_err = 0.0
    total = 0
    for batch in loader:
        value_a, value_b = _ranking_forward(model, batch, device)
        target = batch["delta_target"].to(device)
        total_err += (value_a - value_b - target).abs().sum().item()
        total += value_a.size(0)
    return total_err / total if total else 0.0
```

`train_epoch` にパラメータ `delta_loader: DataLoader | None = None` を追加し、冒頭に:

```python
    delta_iter = iter(delta_loader) if delta_loader is not None else None
```

ranking 分岐の後（損失合成の前）に追加:

```python
        # 摂動ペアのΔV差分回帰損失
        perturb_loss: torch.Tensor | None = None
        if delta_iter is not None:
            try:
                delta_batch = next(delta_iter)
            except StopIteration:
                delta_iter = iter(delta_loader)
                delta_batch = next(delta_iter)
            value_a, value_b = _ranking_forward(model, delta_batch, device)
            perturb_loss = compute_delta_loss(
                value_a, value_b,
                delta_batch["delta_target"].to(device),
                huber_delta,
            )
```

損失合成に追加:

```python
        if perturb_loss is not None:
            loss = loss + delta_weight * perturb_loss
```

ステップログに追加（`delta_str` の後）:

```python
            perturb_str = (
                f", perturb={perturb_loss.item():.6f}"
                if perturb_loss is not None else ""
            )
```

（ログの f-string 末尾を `{ranking_str}{delta_str}{perturb_str})` に変更）

`train()` に統合:

```python
    # 摂動ペアローダー（--delta-data、対局分割をメインと共有）
    delta_train_loader, delta_val_loader = _build_delta_loaders(
        config, dataset, device
    )
```

`train_epoch(...)` 呼び出しに `delta_loader=delta_train_loader,` を追加。ranking 検証の後に:

```python
        # 摂動ペア検証（ΔV差分の予測誤差）
        if delta_val_loader is not None:
            delta_mae = validate_delta(eval_model, delta_val_loader, device)
            state.delta_val.append({"epoch": epoch + 1, "mae": delta_mae})
            logger.info(f"Delta val: mae={delta_mae:.4f}")
```

`write_log` の `log_data` に追加:

```python
        "delta_val": state.delta_val,
```

CLI に追加:

```python
    parser.add_argument("--delta-data", type=str, default=None,
                        help="摂動ペアJSONL（gen_perturb_pairs.py出力。--delta-weightと併用）")
    parser.add_argument("--delta-same-material-only", action="store_true",
                        help="素材一致（material_diff==0）の摂動ペアのみ使用")
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/ -v`
Expected: 全 PASS（全テストファイル横断で無退行）

- [ ] **Step 5: CLAUDE.md を更新**

「### Pairwise ranking損失」セクションの後に追加:

```markdown
### 局面感度蒸留（ΔV差分回帰）

似た局面ペアの「評価値の差分」を教師差分に合わせる補助損失。
1手読みの手選びはargmaxで決まるため、差分方向の誤差を直接抑える。

```bash
# candidatesペアの差分回帰（データ再生成不要）
python train/train.py --data data_mpv.jsonl --delta-weight 0.3

# 摂動ペアの生成（安全リスト内の摂動 + 安定性フィルタ）
python tools/gen_perturb_pairs.py \
    --data data/raw/dataset_mpv.jsonl \
    -o data/raw/perturb_pairs.jsonl

# 摂動ペアも使った学習
python train/train.py --data data_mpv.jsonl \
    --delta-weight 0.3 --delta-data data/raw/perturb_pairs.jsonl
```

- ペア種別: rewind_branch（本譜vs分岐）/ move_dest（移動先違い）/
  promotion（成/不成、素材一致）
- エンジンで2回評価（`--label-nodes` / `--stability-nodes`）し、
  差分の符号が反転するペアはノイズとして破棄
- `material_diff`が記録され、`--delta-same-material-only`で素材一致
  ペアのみに絞れる（駒価値で説明できる差分を除き、位置の感度に集中）
- 訓練/検証は入力JSONLのgame_idを引き継いでメインと同じ対局分割を共有
```

`train.py` の主なオプション表に追加:

```markdown
| `--delta-weight` | 0 | ΔV差分回帰損失の重み（0=無効、candidatesペアの評価値差分を回帰） |
| `--delta-data` | なし | 摂動ペアJSONL（gen_perturb_pairs.py出力） |
| `--delta-same-material-only` | - | 素材一致の摂動ペアのみ使用 |
```

Phase 5 タスクリストの優先度Bセクションに追加:

```markdown
- [x] **cp regret計測** - 一致率より頑健な手選び指標（censored集計・clamp付き）
  - `move_agreement.py`: `--regret-clamp`オプション、`summarize_regret`
  - `train.py`: 自動計測ログにregretを記録、holdout同一ファイル警告

- [x] **局面感度蒸留（ΔV差分回帰）** - 似た局面間の評価差を直接学習
  - `train.py`: `--delta-weight` / `--delta-data` / `--delta-same-material-only`
  - `tools/gen_perturb_pairs.py`: 安全リスト内の摂動ペア生成+安定性フィルタ
  - `models/dataset.py`: `ShogiDeltaPairDataset`、candidatesペアのdelta_target
```

- [ ] **Step 6: コミット**

```bash
git add train/train.py tests/test_train.py CLAUDE.md
git commit -m "摂動ペアの学習統合（--delta-data）とドキュメント更新"
```

---

## スコープ外（別計画で対応）

- **公式駒落ち初期局面からのデータ生成**（設計書 Phase 1b 安全リスト4項目目）:
  `gen_dataset.py` は「startpos moves ...」形式を対局ループ・並列ワーカー・
  pv_leaf/multipvレコード構築の6箇所で前提にしており、任意の初期SFEN対応は
  それらを横断する独立した機能追加になる。ΔV学習の仕組みとは独立
  （ペアではなく分布拡張）のため、本計画から分離して別途計画する。

## 完了後の計測手順（実装タスク外、参考）

Phase 1 の効果判定は Windows 環境での再学習時に以下の A/B で行う:

```bash
# ベースライン（ΔVなし）
python train/train.py --data <train.jsonl> --agreement-data <holdout.jsonl> \
    --ranking-weight 0.3 --output-dir checkpoints/baseline

# Phase 1a（candidatesペアΔV）
python train/train.py --data <train.jsonl> --agreement-data <holdout.jsonl> \
    --ranking-weight 0.3 --delta-weight 0.3 --output-dir checkpoints/delta1a

# Phase 1b（+摂動ペア）
python tools/gen_perturb_pairs.py --data <train.jsonl> -o <pairs.jsonl>
python train/train.py --data <train.jsonl> --agreement-data <holdout.jsonl> \
    --ranking-weight 0.3 --delta-weight 0.3 --delta-data <pairs.jsonl> \
    --output-dir checkpoints/delta1b
```

### 学習完了後の判定手順

1. **自動計測の確認**: 各構成の `checkpoints/<構成名>/log_*.json` の
   `agreement` 配列に holdout での regret/一致率がエポック推移込みで
   記録されている。val_loss は下がるのに regret が悪化していないかを見る。
2. **best.pt の本計測**（3構成とも同一の holdout・`--limit`・シードで）:

   ```bash
   python scripts/move_agreement.py --model checkpoints/baseline/best.pt \
       --data data/raw/holdout_mpv.jsonl --offline --limit 500 \
       --output reports/baseline_agreement.json
   # delta1a / delta1b も --model と --output だけ変えて同様に
   ```

   見る指標: `regret_mean_cp`（主指標、小さいほど良い）、`agreement`（参考）、
   `multipv_hit_rate`、`regret_censored`（多すぎる場合は regret の信頼性低下）。
3. **自己対局で裏取り**:

   ```bash
   python scripts/selfplay_match.py --model-a checkpoints/delta1a/best.pt \
       --model-b checkpoints/baseline/best.pt --games 20 \
       --output reports/match_1a_vs_base.json
   # delta1b vs baseline も同様に
   ```

   20局では勝率が±10%程度ブレるため、regret を主判定とし自己対局は
   「矛盾していないかの確認」に使う。時間が許せば50局以上。

### 採否判定の基準

- delta1a の regret が baseline より明確に小さく（数cpはノイズの可能性
  あり。目安: 相対5〜10%以上の改善）、自己対局で負け越していない → 採用
- 同様に delta1b の追加効果を判定
- regret は改善したのに自己対局で明確に負ける場合は要調査
  （計測条件の食い違いか、regret が捉えていない弱点のサイン）
- 効果ゼロなら `--delta-weight` を 0.1〜0.5 の範囲で1〜2回調整してから
  見切る。効果確認後は Phase 2（ハードネガティブ）へ進む
