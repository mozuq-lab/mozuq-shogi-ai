"""学習スクリプト関連のテスト."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train.train import TrainConfig, compute_value_loss, train


@pytest.fixture
def small_data(tmp_path: Path) -> Path:
    """小規模な学習データを生成（3対局 × 8局面）."""
    moves_seq = ["7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e", "6i7h"]
    lines = []
    for game_id in range(3):
        result = "black_win" if game_id % 2 == 0 else "white_win"
        for ply in range(8):
            moves = " ".join(moves_seq[:ply])
            sfen = f"startpos moves {moves}".strip() if ply > 0 else "startpos"
            score = (game_id + 1) * 30 * (1 if ply % 2 == 0 else -1)
            lines.append(
                f'{{"sfen": "{sfen}", "score_cp": {score}, "ply": {ply}, '
                f'"game_id": {game_id}, "result": "{result}"}}'
            )
    path = tmp_path / "train_data.jsonl"
    path.write_text("\n".join(lines))
    return path


def _small_config(data_path: Path, output_dir: Path, **overrides) -> TrainConfig:
    """テスト用の小規模学習設定."""
    defaults = dict(
        data_path=str(data_path),
        epochs=2,
        batch_size=8,
        device="cpu",
        d_model=32,
        n_heads=2,
        n_layers=1,
        ffn_dim=64,
        dropout=0.0,
        output_dir=str(output_dir),
        warmup_epochs=1,
        num_workers=0,
        val_split=0.34,
        log_every=1000,
        save_every=100,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


class TestTrainEma:
    """EMAのテスト."""

    def test_ema_checkpoint_contains_both_weights(
        self, small_data: Path, tmp_path: Path
    ) -> None:
        """EMA有効時、checkpointにEMA重みと生の重みの両方が保存される."""
        config = _small_config(small_data, tmp_path / "ckpt", ema_decay=0.9)
        train(config)

        ckpt = torch.load(
            tmp_path / "ckpt" / "final.pt", map_location="cpu", weights_only=False
        )
        assert "raw_model_state_dict" in ckpt

        # EMA重み（推論用）と生の重み（再開用）は異なる
        ema_w = ckpt["model_state_dict"]["piece_embedding.weight"]
        raw_w = ckpt["raw_model_state_dict"]["piece_embedding.weight"]
        assert not torch.equal(ema_w, raw_w)

    def test_no_ema_checkpoint_has_single_weights(
        self, small_data: Path, tmp_path: Path
    ) -> None:
        """EMA無効時は従来どおりmodel_state_dictのみ."""
        config = _small_config(small_data, tmp_path / "ckpt", ema_decay=0.0)
        train(config)

        ckpt = torch.load(
            tmp_path / "ckpt" / "final.pt", map_location="cpu", weights_only=False
        )
        assert "raw_model_state_dict" not in ckpt


class TestComputeValueLoss:
    """compute_value_lossのテスト."""

    def test_mse_matches_functional(self) -> None:
        value = torch.tensor([[0.5], [-0.3]])
        target = torch.tensor([[0.2], [0.1]])
        loss = compute_value_loss(value, target, "mse")
        expected = torch.nn.functional.mse_loss(value, target)
        assert torch.allclose(loss, expected)

    def test_huber_small_error_quadratic(self) -> None:
        """delta以内の誤差ではHuberはMSE/2と一致する."""
        value = torch.tensor([[0.1]])
        target = torch.tensor([[0.0]])
        loss = compute_value_loss(value, target, "huber", huber_delta=0.5)
        assert torch.allclose(loss, torch.tensor(0.005), atol=1e-6)  # 0.5 * 0.1^2

    def test_huber_large_error_linear(self) -> None:
        """delta超の誤差ではHuberはMSEより小さくなる（外れ値にロバスト）."""
        value = torch.tensor([[2.0]])
        target = torch.tensor([[0.0]])
        huber = compute_value_loss(value, target, "huber", huber_delta=0.5)
        mse = compute_value_loss(value, target, "mse")
        assert huber.item() < mse.item()

    def test_unknown_loss_raises(self) -> None:
        value = torch.zeros(1, 1)
        with pytest.raises(ValueError):
            compute_value_loss(value, value, "unknown")

    def test_train_with_huber(self, small_data: Path, tmp_path: Path) -> None:
        """Huber lossで学習が正常に完了する."""
        config = _small_config(
            small_data, tmp_path / "ckpt", value_loss="huber", huber_delta=0.5
        )
        train(config)
        assert (tmp_path / "ckpt" / "final.pt").exists()


class TestTrainWdl:
    """WDLターゲットでの学習テスト."""

    def test_train_with_wdl(self, small_data: Path, tmp_path: Path) -> None:
        """wdlターゲットで学習が完了し、configがcheckpointに保存される."""
        config = _small_config(
            small_data, tmp_path / "ckpt",
            target_mode="wdl", wdl_scale=600.0, wdl_lambda=0.5,
            cp_clamp=2000.0,
        )
        train(config)

        ckpt = torch.load(
            tmp_path / "ckpt" / "final.pt", map_location="cpu", weights_only=False
        )
        assert ckpt["config"]["target_mode"] == "wdl"
        assert ckpt["config"]["wdl_scale"] == 600.0
        assert ckpt["config"]["cp_clamp"] == 2000.0


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


@pytest.fixture
def candidates_data(tmp_path: Path) -> Path:
    """candidatesフィールド付きの小規模学習データを生成（3対局 × 4局面）."""
    import json

    moves_seq = ["7g7f", "3c3d", "2g2f", "8c8d"]
    # 各手番で合法な候補手（先手番/後手番）
    black_candidates = [
        {"move": "2g2f", "score_cp": 60, "rank": 1},
        {"move": "9g9f", "score_cp": -50, "rank": 2},
    ]
    white_candidates = [
        {"move": "8c8d", "score_cp": 40, "rank": 1},
        {"move": "1c1d", "score_cp": -70, "rank": 2},
    ]

    lines = []
    for game_id in range(3):
        result = "black_win" if game_id % 2 == 0 else "white_win"
        for ply in range(4):
            moves = " ".join(moves_seq[:ply])
            sfen = f"startpos moves {moves}".strip() if ply > 0 else "startpos"
            score = (game_id + 1) * 30 * (1 if ply % 2 == 0 else -1)
            candidates = black_candidates if ply % 2 == 0 else white_candidates
            lines.append(json.dumps({
                "sfen": sfen, "score_cp": score, "ply": ply,
                "game_id": game_id, "result": result,
                "candidates": candidates,
            }))
    path = tmp_path / "candidates_data.jsonl"
    path.write_text("\n".join(lines))
    return path


class TestTrainRanking:
    """ranking損失・一致率計測を有効にした学習テスト."""

    def test_train_with_ranking(
        self, candidates_data: Path, tmp_path: Path
    ) -> None:
        """ranking損失付きで学習が完了し、検証メトリクスが記録される."""
        config = _small_config(
            candidates_data, tmp_path / "ckpt",
            ranking_weight=0.5, ranking_min_gap=30.0,
        )
        train(config)

        ckpt = torch.load(
            tmp_path / "ckpt" / "final.pt", map_location="cpu", weights_only=False
        )
        assert ckpt["config"]["ranking_weight"] == 0.5
        # 検証対局にもcandidatesがあるためranking検証メトリクスが記録される
        ranking_val = ckpt["state"]["ranking_val"]
        assert len(ranking_val) == config.epochs
        assert 0.0 <= ranking_val[0]["accuracy"] <= 1.0

    def test_ranking_without_candidates_raises(
        self, small_data: Path, tmp_path: Path
    ) -> None:
        """candidatesの無いデータでranking有効化は明示エラー."""
        config = _small_config(
            small_data, tmp_path / "ckpt", ranking_weight=0.5
        )
        with pytest.raises(ValueError, match="ranking"):
            train(config)

    def test_train_with_agreement(
        self, candidates_data: Path, tmp_path: Path
    ) -> None:
        """一致率計測付きで学習が完了し、履歴がcheckpoint・ログに残る."""
        config = _small_config(
            candidates_data, tmp_path / "ckpt",
            agreement_data=str(candidates_data),
            agreement_limit=10,
            save_every=2,  # epoch 2で定期保存＋計測
        )
        train(config)

        import json

        log_files = list((tmp_path / "ckpt").glob("log_*.json"))
        assert len(log_files) == 1
        log_data = json.loads(log_files[0].read_text())
        # 定期保存時（epoch 2）・学習終了時・best.ptの3回計測される
        assert len(log_data["agreement"]) == 3
        for entry in log_data["agreement"]:
            assert 0.0 <= entry["agreement"] <= 1.0
            assert 0.0 <= entry["multipv_hit_rate"] <= 1.0

        # checkpointにも自身の計測結果まで含めて残る（計測後に再保存）
        epoch_ckpt = torch.load(
            tmp_path / "ckpt" / "epoch_0002.pt",
            map_location="cpu", weights_only=False,
        )
        assert len(epoch_ckpt["state"]["agreement"]) == 1
        assert epoch_ckpt["state"]["agreement"][0]["epoch"] == 2

        final_ckpt = torch.load(
            tmp_path / "ckpt" / "final.pt",
            map_location="cpu", weights_only=False,
        )
        assert final_ckpt["state"]["agreement"][-1]["final"] is True

        # best.ptには自身の重みで計測した結果が書き戻される
        best_ckpt = torch.load(
            tmp_path / "ckpt" / "best.pt",
            map_location="cpu", weights_only=False,
        )
        best_entries = best_ckpt["state"]["agreement"]
        assert len(best_entries) >= 1
        assert best_entries[-1]["best"] is True

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

    def test_ranking_single_game_raises(self, tmp_path: Path) -> None:
        """game_idが1対局のみのデータでranking有効化は明示エラー（リーク防止）."""
        import json

        candidates = [
            {"move": "2g2f", "score_cp": 60, "rank": 1},
            {"move": "9g9f", "score_cp": -50, "rank": 2},
        ]
        lines = [
            json.dumps({
                "sfen": "startpos", "score_cp": 50, "ply": 0,
                "game_id": 0, "result": "black_win",
                "candidates": candidates,
            })
            for _ in range(8)
        ]
        data_path = tmp_path / "single_game.jsonl"
        data_path.write_text("\n".join(lines))

        config = _small_config(data_path, tmp_path / "ckpt", ranking_weight=0.5)
        with pytest.raises(ValueError, match="game_id"):
            train(config)


class TestComputeOutcomeLoss:
    """重み付き勝敗損失（source別マスク）のテスト."""

    def test_all_ones_matches_plain_bce(self) -> None:
        """全重み1.0なら通常のBCE平均と一致する."""
        from train.train import compute_outcome_loss

        pred = torch.tensor([[0.7], [0.3], [0.9]])
        target = torch.tensor([[1.0], [0.0], [1.0]])
        weight = torch.ones(3, 1)

        expected = torch.nn.functional.binary_cross_entropy(pred, target)
        actual = compute_outcome_loss(pred, target, weight)
        assert actual.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_masked_sample_excluded(self) -> None:
        """重み0のサンプルは損失に寄与しない."""
        from train.train import compute_outcome_loss

        pred = torch.tensor([[0.7], [0.01]])
        target = torch.tensor([[1.0], [1.0]])
        weight = torch.tensor([[1.0], [0.0]])

        # 2番目（大きなBCEを持つ）が無視され、1番目のみのBCEになる
        expected = torch.nn.functional.binary_cross_entropy(
            pred[:1], target[:1]
        )
        actual = compute_outcome_loss(pred, target, weight)
        assert actual.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_all_zero_weights_returns_zero(self) -> None:
        """全重み0（バッチ全体が分岐局面）なら損失0."""
        from train.train import compute_outcome_loss

        pred = torch.tensor([[0.7], [0.3]])
        target = torch.tensor([[1.0], [0.0]])
        weight = torch.zeros(2, 1)

        actual = compute_outcome_loss(pred, target, weight)
        assert actual.item() == pytest.approx(0.0)


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
