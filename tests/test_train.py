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
