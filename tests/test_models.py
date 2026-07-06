"""モデル関連のテスト."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from models import (
    ParsedPosition,
    ShogiValueDataset,
    ValueTransformer,
    collate_fn,
    cp_to_wdl,
    denormalize_cp,
    normalize_cp,
    parse_sfen,
    wdl_to_cp,
)
from models.dataset import normalize_to_black_view


class TestNormalizeCp:
    """評価値正規化のテスト."""

    def test_zero(self) -> None:
        assert normalize_cp(0) == 0.0

    def test_positive(self) -> None:
        result = normalize_cp(1200)
        assert 0.7 < result < 0.8  # tanh(1) ≈ 0.76

    def test_negative(self) -> None:
        result = normalize_cp(-1200)
        assert -0.8 < result < -0.7

    def test_round_trip(self) -> None:
        for cp in [-2000, -500, 0, 500, 2000]:
            normalized = normalize_cp(cp)
            denormalized = denormalize_cp(normalized)
            assert abs(denormalized - cp) < 1.0  # 誤差1cp以内


class TestWdlConversion:
    """cp⇔勝率変換のテスト."""

    def test_zero_cp_is_even(self) -> None:
        assert cp_to_wdl(0) == 0.5

    def test_positive_cp_above_half(self) -> None:
        assert cp_to_wdl(600) > 0.7  # sigmoid(1) ≈ 0.73

    def test_round_trip(self) -> None:
        for cp in [-2000, -500, 0, 500, 2000]:
            wr = cp_to_wdl(cp)
            assert abs(wdl_to_cp(wr) - cp) < 1.0

    def test_wdl_to_cp_clipped(self) -> None:
        # 極端な勝率でも有限値を返す
        assert wdl_to_cp(1.0) < 10000
        assert wdl_to_cp(0.0) > -10000


class TestSfenParser:
    """SFENパーサーのテスト."""

    def test_startpos(self) -> None:
        result = parse_sfen("startpos")
        assert isinstance(result, ParsedPosition)
        assert result.board.shape == (81,)
        assert result.hand.shape == (14,)
        assert result.turn.item() == 0  # 先手

    def test_startpos_initial_board(self) -> None:
        result = parse_sfen("startpos")
        # 1段目（後手側）: 香桂銀金王金銀桂香
        assert result.board[0].item() == 16  # 後手香 (l)
        assert result.board[1].item() == 17  # 後手桂 (n)
        assert result.board[4].item() == 22  # 後手王 (k)
        # 9段目（先手側）: 香桂銀金王金銀桂香
        assert result.board[72].item() == 2  # 先手香 (L)
        assert result.board[76].item() == 8  # 先手王 (K)

    def test_startpos_with_moves(self) -> None:
        result = parse_sfen("startpos moves 7g7f")
        assert result.turn.item() == 1  # 後手番

        # 7七の歩がなくなり、7六に移動
        # idx = (rank - 1) * 9 + (9 - file)
        # 7g: file=7, rank=7 -> (7-1)*9 + (9-7) = 54 + 2 = 56
        # 7f: file=7, rank=6 -> (6-1)*9 + (9-7) = 45 + 2 = 47
        assert result.board[56].item() == 0  # 7gは空
        assert result.board[47].item() == 1  # 7fに先手歩

    def test_startpos_two_moves(self) -> None:
        result = parse_sfen("startpos moves 7g7f 3c3d")
        assert result.turn.item() == 0  # 先手番（2手後）

    def test_empty_hand(self) -> None:
        result = parse_sfen("startpos")
        assert result.hand.sum().item() == 0


class TestNormalizeToBlackView:
    """手番正規化（normalize_to_black_view）のテスト."""

    def test_black_turn_unchanged(self) -> None:
        """先手番の局面はそのまま返る."""
        parsed = parse_sfen("startpos")
        board, hand, turn, value, outcome = normalize_to_black_view(
            parsed.board, parsed.hand, parsed.turn, 0.3, 1.0
        )
        assert torch.equal(board, parsed.board)
        assert torch.equal(hand, parsed.hand)
        assert turn.item() == 0
        assert value == 0.3
        assert outcome == 1.0

    def test_white_turn_label_preserved(self) -> None:
        """後手番の正規化で評価値・勝敗ラベルが不変であること（符号バグの回帰テスト）.

        score_cp・outcomeは手番側視点のラベルなので、先後を入れ替えて
        先手番に正規化しても視点は一致したまま（新先手＝元の手番側）。
        符号やラベルを反転してはいけない。
        """
        parsed = parse_sfen("startpos moves 7g7f")  # 後手番
        assert parsed.turn.item() == 1

        _, _, turn, value, outcome = normalize_to_black_view(
            parsed.board, parsed.hand, parsed.turn, 0.3, 1.0
        )
        assert turn.item() == 0
        assert value == 0.3  # 反転してはいけない
        assert outcome == 1.0  # 反転してはいけない

    def test_white_turn_board_rotated(self) -> None:
        """後手番の盤面が正しく180度回転＋先後入替されること.

        「先手が7六歩を突いて後手番」の局面を正規化すると、
        「相手（後手）が3四歩を突いて先手番」の盤面と一致する。
        """
        parsed = parse_sfen("startpos moves 7g7f")  # 後手番
        board, hand, turn, _, _ = normalize_to_black_view(
            parsed.board, parsed.hand, parsed.turn, 0.0, 0.5
        )

        expected = parse_sfen(
            "sfen lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )
        assert torch.equal(board, expected.board)
        assert torch.equal(hand, expected.hand)
        assert turn.item() == 0

    def test_white_turn_hand_swapped(self) -> None:
        """後手番の正規化で持ち駒の先後が入れ替わること."""
        parsed = parse_sfen("startpos moves 7g7f")
        hand = torch.zeros(14, dtype=torch.long)
        hand[0] = 2  # 先手の歩2枚
        hand[12] = 1  # 後手の角1枚

        _, new_hand, _, _, _ = normalize_to_black_view(
            parsed.board, hand, parsed.turn, 0.0, 0.5
        )
        assert new_hand[7].item() == 2  # 先手の歩 → 後手の歩
        assert new_hand[5].item() == 1  # 後手の角 → 先手の角


class TestValueTransformer:
    """ValueTransformerモデルのテスト."""

    @pytest.fixture
    def model(self) -> ValueTransformer:
        return ValueTransformer(
            d_model=64,
            n_heads=2,
            n_layers=2,
            ffn_dim=128,
            dropout=0.0,
        )

    def test_output_shape(self, model: ValueTransformer) -> None:
        batch_size = 4
        board = torch.zeros(batch_size, 81, dtype=torch.long)
        hand = torch.zeros(batch_size, 14, dtype=torch.long)
        turn = torch.zeros(batch_size, dtype=torch.long)

        value, outcome = model(board, hand, turn)
        assert value.shape == (batch_size, 1)
        assert outcome.shape == (batch_size, 1)

    def test_output_range(self, model: ValueTransformer) -> None:
        batch_size = 4
        board = torch.randint(0, 29, (batch_size, 81))
        hand = torch.randint(0, 5, (batch_size, 14))
        turn = torch.randint(0, 2, (batch_size,))

        value, outcome = model(board, hand, turn)
        # 評価値は [-1, 1]
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)
        # 勝率は [0, 1]
        assert torch.all(outcome >= 0.0)
        assert torch.all(outcome <= 1.0)

    def test_gradient_flow(self, model: ValueTransformer) -> None:
        board = torch.randint(0, 29, (2, 81))
        hand = torch.randint(0, 5, (2, 14))
        turn = torch.randint(0, 2, (2,))

        value, outcome = model(board, hand, turn)
        loss = value.sum() + outcome.sum()
        loss.backward()

        # 全パラメータに勾配が流れていることを確認
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestShogiValueDataset:
    """データセットのテスト."""

    @pytest.fixture
    def sample_data_path(self, tmp_path: Path) -> Path:
        data = [
            '{"sfen": "startpos", "score_cp": 0, "ply": 0, "game_id": 0, "result": "draw"}',
            '{"sfen": "startpos moves 7g7f", "score_cp": 50, "ply": 1, "game_id": 0, "result": "draw"}',
            '{"sfen": "startpos moves 7g7f 3c3d", "score_cp": -30, "ply": 2, "game_id": 0, "result": "draw"}',
        ]
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(data))
        return path

    def test_load(self, sample_data_path: Path) -> None:
        dataset = ShogiValueDataset(sample_data_path)
        assert len(dataset) == 3

    def test_getitem(self, sample_data_path: Path) -> None:
        dataset = ShogiValueDataset(sample_data_path)
        sample = dataset[0]

        assert "board" in sample
        assert "hand" in sample
        assert "turn" in sample
        assert "value" in sample

        assert sample["board"].shape == (81,)
        assert sample["hand"].shape == (14,)
        assert sample["value"].shape == ()

    def test_collate(self, sample_data_path: Path) -> None:
        dataset = ShogiValueDataset(sample_data_path)
        batch = [dataset[i] for i in range(3)]
        collated = collate_fn(batch)

        assert collated["board"].shape == (3, 81)
        assert collated["hand"].shape == (3, 14)
        assert collated["turn"].shape == (3,)
        assert collated["value"].shape == (3,)

    def test_drop_zero_cp(self, sample_data_path: Path) -> None:
        """score_cp==0の局面が除外されること."""
        dataset = ShogiValueDataset(sample_data_path, drop_zero_cp=True)
        assert len(dataset) == 2  # score_cp=0の1件が除外される
        for i in range(len(dataset)):
            assert dataset.samples[i]["score_cp"] != 0

    def test_normalize_turn_value_sign(self, sample_data_path: Path) -> None:
        """normalize_turn有効時も評価値の符号が保存されること（回帰テスト）.

        score_cpは手番側視点なので、後手番局面を先手視点に正規化しても
        value = tanh(score_cp / cp_scale) のまま変わらない。
        """
        cp_scale = 1200.0
        dataset = ShogiValueDataset(
            sample_data_path, cp_scale=cp_scale, normalize_turn=True
        )
        for i in range(len(dataset)):
            sample = dataset[i]
            expected = normalize_cp(dataset.samples[i]["score_cp"], cp_scale)
            assert sample["turn"].item() == 0  # 全て先手番に正規化
            assert sample["value"].item() == pytest.approx(expected, abs=1e-6)

    def test_cp_clamp(self, tmp_path: Path) -> None:
        """cp_clampで評価値が丸められる（除外されない）."""
        data = [
            '{"sfen": "startpos", "score_cp": 5000, "ply": 0, "game_id": 0, "result": "draw"}',
        ]
        path = tmp_path / "clamp.jsonl"
        path.write_text("\n".join(data))

        cp_scale = 1200.0
        dataset = ShogiValueDataset(path, cp_scale=cp_scale, cp_clamp=2000.0)
        assert len(dataset) == 1  # 除外されない
        expected = normalize_cp(2000.0, cp_scale)  # 5000 → 2000に丸め
        assert dataset[0]["value"].item() == pytest.approx(expected, abs=1e-6)

    def test_wdl_target(self, sample_data_path: Path) -> None:
        """wdlターゲットが評価値勝率と勝敗のブレンドになる."""
        wdl_scale = 600.0
        wdl_lambda = 0.5
        dataset = ShogiValueDataset(
            sample_data_path,
            target_mode="wdl",
            wdl_scale=wdl_scale,
            wdl_lambda=wdl_lambda,
        )
        for i in range(len(dataset)):
            raw = dataset.samples[i]
            eval_wr = cp_to_wdl(raw["score_cp"], wdl_scale)
            outcome = 0.5  # sample_data_pathは全てdraw
            expected = 2.0 * (wdl_lambda * eval_wr + (1 - wdl_lambda) * outcome) - 1.0
            assert dataset[i]["value"].item() == pytest.approx(expected, abs=1e-6)

    def test_wdl_lambda_one_ignores_outcome(self, tmp_path: Path) -> None:
        """wdl_lambda=1.0では勝敗を無視し評価値勝率のみを使う."""
        data = [
            '{"sfen": "startpos", "score_cp": 300, "ply": 0, "game_id": 0, "result": "black_win"}',
        ]
        path = tmp_path / "wdl.jsonl"
        path.write_text("\n".join(data))

        dataset = ShogiValueDataset(path, target_mode="wdl", wdl_lambda=1.0)
        expected = 2.0 * cp_to_wdl(300, 600.0) - 1.0
        assert dataset[0]["value"].item() == pytest.approx(expected, abs=1e-6)

    def test_invalid_target_mode_raises(self, sample_data_path: Path) -> None:
        with pytest.raises(ValueError):
            ShogiValueDataset(sample_data_path, target_mode="invalid")


class TestSplitByGame:
    """対局単位のtrain/val分割のテスト."""

    @pytest.fixture
    def multi_game_path(self, tmp_path: Path) -> Path:
        lines = []
        for game_id in range(10):
            for ply in range(5):
                lines.append(
                    f'{{"sfen": "startpos", "score_cp": {game_id * 10 + ply}, '
                    f'"ply": {ply}, "game_id": {game_id}, "result": "draw"}}'
                )
        path = tmp_path / "multi_game.jsonl"
        path.write_text("\n".join(lines))
        return path

    def test_no_game_overlap(self, multi_game_path: Path) -> None:
        """train/valに同じ対局の局面が跨らないこと."""
        from train.train import split_by_game

        dataset = ShogiValueDataset(multi_game_path)
        train_subset, val_subset = split_by_game(dataset, val_split=0.2)

        train_games = {dataset.samples[i]["game_id"] for i in train_subset.indices}
        val_games = {dataset.samples[i]["game_id"] for i in val_subset.indices}

        assert train_games.isdisjoint(val_games)
        assert len(train_subset) + len(val_subset) == len(dataset)
        assert len(val_games) == 2  # 10対局 × 0.2

    def test_augment_flip_same_side(self, multi_game_path: Path) -> None:
        """augment_flip有効時、反転版サンプルも同じ側に割り当てられること."""
        from train.train import split_by_game

        dataset = ShogiValueDataset(multi_game_path, augment_flip=True)
        train_subset, val_subset = split_by_game(dataset, val_split=0.2)

        base_len = len(dataset.samples)
        assert len(train_subset) + len(val_subset) == len(dataset)

        for indices in (train_subset.indices, val_subset.indices):
            index_set = set(indices)
            for idx in indices:
                base_idx = idx - base_len if idx >= base_len else idx
                # 元サンプルと反転版がペアで同じ側にあること
                assert base_idx in index_set
                assert base_idx + base_len in index_set


class TestIntegration:
    """統合テスト."""

    def test_model_with_dataset(self, tmp_path: Path) -> None:
        # データ準備
        data = [
            '{"sfen": "startpos", "score_cp": 0, "ply": 0, "game_id": 0, "result": "draw"}',
            '{"sfen": "startpos moves 7g7f", "score_cp": 50, "ply": 1, "game_id": 0, "result": "draw"}',
        ]
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(data))

        # データセット
        dataset = ShogiValueDataset(path)
        batch = collate_fn([dataset[0], dataset[1]])

        # モデル
        model = ValueTransformer(d_model=64, n_heads=2, n_layers=2, ffn_dim=128)

        # 推論
        value, outcome = model(batch["board"], batch["hand"], batch["turn"])
        assert value.shape == (2, 1)
        assert outcome.shape == (2, 1)

        # 損失計算
        target_value = batch["value"].unsqueeze(1)
        target_outcome = batch["outcome"].unsqueeze(1)
        value_loss = torch.nn.functional.mse_loss(value, target_value)
        outcome_loss = torch.nn.functional.binary_cross_entropy(outcome, target_outcome)
        assert value_loss.item() >= 0
        assert outcome_loss.item() >= 0
