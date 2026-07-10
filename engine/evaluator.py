"""局面評価器.

学習済みValue Networkを使用して局面を評価する。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import shogi
import torch

from models import (
    ValueTransformer,
    denormalize_cp,
    parse_sfen,
    stack_features,
    wdl_to_cp,
)
from models.dataset import normalize_to_black_view

if TYPE_CHECKING:
    from models.sfen_parser import ParsedPosition

# 詰みの確定評価値（gen_dataset.pyのmate_to_cpと同じスケール）
MATE_SCORE = 30000


class Evaluator:
    """局面評価器."""

    def __init__(
        self,
        model_path: Path | str,
        device: str = "auto",
    ) -> None:
        """初期化.

        Args:
            model_path: チェックポイントファイルのパス
            device: 推論デバイス（auto/cuda/mps/cpu）
        """
        self.device = self._get_device(device)
        (
            self.model,
            self.use_features,
            self.normalize_turn,
            self.cp_scale,
            self.target_mode,
            self.wdl_scale,
        ) = self._load_model(Path(model_path))

    def _get_device(self, device_str: str) -> torch.device:
        """デバイスを取得."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def _load_model(
        self, model_path: Path
    ) -> tuple[ValueTransformer, bool, bool, float, str, float]:
        """モデルを読み込み."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # チェックポイントからモデル設定を取得
        config = checkpoint.get("config", {})
        use_features = config.get("use_features", False)
        normalize_turn = config.get("normalize_turn", False)
        # 学習時の正規化スケール（denormalizeに同じ値を使わないと評価値が歪む）
        cp_scale = config.get("cp_scale", 500.0)
        # ターゲット空間（cp: tanh正規化, wdl: 勝率空間）
        target_mode = config.get("target_mode", "cp")
        wdl_scale = config.get("wdl_scale", 600.0)

        # Attention Pooling導入前のcheckpointはconfigにキーが無いため、
        # state_dictの実際のキーから判定する
        state_dict = checkpoint["model_state_dict"]
        use_attention_pooling = config.get(
            "use_attention_pooling", "pool_query" in state_dict
        )

        model = ValueTransformer(
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            ffn_dim=config.get("ffn_dim", 512),
            dropout=0.0,  # 推論時はドロップアウト無効
            use_features=use_features,
            use_attention_pooling=use_attention_pooling,
            use_king_relative=config.get("use_king_relative", False),
            use_2d_pos=config.get("use_2d_pos", False),
            use_discrete_hand=config.get("use_discrete_hand", False),
        )

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model, use_features, normalize_turn, cp_scale, target_mode, wdl_scale

    def _to_cp(self, value: float) -> int:
        """モデル出力[-1, 1]をcentipawnに変換.

        学習時のターゲット空間（target_mode）に応じた逆変換を行う。

        Args:
            value: モデル出力

        Returns:
            評価値（centipawn）
        """
        if self.target_mode == "wdl":
            wr = (value + 1.0) / 2.0
            return int(wdl_to_cp(wr, self.wdl_scale))
        return int(denormalize_cp(value, self.cp_scale))

    def _prepare_single(
        self, parsed: ParsedPosition
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """1局面分のモデル入力テンソルを作成（CPU上、バッチ次元なし）.

        normalize_turnで学習したモデルの場合、後手番局面を先手視点に変換する。
        モデル出力は変換後の盤面の手番側（=元の手番側）視点なので、
        呼び出し側での符号反転は不要。

        Args:
            parsed: パース済み局面

        Returns:
            (board, hand, turn, features)のタプル。featuresは
            use_features=Falseの場合None。
        """
        board = parsed.board
        hand = parsed.hand
        turn = parsed.turn

        if self.normalize_turn and turn.item() == 1:
            # ダミーの評価値と勝敗を渡して盤面のみ正規化
            board, hand, turn, _, _ = normalize_to_black_view(
                board, hand, turn, 0.0, 0.5
            )

        features: torch.Tensor | None = None
        if self.use_features:
            features = stack_features(board)

        return board, hand, turn, features

    def _evaluate_batch(self, parsed_list: list[ParsedPosition]) -> torch.Tensor:
        """複数局面をまとめて評価.

        Args:
            parsed_list: パース済み局面のリスト

        Returns:
            各局面の評価値テンソル (N,) - [-1, 1]、手番側視点
        """
        boards: list[torch.Tensor] = []
        hands: list[torch.Tensor] = []
        turns: list[torch.Tensor] = []
        features_list: list[torch.Tensor] = []

        for parsed in parsed_list:
            board, hand, turn, features = self._prepare_single(parsed)
            boards.append(board)
            hands.append(hand)
            turns.append(turn)
            if features is not None:
                features_list.append(features)

        batch_board = torch.stack(boards).to(self.device)
        batch_hand = torch.stack(hands).to(self.device)
        batch_turn = torch.stack(turns).to(self.device)
        batch_features = (
            torch.stack(features_list).to(self.device) if features_list else None
        )

        with torch.no_grad():
            value, _ = self.model(batch_board, batch_hand, batch_turn, batch_features)

        return value.squeeze(1).cpu()

    def evaluate_sfen(self, sfen: str) -> int:
        """SFEN文字列で表された局面を評価.

        Args:
            sfen: SFEN文字列（"startpos"または"startpos moves ..."形式）

        Returns:
            評価値（centipawn、手番側視点）
        """
        parsed = parse_sfen(sfen)
        value = self._evaluate_batch([parsed])[0].item()
        return self._to_cp(value)

    def evaluate_board(self, board: shogi.Board) -> int:
        """python-shogiのBoardオブジェクトを評価.

        Args:
            board: shogi.Boardオブジェクト

        Returns:
            評価値（centipawn、手番側視点）
        """
        sfen = self._board_to_sfen(board)
        return self.evaluate_sfen(sfen)

    def _board_to_sfen(self, board: shogi.Board) -> str:
        """BoardオブジェクトをSFEN文字列に変換."""
        # python-shogiのsfen()は"<board> <turn> <hand> <move_count>"形式を返す
        return "sfen " + board.sfen()

    def find_best_move(self, board: shogi.Board) -> tuple[str, int]:
        """最善手を探索（1手読み、全候補を1バッチで評価）.

        1手詰めが存在する場合はNN評価に依らずルールで確定し、
        (詰ます手, +MATE_SCORE) を即座に返す。

        Args:
            board: 現在の局面

        Returns:
            (最善手のUSI表記, 評価値)
        """
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return "resign", -MATE_SCORE

        # 各合法手を適用した局面をパース（1手詰めがあれば確定評価で即返す）
        parsed_list: list[ParsedPosition] = []
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.usi(), MATE_SCORE
            parsed_list.append(parse_sfen(self._board_to_sfen(board)))
            board.pop()

        # 1バッチでまとめて評価
        # 指した後の局面は相手番なので、符号反転して自分視点に変換
        values = -self._evaluate_batch(parsed_list)

        best_idx = int(torch.argmax(values).item())
        best_score = self._to_cp(values[best_idx].item())

        return legal_moves[best_idx].usi(), best_score
