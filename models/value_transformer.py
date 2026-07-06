"""Value Network: 蒸留Transformerによる局面評価モデル.

81マスをトークンとして扱い、局面から評価値を予測する。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


# 駒種の定義（空、先手駒、後手駒、成駒を含む）
# 0: 空マス
# 1-14: 先手の駒（歩、香、桂、銀、金、角、飛、王、と、成香、成桂、成銀、馬、龍）
# 15-28: 後手の駒（同上）
PIECE_TYPES = 29

# 盤面のサイズ
BOARD_SIZE = 81

# 持ち駒の最大数（各駒種ごと）
# 歩:18, 香:4, 桂:4, 銀:4, 金:4, 角:2, 飛:2
MAX_HAND_PIECES = {
    "P": 18,
    "L": 4,
    "N": 4,
    "S": 4,
    "G": 4,
    "B": 2,
    "R": 2,
}

# 持ち駒トークン数（先手7種 + 後手7種 = 14）
HAND_TOKENS = 14

# 拡張特徴量の次元数
# [attack(2), king_dist(2), piece_value(1), control(1), king_safety(4)]
FEATURE_DIM = 10


class PositionalEncoding(nn.Module):
    """固定の正弦波位置エンコーディング."""

    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """位置エンコーディングを加算.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            位置エンコーディング加算後のテンソル
        """
        return x + self.pe[:, : x.size(1)]


class ValueTransformer(nn.Module):
    """将棋局面評価用Transformerモデル.

    81マス + 持ち駒14トークン = 95トークンを入力とし、
    Attention Poolingで評価値（勝率近似）を出力する。

    Args:
        d_model: 埋め込み次元数
        n_heads: アテンションヘッド数
        n_layers: Transformerレイヤー数
        ffn_dim: FFNの中間次元数
        dropout: ドロップアウト率
        use_features: 拡張特徴量を使用するかどうか
        use_attention_pooling: Attention Poolingを使用するかどうか（Falseの場合Mean Pooling）
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_features: bool = False,
        use_attention_pooling: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.use_features = use_features
        self.use_attention_pooling = use_attention_pooling

        # 駒種埋め込み（盤上の駒用）
        self.piece_embedding = nn.Embedding(PIECE_TYPES, d_model)

        # 拡張特徴量の線形変換（use_features=True時のみ使用）
        if use_features:
            self.feature_linear = nn.Linear(FEATURE_DIM, d_model)

        # 持ち駒埋め込み（持ち駒種 × 持ち駒数の特徴量）
        # 各持ち駒種に対して、枚数を連続値として扱う
        self.hand_type_embedding = nn.Embedding(HAND_TOKENS, d_model)
        self.hand_count_linear = nn.Linear(1, d_model)

        # 手番埋め込み（先手:0, 後手:1）
        self.turn_embedding = nn.Embedding(2, d_model)

        # 位置エンコーディング
        # 81(盤面) + 14(持ち駒) = 95
        self.pos_encoding = PositionalEncoding(d_model, max_len=BOARD_SIZE + HAND_TOKENS)

        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention Pooling用のパラメータ（use_attention_pooling=True時のみ使用）
        if use_attention_pooling:
            # 学習可能なクエリベクトル（「何が重要か」を学習）
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            # Attention計算用の線形層
            self.pool_key = nn.Linear(d_model, d_model)

        # 評価値出力ヘッド
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),  # [-1, 1] に正規化
        )

        # 勝敗予測ヘッド（補助タスク）
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # [0, 1] 勝率
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """重みの初期化."""
        # 埋め込み層のstdはd_modelに基づいて設定
        embed_std = 1.0 / math.sqrt(self.d_model)  # d_model=256なら0.0625
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=embed_std)

        # Attention Poolingのクエリを初期化
        if self.use_attention_pooling:
            nn.init.normal_(self.pool_query, mean=0.0, std=embed_std)

        # 拡張特徴量の線形層を控えめに初期化（加算後の爆発を防ぐ）
        if self.use_features:
            nn.init.normal_(self.feature_linear.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.feature_linear.bias)

        # 出力層の重みを初期化（Tanh飽和を避けるため控えめに）
        # output_head: Linear -> GELU -> Dropout -> Linear -> Tanh
        # std=0.2で初期から[-0.2, 0.2]程度の出力、飽和を回避
        nn.init.normal_(self.output_head[3].weight, mean=0.0, std=0.2)
        nn.init.zeros_(self.output_head[3].bias)

        # outcome_head: Linear -> GELU -> Dropout -> Linear -> Sigmoid
        nn.init.normal_(self.outcome_head[3].weight, mean=0.0, std=0.2)
        nn.init.zeros_(self.outcome_head[3].bias)

    def forward(
        self,
        board: torch.Tensor,
        hand: torch.Tensor,
        turn: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """順伝播.

        Args:
            board: 盤面の駒配置 (batch, 81) - 駒種ID
            hand: 持ち駒の枚数 (batch, 14) - 各持ち駒種の枚数
            turn: 手番 (batch,) - 0:先手, 1:後手
            features: 拡張特徴量 (batch, 81, 10) - use_features=True時のみ使用

        Returns:
            tuple[Tensor, Tensor]:
                - value: 評価値 (batch, 1) - [-1, 1]の範囲
                - outcome: 勝率予測 (batch, 1) - [0, 1]の範囲
        """
        batch_size = board.size(0)

        # 盤面トークンの埋め込み
        board_emb = self.piece_embedding(board)  # (batch, 81, d_model)

        # 拡張特徴量を追加
        if self.use_features and features is not None:
            feature_emb = self.feature_linear(features)  # (batch, 81, d_model)
            board_emb = board_emb + feature_emb

        # 持ち駒トークンの埋め込み
        hand_type_idx = torch.arange(HAND_TOKENS, device=board.device)
        hand_type_emb = self.hand_type_embedding(hand_type_idx)  # (14, d_model)
        hand_type_emb = hand_type_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # 持ち駒の枚数を特徴量に変換
        hand_count = hand.unsqueeze(-1).float()  # (batch, 14, 1)
        hand_count_emb = self.hand_count_linear(hand_count)  # (batch, 14, d_model)

        # 持ち駒埋め込み = 駒種埋め込み + 枚数埋め込み
        hand_emb = hand_type_emb + hand_count_emb  # (batch, 14, d_model)

        # 全トークンを結合（盤面 + 持ち駒）
        tokens = torch.cat([board_emb, hand_emb], dim=1)  # (batch, 95, d_model)

        # 位置エンコーディング
        tokens = self.pos_encoding(tokens)

        # 手番埋め込みを全トークンに加算
        turn_emb = self.turn_embedding(turn)  # (batch, d_model)
        tokens = tokens + turn_emb.unsqueeze(1)

        # Transformerエンコーダ
        encoded = self.transformer(tokens)  # (batch, 95, d_model)

        # Pooling
        if self.use_attention_pooling:
            # Attention Pooling（学習された重みで加重平均）
            # Query: 学習可能なベクトル (batch, 1, d_model)
            query = self.pool_query.expand(batch_size, -1, -1)
            # Key: 各トークンを変換 (batch, 95, d_model)
            keys = self.pool_key(encoded)
            # Attention scores: (batch, 1, 95)
            attn_scores = torch.bmm(query, keys.transpose(-2, -1)) / math.sqrt(self.d_model)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            # 加重平均: (batch, 1, d_model) -> (batch, d_model)
            pooled = torch.bmm(attn_weights, encoded).squeeze(1)
        else:
            # Mean Pooling（単純平均）
            pooled = encoded.mean(dim=1)  # (batch, d_model)

        # 評価値出力
        value = self.output_head(pooled)  # (batch, 1)

        # 勝敗予測出力
        outcome = self.outcome_head(pooled)  # (batch, 1)

        return value, outcome


def normalize_cp(cp: int, scale: float = 1200.0) -> float:
    """centipawnを[-1, 1]に正規化.

    Args:
        cp: 評価値（centipawn）
        scale: スケーリングパラメータ

    Returns:
        正規化された評価値
    """
    return math.tanh(cp / scale)


def denormalize_cp(value: float, scale: float = 1200.0) -> float:
    """[-1, 1]をcentipawnに戻す.

    Args:
        value: 正規化された評価値
        scale: スケーリングパラメータ

    Returns:
        評価値（centipawn）
    """
    # tanh^-1 = atanh
    # クリップして数値安定性を確保
    value = max(-0.9999, min(0.9999, value))
    return math.atanh(value) * scale
