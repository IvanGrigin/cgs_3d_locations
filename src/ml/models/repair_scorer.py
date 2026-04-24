from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass
class RepairScorerOutput:
    quality_score: torch.Tensor
    best_logit: torch.Tensor


class RepairScorer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_target_categories: int,
        num_target_supers: int,
        num_corruption_types: int,
        num_room_types: int,
        hidden_dim: int = 128,
        emb_dim: int = 16,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.target_cat_emb = nn.Embedding(max(1, int(num_target_categories) + 1), emb_dim)
        self.target_super_emb = nn.Embedding(max(1, int(num_target_supers) + 1), emb_dim)
        self.corruption_emb = nn.Embedding(max(1, int(num_corruption_types) + 1), emb_dim)
        self.room_type_emb = nn.Embedding(max(1, int(num_room_types) + 1), emb_dim)

        in_dim = int(feature_dim) + emb_dim * 4
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim, dropout=dropout) for _ in range(int(num_layers))])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.best_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        target_cat: torch.Tensor,
        target_super: torch.Tensor,
        corruption_type: torch.Tensor,
        room_type: torch.Tensor,
    ) -> RepairScorerOutput:
        x = torch.cat(
            [
                features,
                self.target_cat_emb(target_cat.clamp(min=0)),
                self.target_super_emb(target_super.clamp(min=0)),
                self.corruption_emb(corruption_type.clamp(min=0)),
                self.room_type_emb(room_type.clamp(min=0)),
            ],
            dim=-1,
        )
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return RepairScorerOutput(
            quality_score=self.quality_head(x).squeeze(-1),
            best_logit=self.best_head(x).squeeze(-1),
        )
