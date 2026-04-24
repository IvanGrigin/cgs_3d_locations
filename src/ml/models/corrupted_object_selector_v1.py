from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CorruptedObjectSelectorOutput:
    logits: torch.Tensor


class CorruptedObjectSelectorV1(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_categories: int,
        num_supers: int,
        num_mount_types: int,
        num_room_types: int,
        hidden_dim: int = 192,
        emb_dim: int = 24,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.category_emb = nn.Embedding(max(1, int(num_categories) + 1), emb_dim)
        self.super_emb = nn.Embedding(max(1, int(num_supers) + 1), emb_dim)
        self.mount_emb = nn.Embedding(max(1, int(num_mount_types) + 1), emb_dim)
        self.room_emb = nn.Embedding(max(1, int(num_room_types) + 1), emb_dim)

        in_dim = int(feature_dim) + emb_dim * 4
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.logit_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        category: torch.Tensor,
        super_category: torch.Tensor,
        mount_type: torch.Tensor,
        room_type: torch.Tensor,
        mask: torch.Tensor,
    ) -> CorruptedObjectSelectorOutput:
        room_emb = self.room_emb(room_type.clamp(min=0))[:, None, :].expand(-1, features.shape[1], -1)
        x = torch.cat(
            [
                features,
                self.category_emb(category.clamp(min=0)),
                self.super_emb(super_category.clamp(min=0)),
                self.mount_emb(mount_type.clamp(min=0)),
                room_emb,
            ],
            dim=-1,
        )
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=(mask <= 0.5))
        x = self.final_norm(x)
        logits = self.logit_head(x).squeeze(-1)
        logits = logits.masked_fill(mask <= 0.5, -1e9)
        return CorruptedObjectSelectorOutput(logits=logits)
