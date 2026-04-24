from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q1 = nn.LayerNorm(dim)
        self.ln_q2 = nn.LayerNorm(dim)
        self.ln_q3 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        x = self.ln_q1(q)
        x2, _ = self.self_attn(x, x, x)
        q = q + x2

        x = self.ln_q2(q)
        x2, _ = self.cross_attn(x, ctx, ctx, key_padding_mask=~ctx_mask)
        q = q + x2

        q = q + self.ff(self.ln_q3(q))
        return q


@dataclass
class RepairProposalOutput:
    clean_pose: torch.Tensor


class RepairProposalNetV1(nn.Module):
    def __init__(
        self,
        num_categories: int,
        num_corruption_types: int,
        num_room_types: int,
        dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.cat_emb = nn.Embedding(max(1, int(num_categories) + 1), dim)
        self.corruption_emb = nn.Embedding(max(1, int(num_corruption_types) + 1), dim)
        self.room_type_emb = nn.Embedding(max(1, int(num_room_types) + 1), dim)

        self.ctx_mlp = nn.Sequential(
            nn.Linear(3 + 3 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(5 + 3 + 7 + 3, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([CrossBlock(dim, num_heads, dropout=dropout) for _ in range(int(num_layers))])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 5),
        )

    def forward(
        self,
        corrupted_pose: torch.Tensor,
        context_pos: torch.Tensor,
        context_size: torch.Tensor,
        context_cat: torch.Tensor,
        context_mask: torch.Tensor,
        target_index: torch.Tensor,
        target_cat: torch.Tensor,
        target_size: torch.Tensor,
        corruption_type: torch.Tensor,
        room_type: torch.Tensor,
        room_scale: torch.Tensor,
        corrupted_flags: torch.Tensor,
    ) -> RepairProposalOutput:
        bsz, nctx, _ = context_pos.shape

        is_target = torch.zeros((bsz, nctx, 1), dtype=context_pos.dtype, device=context_pos.device)
        is_target.scatter_(1, target_index.view(bsz, 1, 1), 1.0)
        ctx_feat = torch.cat([context_pos, context_size, is_target], dim=-1)
        ctx = self.ctx_mlp(ctx_feat) + self.cat_emb(context_cat.clamp(min=0))
        ctx = ctx * context_mask.unsqueeze(-1).float()

        q_feat = torch.cat([corrupted_pose, target_size, corrupted_flags, room_scale], dim=-1)
        q = self.query_mlp(q_feat)
        q = (
            q
            + self.cat_emb(target_cat.clamp(min=0))
            + self.corruption_emb(corruption_type.clamp(min=0))
            + self.room_type_emb(room_type.clamp(min=0))
        )
        q = q.unsqueeze(1)

        mask = context_mask > 0.5
        for block in self.blocks:
            q = block(q, ctx, mask)

        raw = self.head(q[:, 0, :])
        pred_xy = torch.tanh(raw[:, 0:2])
        pred_z = raw[:, 2:3]
        pred_yaw = F.normalize(raw[:, 3:5], dim=-1, eps=1e-6)
        clean_pose = torch.cat([pred_xy, pred_z, pred_yaw], dim=-1)
        return RepairProposalOutput(clean_pose=clean_pose)
