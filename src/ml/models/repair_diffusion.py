from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / max(1, half))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


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


class RepairDiffusionNet(nn.Module):
    def __init__(
        self,
        num_categories: int,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_categories = int(max(1, num_categories))
        self.cat_emb = nn.Embedding(self.num_categories, dim)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(2 + 2 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(2 + 2 + 3 + 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_emb = SinusoidalTimeEmbedding(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([CrossBlock(dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context_pos: torch.Tensor,
        context_size: torch.Tensor,
        context_cat: torch.Tensor,
        context_mask: torch.Tensor,
        target_index: torch.Tensor,
        target_cat: torch.Tensor,
        target_size: torch.Tensor,
        corruption_type: torch.Tensor,
        room_h_world: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = context_pos.shape

        is_target = torch.zeros((B, N, 1), dtype=context_pos.dtype, device=context_pos.device)
        is_target.scatter_(1, target_index.view(B, 1, 1), 1.0)

        ctx_feat = torch.cat([context_pos, context_size, is_target], dim=-1)
        ctx = self.ctx_mlp(ctx_feat) + self.cat_emb(context_cat.clamp(min=0))
        ctx = ctx * context_mask.unsqueeze(-1).float()

        target_feat = torch.cat([x_t, target_size, corruption_type, room_h_world], dim=-1)
        q = self.target_mlp(target_feat) + self.cat_emb(target_cat.clamp(min=0))
        q = q + self.time_mlp(self.time_emb(t))
        q = q.unsqueeze(1)

        mask = context_mask > 0.5
        for blk in self.blocks:
            q = blk(q, ctx, mask)

        return self.out(q[:, 0, :])


@dataclass
class DiffusionSchedule:
    T: int
    beta_start: float
    beta_end: float

    def build(self, device: torch.device) -> Dict[str, torch.Tensor]:
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=device, dtype=torch.float32)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        return {
            "betas": betas,
            "alphas": alphas,
            "abar": abar,
            "sqrt_abar": torch.sqrt(abar),
            "sqrt_one_minus_abar": torch.sqrt(1.0 - abar),
        }


def gather_step(coeff: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return coeff.gather(0, t).view(-1, 1)


def q_sample(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return gather_step(schedule["sqrt_abar"], t) * x0 + gather_step(schedule["sqrt_one_minus_abar"], t) * noise


@torch.no_grad()
def predict_x0_from_eps(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_abar = gather_step(schedule["sqrt_abar"], t)
    sqrt_one_minus_abar = gather_step(schedule["sqrt_one_minus_abar"], t)
    return (x_t - sqrt_one_minus_abar * eps_pred) / torch.clamp(sqrt_abar, min=1e-6)
