#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/ml/infer/diffusion_placer.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Common helpers
# -----------------------------

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def polygon_bbox(poly_xz: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    xs = [safe_float(p.get("x", 0.0)) for p in poly_xz]
    zs = [safe_float(p.get("z", 0.0)) for p in poly_xz]
    if len(xs) == 0 or len(zs) == 0:
        return (0.0, 1.0, 0.0, 1.0)
    return min(xs), max(xs), min(zs), max(zs)


def iso_params_from_bbox(bb: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    x0, x1, z0, z1 = bb
    dx = max(1e-6, x1 - x0)
    dz = max(1e-6, z1 - z0)
    s = max(dx, dz)
    cx = 0.5 * (x0 + x1)
    cz = 0.5 * (z0 + z1)
    return cx, cz, s


def normalize_xz_iso(x: float, z: float, bb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    cx, cz, s = iso_params_from_bbox(bb)
    xn = (x - (cx - 0.5 * s)) / s
    zn = (z - (cz - 0.5 * s)) / s
    return float(xn), float(zn)


def denormalize_xz_iso(xn: float, zn: float, bb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    cx, cz, s = iso_params_from_bbox(bb)
    x = (cx - 0.5 * s) + xn * s
    z = (cz - 0.5 * s) + zn * s
    return float(x), float(z)


def point_in_poly_mask(poly_norm: np.ndarray, H: int, W: int) -> np.ndarray:
    xs = (np.arange(W) + 0.5) / W
    ys = (np.arange(H) + 0.5) / H
    xx, yy = np.meshgrid(xs, ys)

    poly = poly_norm
    n = poly.shape[0]
    if n < 3:
        return np.zeros((H, W), dtype=np.float32)

    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    y0b = y0.reshape(1, 1, n)
    y1b = y1.reshape(1, 1, n)
    x0b = x0.reshape(1, 1, n)
    x1b = x1.reshape(1, 1, n)

    yb = yy[:, :, None]
    xb = xx[:, :, None]

    cond1 = (y0b > yb) != (y1b > yb)
    denom = (y1b - y0b)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    x_int = x0b + (yb - y0b) * (x1b - x0b) / denom
    cond2 = xb < x_int
    inside = np.sum(cond1 & cond2, axis=-1) % 2 == 1
    return inside.astype(np.float32)


def approx_sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    # Лёгкая аппроксимация SDF без scipy:
    H, W = mask.shape
    inside = mask > 0.5
    dist_out = np.where(~inside, 0.0, 1e6).astype(np.float32)
    dist_in  = np.where(inside, 0.0, 1e6).astype(np.float32)

    # простая волновая релаксация (достаточно для conditioning)
    for _ in range(96):
        dist_out = np.minimum(dist_out, np.minimum(np.roll(dist_out, 1, 0) + 1.0, np.roll(dist_out, 1, 1) + 1.0))
        dist_in  = np.minimum(dist_in,  np.minimum(np.roll(dist_in,  1, 0) + 1.0, np.roll(dist_in,  1, 1) + 1.0))

    sdf = dist_out - dist_in  # >0 outside, <0 inside
    norm = math.sqrt(H * H + W * W) + 1e-6
    return (sdf / norm).astype(np.float32)


def build_room_tensors_iso(room_spec: Dict[str, Any], grid_size: int) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    poly = room_spec.get("floor_polygon_xz", [])
    if not isinstance(poly, list):
        poly = []
    bb = polygon_bbox(poly)

    poly_norm = []
    for p in poly:
        x = safe_float(p.get("x", 0.0))
        z = safe_float(p.get("z", 0.0))
        xn, zn = normalize_xz_iso(x, z, bb)
        poly_norm.append([xn, zn])
    poly_norm = np.asarray(poly_norm, dtype=np.float32)

    H = W = int(grid_size)
    mask = point_in_poly_mask(poly_norm, H, W)
    sdf = approx_sdf_from_mask(mask)
    return mask, sdf, bb


def dims_norm_iso_from_size_mm(size_mm: List[int], bb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    # size_mm=[sx,sy,sz] мм -> w,d в метрах -> нормируем на iso-scale
    sx = max(0.0, float(size_mm[0]) / 1000.0)
    sz = max(0.0, float(size_mm[2]) / 1000.0)
    _, _, s = iso_params_from_bbox(bb)
    s = max(1e-6, s)
    return float(sx / s), float(sz / s)


# -----------------------------
# Model blocks (должны совпадать с обучением)
# -----------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

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


class RoomEncoder(nn.Module):
    def __init__(self, in_ch: int = 2, dim: int = 256, tokens_hw: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.to_tokens = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d((tokens_hw, tokens_hw))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, room_mask: torch.Tensor, room_sdf: torch.Tensor):
        x = torch.cat([room_mask, room_sdf], dim=1)
        h = self.conv(x)
        tok = self.pool(self.to_tokens(h))
        B, D, th, tw = tok.shape
        tokens = tok.view(B, D, th * tw).transpose(1, 2).contiguous()
        g = self.global_pool(h).view(B, D)
        return tokens, g


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, ctx: torch.Tensor):
        q = self.ln1(x)
        x2, _ = self.self_attn(q, q, q, key_padding_mask=~x_mask)
        x = x + x2
        q = self.ln2(x)
        x2, _ = self.cross_attn(q, ctx, ctx, key_padding_mask=None)
        x = x + x2
        x = x + self.mlp(self.ln3(x))
        return x * x_mask.unsqueeze(-1).float()


class FurnitureDenoiser(nn.Module):
    def __init__(self, num_categories: int, dim: int = 256, num_layers: int = 6, num_heads: int = 8,
                 room_tokens_hw: int = 8, dropout: float = 0.0):
        super().__init__()
        self.room_enc = RoomEncoder(in_ch=2, dim=dim, tokens_hw=room_tokens_hw)
        self.cat_emb = nn.Embedding(max(1, num_categories), dim)
        self.size_mlp = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.y_mlp = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_emb = SinusoidalTimeEmbedding(dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([CrossAttentionBlock(dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 4))

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, room_mask: torch.Tensor, room_sdf: torch.Tensor,
                obj_cat: torch.Tensor, obj_wd: torch.Tensor, attn_mask: torch.Tensor):
        room_tokens, room_g = self.room_enc(room_mask, room_sdf)
        e_cat = self.cat_emb(obj_cat.clamp(min=0))
        e_size = self.size_mlp(obj_wd)
        e_y = self.y_mlp(y_t)
        te = self.time_mlp(self.time_emb(t)).unsqueeze(1)
        x = (e_cat + e_size + e_y + te + room_g.unsqueeze(1)) * attn_mask.unsqueeze(-1).float()
        for blk in self.blocks:
            x = blk(x, attn_mask, room_tokens)
        eps = self.out(x)
        return eps * attn_mask.unsqueeze(-1).float()


class DDPM:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str = "cpu"):
        self.T = int(T)
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, self.T, device=self.device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.abar = abar

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, room_mask: torch.Tensor, room_sdf: torch.Tensor,
                    obj_cat: torch.Tensor, obj_wd: torch.Tensor, attn_mask: torch.Tensor,
                    steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        B, N, D = obj_wd.shape[0], obj_wd.shape[1], 4
        device = room_mask.device

        x = torch.randn((B, N, D), device=device) * attn_mask.unsqueeze(-1).float()

        idx = torch.linspace(self.T - 1, 0, int(steps), device=device).long()
        for k in range(int(steps)):
            t = idx[k].repeat(B)
            eps = model(x, t, room_mask, room_sdf, obj_cat, obj_wd, attn_mask)

            a_bar = self.abar[t].view(-1, 1, 1)
            sqrt_a_bar = torch.sqrt(a_bar)
            sqrt_1m = torch.sqrt(1.0 - a_bar)

            x0 = (x - sqrt_1m * eps) / torch.clamp(sqrt_a_bar, min=1e-6)

            if k == int(steps) - 1:
                x = x0
                break

            t_prev = idx[k + 1].repeat(B)
            a_bar_prev = self.abar[t_prev].view(-1, 1, 1)

            sigma = eta * torch.sqrt((1 - a_bar_prev) / (1 - a_bar) * (1 - a_bar / a_bar_prev))
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = torch.sqrt(a_bar_prev) * x0 \
                + torch.sqrt(torch.clamp(1 - a_bar_prev - sigma * sigma, min=0.0)) * eps \
                + sigma * noise

            x = x * attn_mask.unsqueeze(-1).float()

        return x


# -----------------------------
# Placer
# -----------------------------

@dataclass
class DiffusionBundle:
    cfg: Dict[str, Any]
    cat_vocab: Dict[str, int]
    num_categories: int
    state: Dict[str, Any]


class DiffusionPlacer:
    def __init__(self, bundle: DiffusionBundle, device: str):
        self.device = torch.device(device)

        cfg = bundle.cfg
        self.grid_size = int(cfg.get("grid_size", 64))
        self.max_objects = int(cfg.get("max_objects", 64))

        dim = int(cfg.get("dim", 256))
        num_layers = int(cfg.get("num_layers", 6))
        num_heads = int(cfg.get("num_heads", 8))
        room_tokens_hw = int(cfg.get("room_tokens_hw", 8))
        dropout = float(cfg.get("dropout", 0.0))

        self.cat_vocab = dict(bundle.cat_vocab)
        self.unk_id = int(self.cat_vocab.get("UNK", 0))

        self.model = FurnitureDenoiser(
            num_categories=int(bundle.num_categories),
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            room_tokens_hw=room_tokens_hw,
            dropout=dropout,
        ).to(self.device)
        self.model.load_state_dict(bundle.state, strict=True)
        self.model.eval()

        self.diff = DDPM(
            T=int(cfg.get("T", 1000)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 2e-2)),
            device=device,
        )

    @staticmethod
    def load(model_path: str, device: str) -> "DiffusionPlacer":
        obj = torch.load(model_path, map_location=device)

        # Поддерживаем 2 наиболее вероятных формата:
        # (A) export = { model_state, cfg, cat_vocab, num_categories }
        # (B) ckpt   = { model: state_dict, cfg:..., num_categories:... } (если вы сохранили иначе)
        if isinstance(obj, dict) and "model_state" in obj and "cfg" in obj:
            bundle = DiffusionBundle(
                cfg=dict(obj["cfg"]),
                cat_vocab=dict(obj.get("cat_vocab", {})),
                num_categories=int(obj.get("num_categories", len(obj.get("cat_vocab", {})) or 1)),
                state=dict(obj["model_state"]),
            )
            return DiffusionPlacer(bundle=bundle, device=device)

        if isinstance(obj, dict) and "model" in obj and "cfg" in obj:
            bundle = DiffusionBundle(
                cfg=dict(obj["cfg"]),
                cat_vocab=dict(obj.get("cat_vocab", {"UNK": 0})),
                num_categories=int(obj.get("num_categories", 1)),
                state=dict(obj["model"]),
            )
            return DiffusionPlacer(bundle=bundle, device=device)

        raise RuntimeError(f"Unsupported diffusion checkpoint format: {model_path}")

    def _cat_id(self, name: str) -> int:
        if name in self.cat_vocab:
            return int(self.cat_vocab[name])
        return int(self.unk_id)

    @torch.no_grad()
    def predict(self, room_spec: Dict[str, Any], placed_items: List[Dict[str, Any]], steps: int = 50) -> List[Dict[str, Any]]:
        poly = room_spec.get("floor_polygon_xz", [])
        if not isinstance(poly, list) or len(poly) < 3:
            # fallback: всё в 0
            return [{"x": 0.0, "z": 0.0, "yaw_deg": 0.0} for _ in placed_items]

        mask_np, sdf_np, bb = build_room_tensors_iso(room_spec, grid_size=self.grid_size)
        _, _, s_iso = iso_params_from_bbox(bb)

        N = min(len(placed_items), int(self.max_objects))
        # если объектов больше max_objects, оставшиеся — fallback
        tail = len(placed_items) - N

        room_mask = torch.from_numpy(mask_np[None, None, ...]).float().to(self.device)
        room_sdf  = torch.from_numpy(sdf_np[None, None, ...]).float().to(self.device)

        obj_cat = torch.zeros((1, self.max_objects), dtype=torch.long, device=self.device)
        obj_wd  = torch.zeros((1, self.max_objects, 2), dtype=torch.float32, device=self.device)
        attn    = torch.zeros((1, self.max_objects), dtype=torch.bool, device=self.device)

        for i in range(N):
            it = placed_items[i]
            name = str(it.get("name", "UNK"))
            cid = self._cat_id(name)
            obj_cat[0, i] = int(cid)

            size_mm = it.get("size_mm", [600, 400, 600])
            w_norm, d_norm = dims_norm_iso_from_size_mm(size_mm, bb)
            obj_wd[0, i, 0] = float(w_norm)
            obj_wd[0, i, 1] = float(d_norm)

            # допускаем даже маленькие, но не нулевые
            if w_norm > 1e-6 and d_norm > 1e-6:
                attn[0, i] = True

        y0 = self.diff.ddim_sample(
            model=self.model,
            room_mask=room_mask,
            room_sdf=room_sdf,
            obj_cat=obj_cat,
            obj_wd=obj_wd,
            attn_mask=attn,
            steps=int(steps),
            eta=0.0,
        )

        y0 = y0[0, :N].detach().cpu().numpy()
        xz = np.clip(y0[:, 0:2], 0.0, 1.0)
        s = y0[:, 2]
        c = y0[:, 3]

        out: List[Dict[str, Any]] = []
        for i in range(N):
            xn, zn = float(xz[i, 0]), float(xz[i, 1])
            xw, zw = denormalize_xz_iso(xn, zn, bb)
            yaw_deg = float(math.degrees(math.atan2(float(s[i]), float(c[i]))))
            out.append({"x": xw, "z": zw, "yaw_deg": yaw_deg})

        # хвост, если объектов больше max_objects
        for _ in range(max(0, tail)):
            out.append({"x": 0.0, "z": 0.0, "yaw_deg": 0.0})

        return out
