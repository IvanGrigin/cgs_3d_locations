import os
import math
import json
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


@dataclass
class Config:
    npz_path: str
    out_dir: str = "./checkpoints_layout"

    epochs: int = 120
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    train_split: float = 0.9
    seed: int = 42
    num_workers: int = 0

    pos_noise_std: float = 0.08
    siz_noise_std: float = 0.05
    ang_noise_std_deg: float = 15.0

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1

    denoise_steps: int = 3

    w_pos: float = 1.0
    w_siz: float = 1.0
    w_ang: float = 0.5
    w_collision: float = 0.2
    w_boundary: float = 0.2

    save_every: int = 10
    val_save_count: int = 64
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS недоступен")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA недоступна")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_vec2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))
    return x / norm


def masked_mean(loss: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    loss = loss * valid_mask.unsqueeze(-1)
    denom = valid_mask.sum().clamp_min(1) * loss.shape[-1]
    return loss.sum() / denom


def smooth_l1_masked(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return masked_mean(loss, valid_mask)


def pairwise_intersection_area(
    centers: torch.Tensor,  # [B, N, 2]
    sizes: torch.Tensor,    # [B, N, 2]
) -> torch.Tensor:
    """
    Упрощённый collision penalty по осевым bbox.
    Повороты здесь не учитываются: это аппроксимация.
    """
    half = sizes / 2.0
    min_xy = centers - half
    max_xy = centers + half

    min_xy_i = min_xy.unsqueeze(2)  # [B, N, 1, 2]
    max_xy_i = max_xy.unsqueeze(2)
    min_xy_j = min_xy.unsqueeze(1)  # [B, 1, N, 2]
    max_xy_j = max_xy.unsqueeze(1)

    inter_min = torch.maximum(min_xy_i, min_xy_j)
    inter_max = torch.minimum(max_xy_i, max_xy_j)
    inter_wh = torch.clamp(inter_max - inter_min, min=0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # [B, N, N]
    return inter_area


def collision_loss(
    pred_pos_denorm: torch.Tensor,
    pred_siz_denorm: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    inter_area = pairwise_intersection_area(pred_pos_denorm, pred_siz_denorm)  # [B, N, N]

    pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    diag_mask = ~torch.eye(inter_area.shape[1], device=inter_area.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & diag_mask

    inter_area = inter_area * pair_mask
    denom = pair_mask.sum().clamp_min(1)
    return inter_area.sum() / denom


def boundary_box_from_fpoc(fpoc: torch.Tensor, nfpc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    fpoc: [B, P, 2]
    nfpc: [B]
    Возвращает min_xy, max_xy формы [B, 2].
    """
    B, P, _ = fpoc.shape
    device = fpoc.device

    idx = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
    valid = idx < nfpc.unsqueeze(1)

    huge = torch.full_like(fpoc, 1e9)
    neg_huge = torch.full_like(fpoc, -1e9)

    min_src = torch.where(valid.unsqueeze(-1), fpoc, huge)
    max_src = torch.where(valid.unsqueeze(-1), fpoc, neg_huge)

    min_xy = min_src.min(dim=1).values
    max_xy = max_src.max(dim=1).values
    return min_xy, max_xy


def boundary_loss(
    pred_pos_denorm: torch.Tensor,
    pred_siz_denorm: torch.Tensor,
    valid_mask: torch.Tensor,
    fpoc: torch.Tensor,
    nfpc: torch.Tensor,
) -> torch.Tensor:
    """
    Штраф за выход объекта за bbox комнаты,
    построенный по floorplan points.
    """
    room_min, room_max = boundary_box_from_fpoc(fpoc, nfpc)  # [B, 2], [B, 2]
    half = pred_siz_denorm / 2.0
    obj_min = pred_pos_denorm - half
    obj_max = pred_pos_denorm + half

    left_over = torch.clamp(room_min.unsqueeze(1) - obj_min, min=0.0)
    right_over = torch.clamp(obj_max - room_max.unsqueeze(1), min=0.0)

    over = left_over + right_over  # [B, N, 2]
    over = over * valid_mask.unsqueeze(-1)
    denom = valid_mask.sum().clamp_min(1) * 2
    return over.sum() / denom


class FrontLayoutDataset(Dataset):
    """
    npz:
      nbj  : [B]
      pos  : [B, N, 2]
      ang  : [B, N, 2]
      siz  : [B, N, 2]
      cla  : [B, N, C]
      fpoc : [B, P, 2]
      nfpc : [B]
    """

    def __init__(
        self,
        npz_path: str,
        pos_noise_std: float = 0.08,
        siz_noise_std: float = 0.05,
        ang_noise_std_deg: float = 15.0,
    ) -> None:
        super().__init__()
        self.npz_path = npz_path
        self.pos_noise_std = pos_noise_std
        self.siz_noise_std = siz_noise_std
        self.ang_noise_std = math.radians(ang_noise_std_deg)

        x = np.load(npz_path, allow_pickle=True)

        required = ["nbj", "pos", "ang", "siz", "cla", "fpoc", "nfpc"]
        for k in required:
            if k not in x.files:
                raise ValueError(f"В npz отсутствует обязательный ключ: {k}")

        self.nbj = x["nbj"].astype(np.int64)
        self.pos = x["pos"].astype(np.float32)
        self.ang = x["ang"].astype(np.float32)
        self.siz = x["siz"].astype(np.float32)
        self.cla = x["cla"].astype(np.float32)
        self.fpoc = x["fpoc"].astype(np.float32)
        self.nfpc = x["nfpc"].astype(np.int64)

        self.scenedirs = x["scenedirs"] if "scenedirs" in x.files else np.array(
            [f"scene_{i}" for i in range(len(self.nbj))]
        )

        if self.pos.ndim != 3 or self.pos.shape[-1] != 2:
            raise ValueError(f"Некорректная форма pos: {self.pos.shape}")
        if self.ang.ndim != 3 or self.ang.shape[-1] != 2:
            raise ValueError(f"Некорректная форма ang: {self.ang.shape}")
        if self.siz.ndim != 3 or self.siz.shape[-1] != 2:
            raise ValueError(f"Некорректная форма siz: {self.siz.shape}")
        if self.cla.ndim != 3:
            raise ValueError(f"Некорректная форма cla: {self.cla.shape}")

        self.num_scenes = self.pos.shape[0]
        self.max_objects = self.pos.shape[1]
        self.class_dim = self.cla.shape[2]
        self.floorplan_points = self.fpoc.shape[1]

        self.pos_mean = None
        self.pos_std = None
        self.siz_mean = None
        self.siz_std = None

    def set_normalization_stats(
        self,
        pos_mean: np.ndarray,
        pos_std: np.ndarray,
        siz_mean: np.ndarray,
        siz_std: np.ndarray,
    ) -> None:
        self.pos_mean = pos_mean.astype(np.float32)
        self.pos_std = sizelike_clip_std(pos_std.astype(np.float32))
        self.siz_mean = siz_mean.astype(np.float32)
        self.siz_std = sizelike_clip_std(siz_std.astype(np.float32))

    def __len__(self) -> int:
        return self.num_scenes

    def _make_valid_mask(self, nbj: int) -> np.ndarray:
        mask = np.zeros(self.max_objects, dtype=bool)
        mask[:nbj] = True
        return mask

    def _rotate_vec(self, vec: np.ndarray, delta: float) -> np.ndarray:
        c = math.cos(delta)
        s = math.sin(delta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        out = vec @ rot.T
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.clip(norm, 1e-8, None)
        return out

    def _make_noisy(
        self,
        pos: np.ndarray,
        ang: np.ndarray,
        siz: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        noisy_pos = pos.copy()
        noisy_ang = ang.copy()
        noisy_siz = siz.copy()

        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return noisy_pos, noisy_ang, noisy_siz

        noisy_pos[valid_idx] += np.random.randn(len(valid_idx), 2).astype(np.float32) * self.pos_noise_std

        scale = 1.0 + np.random.randn(len(valid_idx), 2).astype(np.float32) * self.siz_noise_std
        noisy_siz[valid_idx] *= scale
        noisy_siz[valid_idx] = np.clip(noisy_siz[valid_idx], 1e-4, None)

        for i in valid_idx:
            delta = np.random.randn() * self.ang_noise_std
            noisy_ang[i] = self._rotate_vec(noisy_ang[i][None, :], delta)[0]

        return noisy_pos, noisy_ang, noisy_siz

    def _norm_pos(self, x: np.ndarray) -> np.ndarray:
        if self.pos_mean is None or self.pos_std is None:
            return x
        return (x - self.pos_mean) / self.pos_std

    def _norm_siz(self, x: np.ndarray) -> np.ndarray:
        if self.siz_mean is None or self.siz_std is None:
            return x
        return (x - self.siz_mean) / self.siz_std

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        nbj = int(self.nbj[idx])
        valid_mask = self._make_valid_mask(nbj)

        clean_pos = self.pos[idx]
        clean_ang = self.ang[idx]
        clean_siz = self.siz[idx]
        cla = self.cla[idx]
        fpoc = self.fpoc[idx]
        nfpc = int(self.nfpc[idx])
        scenedir = str(self.scenedirs[idx])

        noisy_pos, noisy_ang, noisy_siz = self._make_noisy(clean_pos, clean_ang, clean_siz, valid_mask)

        clean_pos_n = self._norm_pos(clean_pos)
        noisy_pos_n = self._norm_pos(noisy_pos)

        clean_siz_n = self._norm_siz(clean_siz)
        noisy_siz_n = self._norm_siz(noisy_siz)

        return {
            "clean_pos": torch.from_numpy(clean_pos_n),
            "clean_ang": torch.from_numpy(clean_ang),
            "clean_siz": torch.from_numpy(clean_siz_n),
            "clean_pos_denorm": torch.from_numpy(clean_pos),
            "clean_siz_denorm": torch.from_numpy(clean_siz),

            "noisy_pos": torch.from_numpy(noisy_pos_n),
            "noisy_ang": torch.from_numpy(noisy_ang),
            "noisy_siz": torch.from_numpy(noisy_siz_n),
            "noisy_pos_denorm": torch.from_numpy(noisy_pos),
            "noisy_siz_denorm": torch.from_numpy(noisy_siz),

            "cla": torch.from_numpy(cla),
            "fpoc": torch.from_numpy(fpoc),
            "nfpc": torch.tensor(nfpc, dtype=torch.long),
            "valid_mask": torch.from_numpy(valid_mask),
            "scenedir": scenedir,
        }


def sizelike_clip_std(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.maximum(x, eps)


def compute_train_stats(dataset: FrontLayoutDataset, indices: list[int]) -> dict[str, np.ndarray]:
    pos_list = []
    siz_list = []

    for idx in indices:
        nbj = int(dataset.nbj[idx])
        pos_list.append(dataset.pos[idx, :nbj])
        siz_list.append(dataset.siz[idx, :nbj])

    pos_cat = np.concatenate(pos_list, axis=0)
    siz_cat = np.concatenate(siz_list, axis=0)

    pos_mean = pos_cat.mean(axis=0)
    pos_std = pos_cat.std(axis=0)
    siz_mean = siz_cat.mean(axis=0)
    siz_std = siz_cat.std(axis=0)

    return {
        "pos_mean": pos_mean.astype(np.float32),
        "pos_std": sizelike_clip_std(pos_std.astype(np.float32)),
        "siz_mean": siz_mean.astype(np.float32),
        "siz_std": sizelike_clip_std(siz_std.astype(np.float32)),
    }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "scenedir":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch], dim=0)
    return out


class FloorplanEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, fpoc: torch.Tensor, nfpc: torch.Tensor) -> torch.Tensor:
        """
        fpoc: [B, P, 2]
        nfpc: [B]
        return: [B, d_model]
        """
        B, P, _ = fpoc.shape
        device = fpoc.device
        idx = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        valid = idx < nfpc.unsqueeze(1)

        h = self.net(fpoc)  # [B, P, d_model]
        h = h * valid.unsqueeze(-1)

        denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = h.sum(dim=1) / denom
        return pooled


class LayoutRefiner(nn.Module):
    def __init__(
        self,
        class_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.class_dim = class_dim
        in_dim = 2 + 2 + 2 + class_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.floorplan_encoder = FloorplanEncoder(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),
        )

    def forward_once(
        self,
        pos: torch.Tensor,
        ang: torch.Tensor,
        siz: torch.Tensor,
        cla: torch.Tensor,
        valid_mask: torch.Tensor,
        fpoc: torch.Tensor,
        nfpc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([pos, ang, siz, cla], dim=-1)
        h = self.input_proj(x)

        fp_ctx = self.floorplan_encoder(fpoc, nfpc)   # [B, d_model]
        h = h + fp_ctx.unsqueeze(1)

        padding_mask = ~valid_mask
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        out = self.head(h)
        delta_pos = out[..., 0:2]
        delta_ang = out[..., 2:4]
        delta_siz = out[..., 4:6]

        pred_pos = pos + delta_pos
        pred_ang = normalize_vec2(ang + delta_ang)
        pred_siz = torch.clamp(siz + delta_siz, min=-10.0)  # в нормализованном пространстве

        return pred_pos, pred_ang, pred_siz

    def forward(
        self,
        noisy_pos: torch.Tensor,
        noisy_ang: torch.Tensor,
        noisy_siz: torch.Tensor,
        cla: torch.Tensor,
        valid_mask: torch.Tensor,
        fpoc: torch.Tensor,
        nfpc: torch.Tensor,
        denoise_steps: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos, ang, siz = noisy_pos, noisy_ang, noisy_siz
        for _ in range(denoise_steps):
            pos, ang, siz = self.forward_once(pos, ang, siz, cla, valid_mask, fpoc, nfpc)
        return pos, ang, siz


def denorm_pos(x: torch.Tensor, pos_mean: torch.Tensor, pos_std: torch.Tensor) -> torch.Tensor:
    return x * pos_std.view(1, 1, 2) + pos_mean.view(1, 1, 2)


def denorm_siz(x: torch.Tensor, siz_mean: torch.Tensor, siz_std: torch.Tensor) -> torch.Tensor:
    return x * siz_std.view(1, 1, 2) + siz_mean.view(1, 1, 2)


def layout_loss(
    pred_pos: torch.Tensor,
    pred_ang: torch.Tensor,
    pred_siz: torch.Tensor,
    clean_pos: torch.Tensor,
    clean_ang: torch.Tensor,
    clean_siz: torch.Tensor,
    valid_mask: torch.Tensor,
    pred_pos_denorm: torch.Tensor,
    pred_siz_denorm: torch.Tensor,
    fpoc: torch.Tensor,
    nfpc: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_loss = smooth_l1_masked(pred_pos, clean_pos, valid_mask)
    siz_loss = smooth_l1_masked(pred_siz, clean_siz, valid_mask)
    ang_loss = smooth_l1_masked(pred_ang, clean_ang, valid_mask)

    coll_loss = collision_loss(pred_pos_denorm, pred_siz_denorm, valid_mask)
    bnd_loss = boundary_loss(pred_pos_denorm, pred_siz_denorm, valid_mask, fpoc, nfpc)

    total = (
        cfg.w_pos * pos_loss
        + cfg.w_siz * siz_loss
        + cfg.w_ang * ang_loss
        + cfg.w_collision * coll_loss
        + cfg.w_boundary * bnd_loss
    )

    return total, {
        "pos": float(pos_loss.detach().cpu()),
        "siz": float(siz_loss.detach().cpu()),
        "ang": float(ang_loss.detach().cpu()),
        "collision": float(coll_loss.detach().cpu()),
        "boundary": float(bnd_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def make_splits(dataset: Dataset, train_ratio: float, seed: int):
    n = len(dataset)
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    cut = int(n * train_ratio)
    train_idx = idx[:cut]
    val_idx = idx[cut:] if cut < n else idx[-max(1, n // 10):]

    return train_idx, val_idx, Subset(dataset, train_idx), Subset(dataset, val_idx)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    class_dim: int,
    stats: dict[str, np.ndarray],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "class_dim": class_dim,
            "stats": stats,
        },
        path,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: Config,
    stats_torch: dict[str, torch.Tensor],
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    sums = {
        "pos": 0.0,
        "siz": 0.0,
        "ang": 0.0,
        "collision": 0.0,
        "boundary": 0.0,
        "total": 0.0,
    }
    steps = 0

    for batch in loader:
        clean_pos = batch["clean_pos"].to(device)
        clean_ang = batch["clean_ang"].to(device)
        clean_siz = batch["clean_siz"].to(device)

        noisy_pos = batch["noisy_pos"].to(device)
        noisy_ang = batch["noisy_ang"].to(device)
        noisy_siz = batch["noisy_siz"].to(device)

        cla = batch["cla"].to(device)
        fpoc = batch["fpoc"].to(device)
        nfpc = batch["nfpc"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            pred_pos, pred_ang, pred_siz = model(
                noisy_pos, noisy_ang, noisy_siz,
                cla, valid_mask,
                fpoc, nfpc,
                denoise_steps=cfg.denoise_steps,
            )

            pred_pos_denorm = denorm_pos(pred_pos, stats_torch["pos_mean"], stats_torch["pos_std"])
            pred_siz_denorm = denorm_siz(pred_siz, stats_torch["siz_mean"], stats_torch["siz_std"])
            pred_siz_denorm = torch.clamp(pred_siz_denorm, min=1e-4)

            loss, parts = layout_loss(
                pred_pos, pred_ang, pred_siz,
                clean_pos, clean_ang, clean_siz,
                valid_mask,
                pred_pos_denorm, pred_siz_denorm,
                fpoc, nfpc,
                cfg,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for k in sums:
            sums[k] += parts[k]
        steps += 1

    means = {k: sums[k] / max(steps, 1) for k in sums}
    return means["total"], means


@torch.no_grad()
def save_val_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    stats_torch: dict[str, torch.Tensor],
    save_path: str,
    max_scenes: int,
) -> None:
    model.eval()

    all_scenedirs = []
    all_valid = []
    all_clean_pos = []
    all_clean_ang = []
    all_clean_siz = []
    all_noisy_pos = []
    all_noisy_ang = []
    all_noisy_siz = []
    all_pred_pos = []
    all_pred_ang = []
    all_pred_siz = []

    saved = 0

    for batch in loader:
        noisy_pos = batch["noisy_pos"].to(device)
        noisy_ang = batch["noisy_ang"].to(device)
        noisy_siz = batch["noisy_siz"].to(device)

        cla = batch["cla"].to(device)
        fpoc = batch["fpoc"].to(device)
        nfpc = batch["nfpc"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        pred_pos, pred_ang, pred_siz = model(
            noisy_pos, noisy_ang, noisy_siz,
            cla, valid_mask,
            fpoc, nfpc,
            denoise_steps=cfg.denoise_steps,
        )

        pred_pos_denorm = denorm_pos(pred_pos, stats_torch["pos_mean"], stats_torch["pos_std"])
        pred_siz_denorm = denorm_siz(pred_siz, stats_torch["siz_mean"], stats_torch["siz_std"])
        pred_siz_denorm = torch.clamp(pred_siz_denorm, min=1e-4)

        noisy_pos_denorm = batch["noisy_pos_denorm"].cpu().numpy()
        noisy_ang_cpu = batch["noisy_ang"].cpu().numpy()
        noisy_siz_denorm = batch["noisy_siz_denorm"].cpu().numpy()

        clean_pos_denorm = batch["clean_pos_denorm"].cpu().numpy()
        clean_ang_cpu = batch["clean_ang"].cpu().numpy()
        clean_siz_denorm = batch["clean_siz_denorm"].cpu().numpy()

        pred_pos_denorm = pred_pos_denorm.cpu().numpy()
        pred_ang_cpu = pred_ang.cpu().numpy()
        pred_siz_denorm = pred_siz_denorm.cpu().numpy()

        valid_mask_cpu = batch["valid_mask"].cpu().numpy()
        scenedirs = batch["scenedir"]

        B = len(scenedirs)
        for i in range(B):
            if saved >= max_scenes:
                break
            all_scenedirs.append(scenedirs[i])
            all_valid.append(valid_mask_cpu[i])
            all_clean_pos.append(clean_pos_denorm[i])
            all_clean_ang.append(clean_ang_cpu[i])
            all_clean_siz.append(clean_siz_denorm[i])
            all_noisy_pos.append(noisy_pos_denorm[i])
            all_noisy_ang.append(noisy_ang_cpu[i])
            all_noisy_siz.append(noisy_siz_denorm[i])
            all_pred_pos.append(pred_pos_denorm[i])
            all_pred_ang.append(pred_ang_cpu[i])
            all_pred_siz.append(pred_siz_denorm[i])
            saved += 1

        if saved >= max_scenes:
            break

    np.savez_compressed(
        save_path,
        scenedirs=np.array(all_scenedirs, dtype=object),
        valid_mask=np.array(all_valid),
        clean_pos=np.array(all_clean_pos),
        clean_ang=np.array(all_clean_ang),
        clean_siz=np.array(all_clean_siz),
        noisy_pos=np.array(all_noisy_pos),
        noisy_ang=np.array(all_noisy_ang),
        noisy_siz=np.array(all_noisy_siz),
        pred_pos=np.array(all_pred_pos),
        pred_ang=np.array(all_pred_ang),
        pred_siz=np.array(all_pred_siz),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints_layout")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--pos_noise_std", type=float, default=0.08)
    parser.add_argument("--siz_noise_std", type=float, default=0.05)
    parser.add_argument("--ang_noise_std_deg", type=float, default=15.0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--denoise_steps", type=int, default=3)

    parser.add_argument("--w_pos", type=float, default=1.0)
    parser.add_argument("--w_siz", type=float, default=1.0)
    parser.add_argument("--w_ang", type=float, default=0.5)
    parser.add_argument("--w_collision", type=float, default=0.2)
    parser.add_argument("--w_boundary", type=float, default=0.2)

    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_save_count", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])

    args = parser.parse_args()

    cfg = Config(
        npz_path=args.npz_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        seed=args.seed,
        num_workers=args.num_workers,
        pos_noise_std=args.pos_noise_std,
        siz_noise_std=args.siz_noise_std,
        ang_noise_std_deg=args.ang_noise_std_deg,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        denoise_steps=args.denoise_steps,
        w_pos=args.w_pos,
        w_siz=args.w_siz,
        w_ang=args.w_ang,
        w_collision=args.w_collision,
        w_boundary=args.w_boundary,
        save_every=args.save_every,
        val_save_count=args.val_save_count,
        device=args.device,
    )

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "mps":
        print("Используется Apple Metal (MPS)")

    dataset = FrontLayoutDataset(
        npz_path=cfg.npz_path,
        pos_noise_std=cfg.pos_noise_std,
        siz_noise_std=cfg.siz_noise_std,
        ang_noise_std_deg=cfg.ang_noise_std_deg,
    )

    train_idx, val_idx, train_ds, val_ds = make_splits(dataset, cfg.train_split, cfg.seed)
    stats = compute_train_stats(dataset, train_idx)
    dataset.set_normalization_stats(
        stats["pos_mean"], stats["pos_std"],
        stats["siz_mean"], stats["siz_std"],
    )

    with open(os.path.join(cfg.out_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pos_mean": stats["pos_mean"].tolist(),
                "pos_std": stats["pos_std"].tolist(),
                "siz_mean": stats["siz_mean"].tolist(),
                "siz_std": stats["siz_std"].tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"npz_path: {cfg.npz_path}")
    print(f"num_scenes: {dataset.num_scenes}")
    print(f"max_objects: {dataset.max_objects}")
    print(f"class_dim: {dataset.class_dim}")
    print(f"floorplan_points: {dataset.floorplan_points}")
    print(f"train size: {len(train_idx)}")
    print(f"val size: {len(val_idx)}")
    print("stats:")
    print("  pos_mean:", stats["pos_mean"])
    print("  pos_std :", stats["pos_std"])
    print("  siz_mean:", stats["siz_mean"])
    print("  siz_std :", stats["siz_std"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    model = LayoutRefiner(
        class_dim=dataset.class_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    stats_torch = {
        "pos_mean": torch.tensor(stats["pos_mean"], dtype=torch.float32, device=device),
        "pos_std": torch.tensor(stats["pos_std"], dtype=torch.float32, device=device),
        "siz_mean": torch.tensor(stats["siz_mean"], dtype=torch.float32, device=device),
        "siz_std": torch.tensor(stats["siz_std"], dtype=torch.float32, device=device),
    }

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_parts = run_epoch(model, train_loader, optimizer, device, cfg, stats_torch)
        val_loss, val_parts = run_epoch(model, val_loader, None, device, cfg, stats_torch)

        print(
            f"[{epoch:03d}/{cfg.epochs:03d}] "
            f"train={train_loss:.6f} "
            f"(pos={train_parts['pos']:.6f}, siz={train_parts['siz']:.6f}, ang={train_parts['ang']:.6f}, "
            f"col={train_parts['collision']:.6f}, bnd={train_parts['boundary']:.6f}) "
            f"| val={val_loss:.6f} "
            f"(pos={val_parts['pos']:.6f}, siz={val_parts['siz']:.6f}, ang={val_parts['ang']:.6f}, "
            f"col={val_parts['collision']:.6f}, bnd={val_parts['boundary']:.6f})"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(cfg.out_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                cfg,
                dataset.class_dim,
                stats,
            )
            save_val_predictions(
                model,
                val_loader,
                device,
                cfg,
                stats_torch,
                os.path.join(cfg.out_dir, "val_preds_best.npz"),
                max_scenes=cfg.val_save_count,
            )

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.out_dir, f"epoch_{epoch:03d}.pt"),
                model,
                optimizer,
                epoch,
                cfg,
                dataset.class_dim,
                stats,
            )

    print(f"Training finished. best_val={best_val:.6f}")


if __name__ == "__main__":
    main()