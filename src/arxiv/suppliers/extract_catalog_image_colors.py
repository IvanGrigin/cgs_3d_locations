#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract center-weighted dominant colors from one product image per supplier row."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


DEFAULT_CATALOG = "data/sourse/suppliers/supplier_catalog_canonical.json"
DEFAULT_CACHE_DIR = "out/supplier_image_colors/images_512"
DEFAULT_OUT_JSONL = "reports/supplier_image_colors/supplier_catalog_canonical.image_colors.jsonl"
DEFAULT_SUMMARY_JSON = "reports/supplier_image_colors/summary.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _candidate_images(row: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    preview = str(row.get("preview_local_path") or "").strip()
    if preview:
        candidates.append({"kind": "local", "value": preview, "field": "preview_local_path"})

    raw_images = row.get("images")
    if not isinstance(raw_images, list):
        raw_images = []
    for idx, value in enumerate(raw_images):
        if isinstance(value, dict):
            raw = str(value.get("url") or value.get("src") or value.get("image") or value.get("path") or "").strip()
        else:
            raw = str(value or "").strip()
        if not raw:
            continue
        candidates.append({"kind": "url" if _is_url(raw) else "local", "value": raw, "field": f"images[{idx}]"})
    return candidates


def _download_image_urllib(url: str, dest: Path, timeout: float, user_agent: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = str(resp.headers.get("Content-Type") or "").lower()
            if content_type and "image" not in content_type:
                return False
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    if not data:
        return False
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True


def _download_image_curl(url: str, dest: Path, timeout: float, user_agent: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        str(int(max(timeout, 1.0))),
        "-A",
        user_agent,
        "-o",
        str(tmp),
        url,
    ]
    try:
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return False
    if result.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


def _download_image(url: str, dest: Path, timeout: float, user_agent: str, downloader: str) -> bool:
    mode = str(downloader or "auto").strip().lower()
    if mode == "curl":
        return _download_image_curl(url, dest, timeout, user_agent)
    if mode == "urllib":
        return _download_image_urllib(url, dest, timeout, user_agent)
    return _download_image_urllib(url, dest, timeout, user_agent) or _download_image_curl(url, dest, timeout, user_agent)


def _resolve_image(
    row: dict[str, Any],
    *,
    cache_dir: Path,
    download: bool,
    timeout: float,
    user_agent: str,
    downloader: str,
) -> dict[str, Any] | None:
    unique_key = str(row.get("unique_key") or "").strip()
    for candidate in _candidate_images(row):
        value = candidate["value"]
        if candidate["kind"] == "local":
            path = Path(value).expanduser()
            if path.is_file():
                return {**candidate, "path": str(path.resolve()), "downloaded": False}
            continue

        suffix = Path(urllib.parse.urlparse(value).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".jpg"
        cache_path = cache_dir / f"{_stable_key(unique_key or value)}{suffix}"
        if cache_path.is_file():
            return {**candidate, "path": str(cache_path.resolve()), "downloaded": False}
        if not download:
            continue
        if _download_image(value, cache_path, timeout, user_agent, downloader):
            return {**candidate, "path": str(cache_path.resolve()), "downloaded": True}
    return None


def _open_rgb_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in {"RGBA", "LA"}:
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                bg.alpha_composite(img.convert("RGBA"))
                img = bg.convert("RGB")
            else:
                img = img.convert("RGB")
            return img
    except Exception:
        return None


def _resize_to_square(img: Image.Image, size: int = 512) -> Image.Image:
    out = ImageOps.contain(img, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    x = (size - out.width) // 2
    y = (size - out.height) // 2
    canvas.paste(out, (x, y))
    return canvas


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = maxc - minc
    dist_white = np.linalg.norm(255.0 - arr, axis=2)
    dist_black = np.linalg.norm(arr, axis=2)
    return (
        (dist_white > 24.0)
        & (dist_black > 10.0)
        & ~((maxc > 242.0) & (saturation < 10.0))
    )


def _center_weights(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    sigma = min(width, height) * 0.34
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return 0.22 + np.exp(-dist2 / (2.0 * sigma * sigma))


def _sample_pixels(rgb: np.ndarray, mask: np.ndarray, weights: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) < 64:
        ys, xs = np.where(np.ones(mask.shape, dtype=bool))
    pixels = rgb[ys, xs].astype(np.float32)
    sample_weights = weights[ys, xs].astype(np.float32)
    if len(pixels) > max_samples:
        # Deterministic weighted spread: keep center-priority points without random state.
        order = np.argsort(-sample_weights)
        head_count = max_samples // 2
        head = order[:head_count]
        tail_pool = order[head_count:]
        if len(tail_pool):
            step = max(1, len(tail_pool) // max(1, max_samples - head_count))
            tail = tail_pool[::step][: max_samples - head_count]
            keep = np.concatenate([head, tail])
        else:
            keep = head
        pixels = pixels[keep]
        sample_weights = sample_weights[keep]
    return pixels, sample_weights


def _weighted_kmeans(pixels: np.ndarray, weights: np.ndarray, k: int, iterations: int = 18) -> tuple[np.ndarray, np.ndarray]:
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    unique = np.unique(np.round(pixels / 8.0) * 8.0, axis=0)
    k = max(1, min(k, len(unique)))
    luminance = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    qs = np.linspace(0.08, 0.92, k)
    centers = np.array([pixels[np.argmin(np.abs(luminance - np.quantile(luminance, q)))] for q in qs], dtype=np.float32)

    labels = np.zeros((len(pixels),), dtype=np.int32)
    for _ in range(iterations):
        dist = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            m = labels == idx
            if not np.any(m):
                continue
            w = weights[m]
            centers[idx] = (pixels[m] * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-6)

    cluster_weights = np.array([weights[labels == idx].sum() for idx in range(k)], dtype=np.float32)
    order = np.argsort(-cluster_weights)
    return centers[order], cluster_weights[order]


def _hex(rgb: np.ndarray) -> str:
    values = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _basic_color_name(rgb: np.ndarray) -> str:
    r, g, b = [max(0.0, min(1.0, float(x) / 255.0)) for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v <= 0.18:
        return "black"
    if s <= 0.12:
        return "white" if v >= 0.86 else "gray"
    if 0.07 <= h <= 0.16 and v > 0.68 and s < 0.45:
        return "beige"
    if 0.06 <= h <= 0.14 and v < 0.70:
        return "brown"
    if h < 0.04 or h >= 0.96:
        return "red"
    if h < 0.10:
        return "orange"
    if h < 0.16:
        return "yellow"
    if h < 0.42:
        return "green"
    if h < 0.72:
        return "blue"
    if h < 0.86:
        return "purple"
    return "red"


def _palette(rgb: np.ndarray, k: int) -> list[dict[str, Any]]:
    mask = _foreground_mask(rgb)
    weights_img = _center_weights(rgb.shape[0], rgb.shape[1])
    pixels, weights = _sample_pixels(rgb, mask, weights_img, max_samples=24000)
    centers, cluster_weights = _weighted_kmeans(pixels, weights, k)
    total = max(float(cluster_weights.sum()), 1e-6)
    out: list[dict[str, Any]] = []
    for center, weight in zip(centers, cluster_weights):
        out.append(
            {
                "hex": _hex(center),
                "rgb": [int(x) for x in np.clip(np.rint(center), 0, 255)],
                "basic_color": _basic_color_name(center),
                "weight": round(float(weight / total), 6),
            }
        )
    return out


def extract_row_colors(
    row: dict[str, Any],
    *,
    cache_dir: Path,
    download: bool,
    timeout: float,
    user_agent: str,
    downloader: str,
) -> dict[str, Any]:
    resolved = _resolve_image(
        row,
        cache_dir=cache_dir,
        download=download,
        timeout=timeout,
        user_agent=user_agent,
        downloader=downloader,
    )
    base = {
        "unique_key": row.get("unique_key"),
        "source_site": row.get("source_site"),
        "title": row.get("title"),
        "product_url": row.get("product_url") or row.get("source_url"),
    }
    if not resolved:
        return {**base, "status": "missing_image"}

    img = _open_rgb_image(Path(resolved["path"]))
    if img is None:
        return {**base, "status": "image_open_failed", "image": resolved}

    resized = _resize_to_square(img, 512)
    cache_jpg = cache_dir / f"{_stable_key(str(row.get('unique_key') or resolved['value']))}.512.jpg"
    cache_jpg.parent.mkdir(parents=True, exist_ok=True)
    resized.save(cache_jpg, "JPEG", quality=88, optimize=True)
    rgb = np.asarray(resized.convert("RGB"))
    mask = _foreground_mask(rgb)

    colors = {
        "one": _palette(rgb, 1),
        "top2": _palette(rgb, 2),
        "top5": _palette(rgb, 5),
    }
    color_tokens = []
    for entry in colors["top5"]:
        name = str(entry.get("basic_color") or "").strip()
        if name and name not in color_tokens:
            color_tokens.append(name)

    return {
        **base,
        "status": "ok",
        "image": {
            **resolved,
            "resized_512_path": str(cache_jpg.resolve()),
        },
        "foreground_ratio": round(float(mask.mean()), 6),
        "colors": colors,
        "color_tokens": color_tokens,
        "method": {
            "resize": "contain_512_on_white",
            "mask": "non_white_non_black_foreground",
            "weighting": "center_gaussian_plus_base",
            "clusterer": "weighted_kmeans_rgb",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract center-weighted dominant colors from supplier product images.")
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    ap.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    ap.add_argument("--download", action="store_true", help="Download remote image URLs into the cache.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--user-agent", default="Mozilla/5.0")
    ap.add_argument("--downloader", choices=["auto", "urllib", "curl"], default="auto")
    ap.add_argument("--merged-catalog-out", default=None, help="Optional catalog copy with image_color_features merged by unique_key.")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    catalog_path = Path(args.catalog).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    out_jsonl = Path(args.out_jsonl).expanduser()
    summary_json = Path(args.summary_json).expanduser()
    payload = _read_json(catalog_path)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RuntimeError(f"Invalid supplier catalog: {catalog_path}")

    existing: set[str] = set()
    if args.skip_existing and out_jsonl.is_file():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = str(row.get("unique_key") or "").strip()
            if key:
                existing.add(key)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_to_process: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        key = str(row.get("unique_key") or "").strip()
        if key in existing:
            continue
        if args.limit and len(rows_to_process) >= args.limit:
            break
        rows_to_process.append(row)

    counts: dict[str, int] = {}
    started = time.time()
    processed = 0
    with out_jsonl.open("a" if args.skip_existing else "w", encoding="utf-8") as out:
        workers = max(1, int(args.workers or 1))

        def run_one(row: dict[str, Any]) -> dict[str, Any]:
            return extract_row_colors(
                row,
                cache_dir=cache_dir,
                download=bool(args.download),
                timeout=float(args.timeout),
                user_agent=str(args.user_agent),
                downloader=str(args.downloader),
            )

        if workers == 1:
            iterator = (run_one(row) for row in rows_to_process)
            for result in iterator:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                processed += 1
                status = str(result.get("status") or "unknown")
                counts[status] = counts.get(status, 0) + 1
                if processed % 100 == 0:
                    print(f"processed={processed} ok={counts.get('ok', 0)} missing={counts.get('missing_image', 0)}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_one, row) for row in rows_to_process]
                for future in as_completed(futures):
                    result = future.result()
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    processed += 1
                    status = str(result.get("status") or "unknown")
                    counts[status] = counts.get(status, 0) + 1
                    if processed % 100 == 0:
                        print(
                            f"processed={processed} ok={counts.get('ok', 0)} missing={counts.get('missing_image', 0)}",
                            flush=True,
                        )

    _write_json(
        summary_json,
        {
            "catalog": str(catalog_path),
            "out_jsonl": str(out_jsonl),
            "cache_dir": str(cache_dir),
            "download": bool(args.download),
            "processed": processed,
            "status_counts": counts,
            "elapsed_sec": round(time.time() - started, 3),
        },
    )
    if args.merged_catalog_out:
        by_key: dict[str, dict[str, Any]] = {}
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                result = json.loads(line)
            except Exception:
                continue
            if result.get("status") != "ok":
                continue
            key = str(result.get("unique_key") or "").strip()
            if key:
                by_key[key] = result
        merged = dict(payload) if isinstance(payload, dict) else {"items": items}
        merged_items = []
        for item in items:
            if not isinstance(item, dict):
                merged_items.append(item)
                continue
            key = str(item.get("unique_key") or "").strip()
            result = by_key.get(key)
            if not result:
                merged_items.append(item)
                continue
            new_item = dict(item)
            new_item["image_color_features"] = {
                "source_image": result.get("image"),
                "foreground_ratio": result.get("foreground_ratio"),
                "colors": result.get("colors"),
                "color_tokens": result.get("color_tokens") or [],
                "method": result.get("method"),
            }
            merged_items.append(new_item)
        merged["items"] = merged_items
        _write_json(Path(args.merged_catalog_out).expanduser(), merged)
    print(json.dumps(_read_json(summary_json), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
