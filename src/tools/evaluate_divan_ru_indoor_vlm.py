#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Divan.ru indoor reference photos with the Infinigen VLM-style rubric."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.llm_vlm_layout_refinement import evaluation as ev  # noqa: E402
from src.tools.evaluate_vlm_review_views import _ollama_model_capabilities, _score  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

ROOM_PROMPTS: dict[str, str] = {
    "гостиная": (
        "Evaluate this photo as a living room interior reference. Expected visible elements may include "
        "a sofa or armchairs, coffee table, TV/media unit, shelving/storage, rug, lamps, curtains, decor, "
        "and usable circulation."
    ),
    "спальня": (
        "Evaluate this photo as a bedroom interior reference. Expected visible elements may include "
        "a bed, bedside tables, wardrobe/storage, lighting, rug, curtains, desk or chair if present, "
        "and usable circulation around the bed."
    ),
    "столовая": (
        "Evaluate this photo as a dining room interior reference. Expected visible elements may include "
        "a dining table, dining chairs, lighting over/near the table, storage or decor, and enough "
        "clearance around the dining area."
    ),
    "кухня-гостиная": (
        "Evaluate this photo as a combined kitchen-living room interior reference. Expected visible "
        "elements may include kitchen cabinets/countertop, sink or appliances when visible, dining or "
        "living seating, storage, lighting, and coherent zoning/circulation."
    ),
    "ванная": (
        "Evaluate this photo as a bathroom interior reference. Expected visible elements may include "
        "a sink/vanity, toilet, bath or shower, mirror, storage, lighting, tiled or wet-room materials, "
        "and safe usable circulation."
    ),
    "кабинет": (
        "Evaluate this photo as a home office/study interior reference. Expected visible elements may "
        "include a desk, work chair, shelving/storage, task lighting, decor, and practical clearance "
        "for working."
    ),
}

DEFAULT_ROOM_MAP: dict[str, str] = {
    "001": "гостиная",
    "002": "гостиная",
    "007": "гостиная",
    "013": "гостиная",
    "014": "гостиная",
    "015": "гостиная",
    "016": "гостиная",
    "017": "гостиная",
    "019": "гостиная",
    "021": "гостиная",
    "022": "гостиная",
    "003": "спальня",
    "004": "спальня",
    "006": "спальня",
    "018": "спальня",
    "005": "столовая",
    "020": "столовая",
    "008": "кухня-гостиная",
    "023": "кухня-гостиная",
    "009": "ванная",
    "011": "ванная",
    "010": "кабинет",
    "012": "кабинет",
}

DIVAN_EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "total_score": {"type": "number"},
        "prompt_match_score": {"type": "number"},
        "layout_score": {"type": "number"},
        "collision_score": {"type": "number"},
        "asset_quality_score": {"type": "number"},
        "camera_coverage_score": {"type": "number"},
        "confidence": {"type": "number"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "visible_problems": {"type": "array", "items": {"type": "string"}},
        "recommended_fixes": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": [
        "passed",
        "total_score",
        "prompt_match_score",
        "layout_score",
        "collision_score",
        "asset_quality_score",
        "camera_coverage_score",
        "confidence",
        "strengths",
        "weaknesses",
        "visible_problems",
        "recommended_fixes",
        "notes",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
You are a strict VLM evaluator for real interior reference photos.

Use nearly the same practical rubric as generated Blender/Infinigen scene review,
but these are real Divan.ru interior photos, not standardized review renders.
The room category is explicitly provided in Russian in the user payload; use that
category, not the filename and not your own room classification.

Do not evaluate style and do not include style_score. Do not penalize unusual,
editorial, eye-level, wide-angle, cropped, asymmetric, or perspective camera
angles. Penalize camera coverage only when the room is too blurry, dark, cropped,
occluded, or close-up to judge the requested room category.

Use score meanings consistently:
- prompt_match_score: required objects and room-category intent are visible.
- layout_score: placement, circulation, reachability, zoning, wall/floor consistency.
- collision_score: real-world physical plausibility; no impossible intersections,
  blocked circulation, floating objects, or broken layout.
- asset_quality_score: realism, detail, material/furniture quality, object quality.
- camera_coverage_score: enough of the room is visible to judge it.
- total_score: weighted overall result, not just layout.

Scores are 0..10 where 10 is best. Return ONLY JSON matching the schema.
No markdown, no prose outside JSON.
""".strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Divan.ru indoor photos with local VLM.")
    p.add_argument("--images-dir", type=Path, default=Path("data/input/divan_ru_indoor/images"))
    p.add_argument("--out-dir", type=Path, default=Path("data/input/divan_ru_indoor/vlm_eval_20260516"))
    p.add_argument("--room-map-json", type=Path, default=None)
    p.add_argument("--ollama-url", default="http://127.0.0.1:11437")
    p.add_argument("--model", default="qwen2.5vl:7b")
    p.add_argument("--timeout-sec", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-image-side", type=int, default=1280)
    p.add_argument("--jpeg-quality", type=int, default=86)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--allow-non-vision-model", action="store_true")
    return p.parse_args()


def _load_room_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_ROOM_MAP)
    raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"room map must be a JSON object: {path}")
    result: dict[str, str] = {}
    for key, value in raw.items():
        category = str(value).strip()
        if category not in ROOM_PROMPTS:
            raise SystemExit(f"unsupported room category for {key}: {category}")
        num = str(key).strip()
        if num.isdigit():
            num = f"{int(num):03d}"
        result[num] = category
    return result


def _image_number(path: Path) -> str | None:
    m = re.match(r"^(\d{3})_", path.name)
    return m.group(1) if m else None


def _image_paths(images_dir: Path, limit: int) -> list[Path]:
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return images[:limit] if limit > 0 else images


def _prepare_image_for_vlm(*, image: Path, out_dir: Path, max_side: int, jpeg_quality: int) -> Path:
    if max_side <= 0:
        return image
    resized_dir = out_dir / "_resized"
    resized_dir.mkdir(parents=True, exist_ok=True)
    out_path = resized_dir / f"{image.stem}.max{max_side}.jpg"
    if out_path.is_file() and out_path.stat().st_mtime >= image.stat().st_mtime:
        return out_path
    with Image.open(image) as im:
        im = im.convert("RGB")
        longest = max(im.size)
        if longest > max_side:
            scale = float(max_side) / float(longest)
            size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
            im = im.resize(size, Image.Resampling.LANCZOS)
        im.save(out_path, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return out_path


def _payload(*, image: Path, category: str) -> str:
    data = {
        "source": "divan_ru_indoor_photo",
        "photo_name": image.name,
        "room_category_ru": category,
        "category_instruction": "Use this exact room category from the user mapping; do not infer from filename.",
        "original_prompt": ROOM_PROMPTS[category],
        "frame_count": 1,
        "frames": [
            {
                "index": 0,
                "name": image.name,
                "view_type": "real_photo",
            }
        ],
        "evaluation_task": (
            "Assess this real indoor photo as a candidate reference for the provided room category. "
            "Judge prompt/category match, layout/circulation, physical plausibility, asset/material quality, "
            "and camera coverage. Do not evaluate style."
        ),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _evaluate_one(
    *,
    ollama_url: str,
    model: str,
    user_payload: str,
    image: Path,
    timeout_sec: int,
    temperature: float,
) -> dict[str, Any]:
    return ev._ollama_vision_json(
        base_url=ollama_url,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_text=user_payload,
        image_paths=[image],
        timeout_sec=int(timeout_sec),
        temperature=float(temperature),
        response_json_schema=DIVAN_EVAL_SCHEMA,
    )


def _row(*, image: Path, category: str, raw: dict[str, Any], out_json: Path, wall_sec: float) -> dict[str, Any]:
    return {
        "photo": image.name,
        "image_number": _image_number(image) or "",
        "room_category_ru": category,
        "total_score": _score(raw, "total_score"),
        "prompt_match_score": _score(raw, "prompt_match_score"),
        "layout_score": _score(raw, "layout_score"),
        "collision_score": _score(raw, "collision_score"),
        "asset_quality_score": _score(raw, "asset_quality_score"),
        "camera_coverage_score": _score(raw, "camera_coverage_score"),
        "confidence": _score(raw, "confidence"),
        "passed": bool(raw.get("passed", False)),
        "wall_sec": round(float(wall_sec), 3),
        "output": str(out_json),
    }


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: str(r.get("photo", "")))
    out_json = out_dir / "divan_ru_indoor_vlm_summary.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        out_csv = out_dir / "divan_ru_indoor_vlm_summary.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"summary: {out_json}")
        print(f"summary csv: {out_csv}")


def main() -> None:
    args = _parse_args()
    images_dir = args.images_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"images dir not found: {images_dir}")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    room_map = _load_room_map(args.room_map_json)
    (out_dir / "room_map_used.json").write_text(
        json.dumps(room_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        capabilities = _ollama_model_capabilities(
            args.ollama_url,
            args.model,
            timeout_sec=min(30, max(5, int(args.timeout_sec))),
        )
    except Exception as exc:
        raise SystemExit(f"failed to query Ollama model capabilities for {args.model}: {exc!r}") from exc
    if "vision" not in capabilities and not args.allow_non_vision_model:
        raise SystemExit(f"model {args.model!r} is not a vision model; capabilities={capabilities}")

    images = _image_paths(images_dir, int(args.limit))
    mapped_images: list[tuple[Path, str]] = []
    for image in images:
        num = _image_number(image)
        if not num or num not in room_map:
            print(f"skip unmapped image: {image.name}", flush=True)
            continue
        mapped_images.append((image, room_map[num]))
    if not mapped_images:
        raise SystemExit(f"no mapped images found under {images_dir}")

    print(
        f"images={len(mapped_images)} out={out_dir} model={args.model} max_side={args.max_image_side}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for image, category in mapped_images:
        photo_dir = out_dir / image.stem
        photo_dir.mkdir(parents=True, exist_ok=True)
        out_json = photo_dir / "eval.json"
        meta_json = photo_dir / "meta.json"
        payload_json = photo_dir / "payload.json"

        if args.skip_existing and out_json.is_file():
            raw = json.loads(out_json.read_text(encoding="utf-8"))
            wall_sec = 0.0
            if meta_json.is_file():
                try:
                    wall_sec = float(json.loads(meta_json.read_text(encoding="utf-8")).get("wall_sec", 0.0))
                except Exception:
                    wall_sec = 0.0
            rows.append(_row(image=image, category=category, raw=raw, out_json=out_json, wall_sec=wall_sec))
            print(f"skip existing: {image.name}/{category}", flush=True)
            continue

        user_payload = _payload(image=image, category=category)
        payload_json.write_text(user_payload, encoding="utf-8")
        send_image = _prepare_image_for_vlm(
            image=image,
            out_dir=out_dir,
            max_side=int(args.max_image_side),
            jpeg_quality=int(args.jpeg_quality),
        )
        print(f"VLM evaluate divan photo: {image.name} as {category}", flush=True)
        if send_image != image:
            print(f"  resized input: {send_image}", flush=True)

        t0 = perf_counter()
        try:
            raw = _evaluate_one(
                ollama_url=args.ollama_url,
                model=args.model,
                user_payload=user_payload,
                image=send_image,
                timeout_sec=int(args.timeout_sec),
                temperature=float(args.temperature),
            )
        except Exception as exc:
            err = {
                "photo": image.name,
                "room_category_ru": category,
                "model": args.model,
                "error": repr(exc),
                "wall_sec": round(perf_counter() - t0, 3),
            }
            (photo_dir / "error.json").write_text(json.dumps(err, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"ERROR divan photo: {image.name}/{category}: {exc!r}", file=sys.stderr, flush=True)
            _write_summary(out_dir, rows)
            raise SystemExit(2) from exc

        wall_sec = perf_counter() - t0
        out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        meta = {
            "photo": str(image),
            "room_category_ru": category,
            "prompt": ROOM_PROMPTS[category],
            "provider": "ollama",
            "ollama_url": args.ollama_url,
            "model": args.model,
            "sent_image": str(send_image),
            "max_image_side": int(args.max_image_side),
            "jpeg_quality": int(args.jpeg_quality),
            "wall_sec": round(wall_sec, 3),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row = _row(image=image, category=category, raw=raw, out_json=out_json, wall_sec=wall_sec)
        rows.append(row)
        print(
            f"OK divan photo: {image.name}/{category}: total={row['total_score']:.1f} "
            f"layout={row['layout_score']:.1f}",
            flush=True,
        )

    _write_summary(out_dir, rows)


if __name__ == "__main__":
    main()
