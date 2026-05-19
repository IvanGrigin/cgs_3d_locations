#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate real interior photos with the same VLM score schema as review frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.evaluate_vlm_review_views import (  # noqa: E402
    LAYOUT_EVAL_SCHEMA,
    _evaluate_images,
    _ollama_model_capabilities,
    _score,
)
from src.tools.run_full_circle_room_batch import PROMPTS_BY_ROOM_TYPE  # noqa: E402


DEFAULT_ROOM_TYPES = ("bedroom", "living_room", "kitchen", "toilet")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

REAL_PHOTO_SYSTEM_PROMPT = """
You are a strict VLM evaluator for real interior room photos.

Use the same scoring schema as generated Blender scene review evaluation.
Evaluate only what is visible in the photo and compare it to the user prompt.
The room type is provided in the user payload; evaluate against that room type,
not against a generic interior.

These are real photos, not Blender review renders:
- do not penalize a photo for being eye-level, perspective, cropped, handheld,
  or missing top-view/oblique review angles;
- do penalize insufficient visibility only when the photo is too cropped, blurry,
  dark, occluded, or otherwise does not show enough of the room to judge it;
- collision_score should mean real-world physical plausibility: no impossible
  intersections, blocked circulation, or visibly broken layout.

Use score meanings consistently:
- prompt_match_score: required room objects and prompt intent are visible.
- layout_score: placement, circulation, reachability, wall/floor consistency.
- collision_score: no visible intersections, floating objects, or impossible layout.
- style_score: modern/cozy/practical style match for the specified room.
- asset_quality_score: realism, detail, material quality, and furniture/object quality.
- camera_coverage_score: the photo exposes enough of the room to judge it.
- total_score: weighted overall result, not just layout.

Scores are 0..10 where 10 is best. Return ONLY JSON matching the schema.
No markdown, no prose outside JSON.
""".strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate real photos with the layout VLM prompt set.")
    p.add_argument("--photos-dir", type=Path, default=Path("data/input/real_photos"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--provider", choices=["ollama", "openai", "openrouter"], default="ollama")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11437")
    p.add_argument("--model", default="qwen2.5vl:7b")
    p.add_argument("--room-types", default=",".join(DEFAULT_ROOM_TYPES))
    p.add_argument(
        "--photo-room-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping. Supports either {'IMG_7318.jpg':'toilet'} or "
            "{'toilet':['IMG_7318.jpg', ...]} formats. Mapped photos use only their room prompt."
        ),
    )
    p.add_argument(
        "--only-mapped",
        action="store_true",
        help="Skip photos that are not present in --photo-room-map.",
    )
    p.add_argument("--style-label", default="modern")
    p.add_argument("--timeout-sec", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--max-image-side",
        type=int,
        default=1600,
        help="Resize images before sending to VLM so the longest side is at most this many pixels. Use 0 to disable.",
    )
    p.add_argument("--jpeg-quality", type=int, default=88)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--allow-non-vision-model",
        action="store_true",
        help="Debug only: do not fail when Ollama model capabilities do not include vision.",
    )
    return p.parse_args()


def _default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out/runs") / f"real_photos_vlm_eval_{stamp}"


def _photo_paths(photos_dir: Path, limit: int) -> list[Path]:
    photos = sorted(
        p for p in photos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit > 0:
        photos = photos[:limit]
    return photos


def _load_photo_room_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"photo room map must be a JSON object: {path}")
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            mapping[str(key)] = value
        elif isinstance(value, list):
            room_type = str(key)
            for item in value:
                mapping[str(item)] = room_type
        else:
            raise SystemExit(f"unsupported map value for {key!r}: expected string or list")
    return mapping


def _prepare_image_for_vlm(*, photo: Path, out_dir: Path, max_side: int, jpeg_quality: int) -> Path:
    if max_side <= 0:
        return photo
    resized_dir = out_dir / "_resized"
    resized_dir.mkdir(parents=True, exist_ok=True)
    out_path = resized_dir / f"{photo.stem}.max{max_side}.jpg"
    if out_path.is_file() and out_path.stat().st_mtime >= photo.stat().st_mtime:
        return out_path
    with Image.open(photo) as im:
        im = im.convert("RGB")
        longest = max(im.size)
        if longest <= max_side:
            im.save(out_path, format="JPEG", quality=int(jpeg_quality), optimize=True)
            return out_path
        scale = float(max_side) / float(longest)
        new_size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        im.save(out_path, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return out_path


def _payload(*, photo: Path, room_type: str, prompt: str, style_label: str) -> str:
    data = {
        "source": "real_photo",
        "photo_name": photo.name,
        "original_prompt": prompt,
        "room_type": room_type,
        "style_label": style_label,
        "frame_count": 1,
        "frames": [
            {
                "index": 0,
                "name": photo.name,
                "view_type": "real_photo",
            }
        ],
        "evaluation_task": (
            f"Assess this real photo as a candidate {room_type} reference for the same prompt used "
            "in generated-scene evaluation. Judge visible prompt match, layout/circulation, "
            "physical plausibility, style, asset quality, and camera coverage."
        ),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _row(*, photo: Path, room_type: str, raw: dict[str, Any], out_json: Path, wall_sec: float) -> dict[str, Any]:
    return {
        "photo": photo.name,
        "room_type": room_type,
        "total_score": _score(raw, "total_score"),
        "prompt_match_score": _score(raw, "prompt_match_score"),
        "layout_score": _score(raw, "layout_score"),
        "collision_score": _score(raw, "collision_score"),
        "style_score": _score(raw, "style_score"),
        "asset_quality_score": _score(raw, "asset_quality_score"),
        "camera_coverage_score": _score(raw, "camera_coverage_score"),
        "confidence": _score(raw, "confidence"),
        "passed": bool(raw.get("passed", False)),
        "wall_sec": round(float(wall_sec), 3),
        "output": str(out_json),
    }


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_json = out_dir / "real_photo_vlm_eval_summary.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not rows:
        return
    out_csv = out_dir / "real_photo_vlm_eval_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {out_json}")
    print(f"summary csv: {out_csv}")


def main() -> None:
    args = _parse_args()
    photos_dir = args.photos_dir.expanduser().resolve()
    if not photos_dir.is_dir():
        raise SystemExit(f"photos dir not found: {photos_dir}")

    out_dir = (args.out_dir or _default_out_dir()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    room_types = [x.strip() for x in str(args.room_types).split(",") if x.strip()]
    for room_type in room_types:
        if room_type not in PROMPTS_BY_ROOM_TYPE:
            raise SystemExit(f"unsupported room type: {room_type}")
    photo_room_map = _load_photo_room_map(args.photo_room_map)
    for photo_name, room_type in sorted(photo_room_map.items()):
        if room_type not in PROMPTS_BY_ROOM_TYPE:
            raise SystemExit(f"unsupported room type in map for {photo_name}: {room_type}")

    if args.provider == "ollama":
        if not args.model:
            raise SystemExit("--model is required for --provider ollama")
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

    photos = _photo_paths(photos_dir, int(args.limit))
    if not photos:
        raise SystemExit(f"no image files found under {photos_dir}")

    rows: list[dict[str, Any]] = []
    map_note = f" mapped={len(photo_room_map)}" if photo_room_map else ""
    print(f"photos={len(photos)} room_types={','.join(room_types)}{map_note} out={out_dir}", flush=True)
    for photo in photos:
        if photo_room_map:
            mapped_room_type = photo_room_map.get(photo.name) or photo_room_map.get(photo.stem)
            if mapped_room_type:
                eval_room_types = [mapped_room_type]
            elif args.only_mapped:
                print(f"skip unmapped photo: {photo.name}", flush=True)
                continue
            else:
                eval_room_types = room_types
        else:
            eval_room_types = room_types

        photo_dir = out_dir / photo.stem
        photo_dir.mkdir(parents=True, exist_ok=True)
        for room_type in eval_room_types:
            prompt = PROMPTS_BY_ROOM_TYPE[room_type]
            out_json = photo_dir / f"{room_type}.eval.json"
            meta_json = photo_dir / f"{room_type}.meta.json"
            payload_json = photo_dir / f"{room_type}.payload.json"

            if args.skip_existing and out_json.is_file():
                raw = json.loads(out_json.read_text(encoding="utf-8"))
                wall_sec = 0.0
                if meta_json.is_file():
                    try:
                        wall_sec = float(json.loads(meta_json.read_text(encoding="utf-8")).get("wall_sec", 0.0))
                    except Exception:
                        wall_sec = 0.0
                rows.append(_row(photo=photo, room_type=room_type, raw=raw, out_json=out_json, wall_sec=wall_sec))
                print(f"skip existing: {photo.name}/{room_type}", flush=True)
                continue

            user_payload = _payload(
                photo=photo,
                room_type=room_type,
                prompt=prompt,
                style_label=str(args.style_label),
            )
            payload_json.write_text(user_payload, encoding="utf-8")

            print(f"VLM evaluate real photo: {photo.name} as {room_type}", flush=True)
            send_image = _prepare_image_for_vlm(
                photo=photo,
                out_dir=out_dir,
                max_side=int(args.max_image_side),
                jpeg_quality=int(args.jpeg_quality),
            )
            if send_image != photo:
                print(f"  resized input: {send_image}", flush=True)
            t0 = perf_counter()
            try:
                raw, provider_meta = _evaluate_images(
                    provider=args.provider,
                    ollama_url=args.ollama_url,
                    model=args.model,
                    system_prompt=REAL_PHOTO_SYSTEM_PROMPT,
                    user_payload=user_payload,
                    send_images=[send_image],
                    timeout_sec=int(args.timeout_sec),
                    temperature=float(args.temperature),
                )
            except Exception as exc:
                err = {
                    "photo": photo.name,
                    "room_type": room_type,
                    "model": args.model,
                    "error": repr(exc),
                    "wall_sec": round(perf_counter() - t0, 3),
                }
                (photo_dir / f"{room_type}.error.json").write_text(
                    json.dumps(err, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"ERROR real photo: {photo.name}/{room_type}: {exc!r}", file=sys.stderr, flush=True)
                _write_summary(out_dir, rows)
                raise SystemExit(2) from exc

            wall_sec = perf_counter() - t0
            out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            meta = {
                "photo": str(photo),
                "room_type": room_type,
                "prompt": prompt,
                "provider": args.provider,
                "model": provider_meta.get("model") or args.model,
                "provider_meta": provider_meta,
                "sent_image": str(send_image),
                "max_image_side": int(args.max_image_side),
                "wall_sec": round(wall_sec, 3),
            }
            meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            row = _row(photo=photo, room_type=room_type, raw=raw, out_json=out_json, wall_sec=wall_sec)
            rows.append(row)
            print(
                f"OK real photo: {photo.name}/{room_type}: total={row['total_score']:.1f} "
                f"layout={row['layout_score']:.1f}",
                flush=True,
            )

    rows_sorted = sorted(rows, key=lambda r: (str(r.get("photo")), str(r.get("room_type"))))
    _write_summary(out_dir, rows_sorted)


if __name__ == "__main__":
    main()
