#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch VLM evaluation for saved multi-view Blender review frames.

The renderer writes one directory per blend state under ``vlm_review_views``.
This tool evaluates the full set of frames for each state with the existing
Ollama vision helper from ``llm_vlm_layout_refinement.evaluation``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.llm_vlm_layout_refinement import evaluation as ev
from src.topview_vlm_orientation_repair import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    _extract_json_text,
    _load_dotenv_once,
    _openrouter_keys_from_env,
    _provider_config,
    image_to_data_url,
)


FRAME_ORDER = [
    "topview_00_az0.png",
    "topview_01_az72.png",
    "topview_02_az144.png",
    "topview_03_az216.png",
    "topview_04_az288.png",
    "oblique_e60_az45.png",
    "oblique_e60_az135.png",
    "oblique_e60_az225.png",
    "oblique_e60_az315.png",
]


LAYOUT_EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "total_score": {"type": "number"},
        "prompt_match_score": {"type": "number"},
        "layout_score": {"type": "number"},
        "collision_score": {"type": "number"},
        "style_score": {"type": "number"},
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
        "style_score",
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
You are a strict VLM evaluator for generated interior room layouts.

You receive nine frames for one Blender scene state:
- 5 top-view frames of the same room from different azimuths.
- 4 oblique frames at elevation 60 degrees and azimuths 45, 135, 225, 315.
  In oblique frames the two nearest walls should be removed, while the two
  far walls should remain, so judge the room interior rather than the wall cut.

Evaluate only what is visible in the images and compare it to the user prompt.
The room type is provided in the user payload; evaluate against that room type,
not against a generic bedroom. Top-view/isometric review images are intentional,
so do not penalize them for not looking like eye-level real-estate photos.

Each scene state name tells what stage is being reviewed:
- "infinigen_clean_scene" is the raw procedural Infinigen scene.
- "scene_infinigen_clean" is the same layout after local materials/postprocess.
- "scene_infinigen_clean_supplier.optimal" adds supplier/catalog asset replacements.

Score supplier states fairly: reward them when visible furniture, fixtures,
lighting, curtains, rugs, cabinets, vanities, tables, chairs, shelves, or decor
look more realistic, product-like, detailed, and style-consistent than procedural
placeholders. Do not penalize a supplier state just because the exact mesh shape
differs from the raw state. Penalize supplier replacements only when they are
clearly wrong for the room, badly scaled, floating, colliding, duplicated, missing
important parts/materials, or damaging circulation/layout.

Use score meanings consistently:
- prompt_match_score: required room objects and prompt intent are visible.
- layout_score: placement, circulation, reachability, wall/floor consistency.
- collision_score: no visible intersections, floating objects, or out-of-room assets.
- style_score: modern/cozy/practical style match for the specified room.
- asset_quality_score: realism, detail, material quality, and supplier replacement value.
- camera_coverage_score: the review frame(s) expose enough of the room to judge it.
- total_score: weighted overall result, not just layout. If supplier assets are visibly
  more realistic and do not break layout, total_score should improve over the
  non-supplier state.

Focus on:
- required furniture/object presence;
- duplicate or unexpected procedural furniture;
- collisions, overlaps, unreachable circulation, objects outside the room;
- whether walls/floor/curtains/supplier replacements improved or damaged scene;
- style match and asset quality.

Scores are 0..10 where 10 is best. Return ONLY JSON matching the schema.
No markdown, no prose outside JSON.
""".strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate VLM review frame sets under a run directory.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--views-subdir", default="vlm_review_views")
    p.add_argument("--provider", choices=["ollama", "openai", "openrouter"], default="openrouter")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--model", default=None, help="Provider model name")
    p.add_argument("--prompt", required=True, help="Original room prompt")
    p.add_argument("--room-type", default="Bedroom")
    p.add_argument("--style-label", default="modern")
    p.add_argument("--timeout-sec", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--scope",
        choices=["scene", "frames", "both"],
        default="scene",
        help="scene evaluates all frames together; frames evaluates every PNG separately.",
    )
    p.add_argument(
        "--image-mode",
        choices=["contact_sheet", "attachments"],
        default="contact_sheet",
        help="contact_sheet sends one labeled PNG per scene; attachments sends all 9 images separately.",
    )
    p.add_argument("--contact-cell-width", type=int, default=448)
    p.add_argument("--contact-cell-height", type=int, default=336)
    p.add_argument(
        "--allow-non-vision-model",
        action="store_true",
        help="Debug only: do not fail when Ollama model capabilities do not include vision.",
    )
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _view_dirs(views_root: Path) -> list[Path]:
    if not views_root.is_dir():
        raise SystemExit(f"views dir not found: {views_root}")
    return sorted(p for p in views_root.iterdir() if p.is_dir())


def _images_for_dir(view_dir: Path) -> list[Path]:
    images = [view_dir / name for name in FRAME_ORDER]
    missing = [p.name for p in images if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{view_dir}: missing frames: {', '.join(missing)}")
    return images


def _payload(*, view_dir: Path, images: list[Path], args: argparse.Namespace) -> str:
    data = {
        "scene_state": view_dir.name,
        "original_prompt": args.prompt,
        "room_type": args.room_type,
        "style_label": args.style_label,
        "frame_count": len(images),
        "frames": [
            {
                "index": idx,
                "name": image.name,
                "view_type": "topview" if image.name.startswith("topview") else "oblique_e60",
            }
            for idx, image in enumerate(images)
        ],
        "evaluation_task": (
            f"Assess this scene state as a candidate {args.room_type} layout. Compare it to the prompt, "
            "flag visible duplicates/collisions/artifacts, and assign numeric scores."
        ),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _frame_payload(*, view_dir: Path, image: Path, frame_index: int, args: argparse.Namespace) -> str:
    view_type = "topview" if image.name.startswith("topview") else "oblique_e60"
    data = {
        "scene_state": view_dir.name,
        "original_prompt": args.prompt,
        "room_type": args.room_type,
        "style_label": args.style_label,
        "frame_index": frame_index,
        "frame_name": image.name,
        "view_type": view_type,
        "evaluation_task": (
            f"Assess this single frame as one VLM review image for the {args.room_type} layout. "
            "Judge visible prompt match, layout/collisions, style, asset quality, and whether the "
            "review frame exposes enough of the room to evaluate it. "
            "Top-view frames are intentionally top-down/isometric. If this is an oblique view, "
            "the two nearest walls may be removed intentionally."
        ),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _make_contact_sheet(
    *,
    view_dir: Path,
    images: list[Path],
    cell_width: int,
    cell_height: int,
) -> Path:
    cols = 3
    rows = 3
    label_h = 34
    sheet = Image.new("RGB", (cols * cell_width, rows * (cell_height + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, image_path in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = row * (cell_height + label_h)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            px = x + (cell_width - im.width) // 2
            py = y + label_h + (cell_height - im.height) // 2
            sheet.paste(im, (px, py))
        label = f"{idx}: {image_path.name}"
        draw.rectangle((x, y, x + cell_width, y + label_h), fill=(10, 10, 10))
        draw.text((x + 8, y + 10), label, fill=(245, 245, 245), font=font)
    out = view_dir / "vlm_contact_sheet.png"
    sheet.save(out)
    return out


def _score(raw: dict[str, Any], key: str) -> float:
    try:
        return float(raw.get(key, 0.0))
    except Exception:
        return 0.0


def _evaluate_images(
    *,
    provider: str,
    ollama_url: str,
    model: str | None,
    system_prompt: str,
    user_payload: str,
    send_images: list[Path],
    timeout_sec: int,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if provider == "ollama":
        if not model:
            raise RuntimeError("--model is required for provider=ollama")
        raw = ev._ollama_vision_json(
            base_url=ollama_url,
            model=model,
            system_prompt=system_prompt,
            user_text=user_payload,
            image_paths=send_images,
            timeout_sec=int(timeout_sec),
            temperature=float(temperature),
            response_json_schema=LAYOUT_EVAL_SCHEMA,
        )
        return raw, {
            "provider": "ollama",
            "ollama_url": ollama_url,
            "model": model,
        }
    raw, provider_meta = _openai_compatible_vlm_json_multi(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_payload,
        image_paths=send_images,
        timeout_sec=int(timeout_sec),
        temperature=float(temperature),
    )
    return raw, provider_meta


def _ollama_model_capabilities(base_url: str, model: str, timeout_sec: int) -> list[str]:
    url = base_url.rstrip("/") + "/api/show"
    payload = json.dumps({"model": model}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    caps = raw.get("capabilities") or []
    return [str(x) for x in caps]


def _openai_compatible_vlm_json_multi(
    *,
    provider: str,
    model: str | None,
    system_prompt: str,
    user_text: str,
    image_paths: list[Path],
    timeout_sec: int,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = provider.lower().strip()
    _load_dotenv_once()
    if provider == "openrouter":
        keys = _openrouter_keys_from_env()
        if not keys:
            raise RuntimeError("OPENROUTER_API_KEY or ivangrigin_OPENROUTER_API_KEY_* is not set")
        configs = [
            (
                "https://openrouter.ai/api/v1/chat/completions",
                key,
                model or DEFAULT_OPENROUTER_MODEL,
            )
            for key in keys
        ]
    else:
        endpoint, api_key, resolved_model = _provider_config(provider, model or DEFAULT_OPENAI_MODEL)
        configs = [(endpoint, api_key, resolved_model)]

    schema_text = json.dumps(LAYOUT_EVAL_SCHEMA, ensure_ascii=False, indent=2)
    full_user_text = (
        user_text
        + "\n\nReturn a JSON object matching this schema exactly:\n"
        + schema_text
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": full_user_text}]
    for idx, image_path in enumerate(image_paths):
        content.append({"type": "text", "text": f"Frame {idx}: {image_path.name}"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})

    last_error: str | None = None
    for key_index, (endpoint, api_key, resolved_model) in enumerate(configs, start=1):
        payload = {
            "model": resolved_model,
            "temperature": float(temperature),
            "max_tokens": 2200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/IvanGrigin/cgs_3d_locations"
            headers["X-Title"] = "cgs_3d_locations layout VLM review"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                api_response = json.loads(response.read().decode("utf-8"))
            choices = api_response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("VLM response has no choices")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content_obj = message.get("content") if isinstance(message, dict) else None
            if isinstance(content_obj, list):
                text = "\n".join(
                    str(part.get("text", ""))
                    for part in content_obj
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                text = str(content_obj or "")
            parsed = json.loads(_extract_json_text(text))
            return parsed, {
                "provider": provider,
                "endpoint": endpoint,
                "model": resolved_model,
                "raw_api_response": api_response,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail}"
            if provider == "openrouter" and exc.code in {401, 402, 429} and key_index < len(configs):
                continue
            raise RuntimeError(f"VLM request failed: {last_error}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM request failed: {exc}") from exc
    raise RuntimeError(f"VLM request failed: {last_error or 'all keys failed'}")


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    views_root = run_dir / args.views_subdir
    dirs = _view_dirs(views_root)
    if not dirs:
        raise SystemExit(f"no view directories under {views_root}")

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
            raise SystemExit(
                f"model {args.model!r} is not a vision model; capabilities={capabilities}. "
                "Install/use an Ollama VLM such as 'llava' or 'llama3.2-vision:11b'."
            )

    summary: list[dict[str, Any]] = []
    frame_summary: list[dict[str, Any]] = []
    for view_dir in dirs:
        images = _images_for_dir(view_dir)
        if args.scope in {"scene", "both"}:
            out_json = view_dir / "vlm_layout_eval.json"
            if args.skip_existing and out_json.is_file():
                raw = json.loads(out_json.read_text(encoding="utf-8"))
                summary.append(
                    {
                        "scene_state": view_dir.name,
                        "total_score": _score(raw, "total_score"),
                        "passed": bool(raw.get("passed", False)),
                        "output": str(out_json),
                        "skipped_existing": True,
                    }
                )
                print(f"skip existing scene: {view_dir.name}")
            else:
                if args.image_mode == "contact_sheet":
                    send_images = [
                        _make_contact_sheet(
                            view_dir=view_dir,
                            images=images,
                            cell_width=int(args.contact_cell_width),
                            cell_height=int(args.contact_cell_height),
                        )
                    ]
                else:
                    send_images = images
                user_payload = _payload(view_dir=view_dir, images=images, args=args)
                (view_dir / "vlm_layout_eval_user_payload.json").write_text(user_payload, encoding="utf-8")
                print(f"VLM evaluate scene: {view_dir.name} ({len(images)} frames, mode={args.image_mode})", flush=True)

                t0 = perf_counter()
                try:
                    raw, provider_meta = _evaluate_images(
                        provider=args.provider,
                        ollama_url=args.ollama_url,
                        model=args.model,
                        system_prompt=SYSTEM_PROMPT,
                        user_payload=user_payload,
                        send_images=send_images,
                        timeout_sec=int(args.timeout_sec),
                        temperature=float(args.temperature),
                    )
                except Exception as exc:
                    err = {
                        "scene_state": view_dir.name,
                        "model": args.model,
                        "error": repr(exc),
                        "wall_sec": round(perf_counter() - t0, 3),
                    }
                    (view_dir / "vlm_layout_eval_error.json").write_text(
                        json.dumps(err, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"ERROR: {view_dir.name}: {exc!r}", file=sys.stderr, flush=True)
                    raise SystemExit(2) from exc

                out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                meta = {
                    "scene_state": view_dir.name,
                    "provider": args.provider,
                    "model": provider_meta.get("model") or args.model,
                    "provider_meta": provider_meta,
                    "image_count": len(images),
                    "images": [str(p) for p in images],
                    "sent_image_count": len(send_images),
                    "sent_images": [str(p) for p in send_images],
                    "wall_sec": round(perf_counter() - t0, 3),
                }
                (view_dir / "vlm_layout_eval_meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                row = {
                    "scene_state": view_dir.name,
                    "total_score": _score(raw, "total_score"),
                    "prompt_match_score": _score(raw, "prompt_match_score"),
                    "layout_score": _score(raw, "layout_score"),
                    "collision_score": _score(raw, "collision_score"),
                    "asset_quality_score": _score(raw, "asset_quality_score"),
                    "passed": bool(raw.get("passed", False)),
                    "output": str(out_json),
                    "wall_sec": meta["wall_sec"],
                }
                summary.append(row)
                print(
                    f"OK scene: {view_dir.name}: total={row['total_score']:.1f} "
                    f"layout={row['layout_score']:.1f} collision={row['collision_score']:.1f}",
                    flush=True,
                )

        if args.scope in {"frames", "both"}:
            frames_dir = view_dir / "vlm_frame_evals"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for idx, image in enumerate(images):
                out_json = frames_dir / f"{image.stem}.eval.json"
                if args.skip_existing and out_json.is_file():
                    raw = json.loads(out_json.read_text(encoding="utf-8"))
                    row = {
                        "scene_state": view_dir.name,
                        "frame_index": idx,
                        "frame_name": image.name,
                        "view_type": "topview" if image.name.startswith("topview") else "oblique_e60",
                        "total_score": _score(raw, "total_score"),
                        "prompt_match_score": _score(raw, "prompt_match_score"),
                        "layout_score": _score(raw, "layout_score"),
                        "collision_score": _score(raw, "collision_score"),
                        "asset_quality_score": _score(raw, "asset_quality_score"),
                        "passed": bool(raw.get("passed", False)),
                        "output": str(out_json),
                        "skipped_existing": True,
                    }
                    frame_summary.append(row)
                    print(f"skip existing frame: {view_dir.name}/{image.name}", flush=True)
                    continue
                user_payload = _frame_payload(view_dir=view_dir, image=image, frame_index=idx, args=args)
                (frames_dir / f"{image.stem}.payload.json").write_text(user_payload, encoding="utf-8")
                print(f"VLM evaluate frame: {view_dir.name}/{image.name}", flush=True)
                t0 = perf_counter()
                try:
                    raw, provider_meta = _evaluate_images(
                        provider=args.provider,
                        ollama_url=args.ollama_url,
                        model=args.model,
                        system_prompt=SYSTEM_PROMPT,
                        user_payload=user_payload,
                        send_images=[image],
                        timeout_sec=int(args.timeout_sec),
                        temperature=float(args.temperature),
                    )
                except Exception as exc:
                    err = {
                        "scene_state": view_dir.name,
                        "frame_name": image.name,
                        "model": args.model,
                        "error": repr(exc),
                        "wall_sec": round(perf_counter() - t0, 3),
                    }
                    (frames_dir / f"{image.stem}.error.json").write_text(
                        json.dumps(err, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"ERROR frame: {view_dir.name}/{image.name}: {exc!r}", file=sys.stderr, flush=True)
                    raise SystemExit(2) from exc
                out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                meta = {
                    "scene_state": view_dir.name,
                    "frame_index": idx,
                    "frame_name": image.name,
                    "provider": args.provider,
                    "model": provider_meta.get("model") or args.model,
                    "provider_meta": provider_meta,
                    "image": str(image),
                    "wall_sec": round(perf_counter() - t0, 3),
                }
                (frames_dir / f"{image.stem}.meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                row = {
                    "scene_state": view_dir.name,
                    "frame_index": idx,
                    "frame_name": image.name,
                    "view_type": "topview" if image.name.startswith("topview") else "oblique_e60",
                    "total_score": _score(raw, "total_score"),
                    "prompt_match_score": _score(raw, "prompt_match_score"),
                    "layout_score": _score(raw, "layout_score"),
                    "collision_score": _score(raw, "collision_score"),
                    "asset_quality_score": _score(raw, "asset_quality_score"),
                    "passed": bool(raw.get("passed", False)),
                    "output": str(out_json),
                    "wall_sec": meta["wall_sec"],
                }
                frame_summary.append(row)
                print(
                    f"OK frame: {view_dir.name}/{image.name}: total={row['total_score']:.1f} "
                    f"layout={row['layout_score']:.1f}",
                    flush=True,
                )

    if summary:
        summary_sorted = sorted(summary, key=lambda r: float(r.get("total_score", 0.0)), reverse=True)
        summary_path = views_root / "vlm_layout_eval_summary.json"
        summary_path.write_text(json.dumps(summary_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary: {summary_path}")
    if frame_summary:
        frame_summary_sorted = sorted(
            frame_summary,
            key=lambda r: (str(r.get("scene_state")), int(r.get("frame_index", 0))),
        )
        frame_summary_path = views_root / "vlm_frame_eval_summary.json"
        frame_summary_path.write_text(json.dumps(frame_summary_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"frame summary: {frame_summary_path}")


if __name__ == "__main__":
    main()
