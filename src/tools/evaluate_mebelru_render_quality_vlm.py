#!/usr/bin/env python3
"""Evaluate generated mebel.ru GLB renders against source product photos with a VLM.

The script reads mebel.ru cards from supplier_catalog_canonical.json, downloads
the selected render PNG and source photo from the remote TRELLIS job directory,
builds a side-by-side comparison image, and asks an Ollama vision model for a
structured quality score.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_CATALOG = "data/sourse/suppliers/supplier_catalog_canonical.json"
DEFAULT_OUT_DIR = "reports/mebelru_render_quality_vlm"
DEFAULT_MODEL = "llama3.2-vision:11b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
COMPARISON_SCORE_FIELDS = (
    "category_correctness",
    "visual_similarity",
    "size_proportions",
    "material_correctness",
    "color_match",
    "geometry_quality",
    "completeness",
)
RENDER_SCORE_FIELDS = (
    "render_shape_plausibility",
    "render_proportion_plausibility",
    "render_geometry_integrity",
    "render_material_texture_logic",
    "render_completeness",
    "render_overall_quality",
)
SCORE_FIELDS = COMPARISON_SCORE_FIELDS + RENDER_SCORE_FIELDS
VIEW_ORDER = ("front", "left", "right", "three_quarter")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM quality review for mebel.ru generated render PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--catalog", default=DEFAULT_CATALOG)
    p.add_argument("--jobs-root", default="", help="Scan local TRELLIS job dirs directly instead of reading a catalog.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--source-site", default="mebel.ru")
    p.add_argument("--view", default="three_quarter", help="Preferred render view: three_quarter/front/left/right/all")
    p.add_argument(
        "--product-multiview",
        action="store_true",
        help="When --view all, score all available views of one product in a single VLM/LLM call for consistency.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--unique-key", action="append", default=[], help="Evaluate only this unique_key; can be repeated.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prepare-only", action="store_true", help="Download images and build composites, but do not call VLM.")
    p.add_argument("--merge-catalog", action="store_true", help="Merge completed VLM scores back into the catalog JSON.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--score-model", default="", help="Text LLM for scoring VLM observations. Defaults to --model.")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--timeout-sec", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-predict", type=int, default=450)
    p.add_argument("--score-num-predict", type=int, default=700)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--remote-host", default="84.2.13.196")
    p.add_argument("--remote-port", default="28553")
    p.add_argument("--remote-user", default="root")
    p.add_argument("--ssh-key", default=str(Path.home() / ".ssh/id_ed25519"))
    p.add_argument("--composite-size", type=int, default=768, help="Square panel size for each side of the composite.")
    p.add_argument("--max-reference-rank", type=int, default=3, help="Try image_01.jpg..image_N.jpg on remote.")
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                done.add(str(row.get("evaluation_id") or ""))
    return done


def slug(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_")
    return value[:180] or "item"


def select_items(catalog: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = [x for x in catalog.get("items", []) if isinstance(x, dict)]
    selected = [x for x in items if x.get("source_site") == args.source_site]
    if args.unique_key:
        wanted = set(args.unique_key)
        selected = [x for x in selected if x.get("unique_key") in wanted]
    if args.offset:
        selected = selected[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def load_items_from_jobs_root(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    wanted = set(args.unique_key or [])
    for card_path in sorted(root.glob("*/card.normalized.json")):
        try:
            item = read_json(card_path)
        except Exception as exc:
            print(f"[warn] failed to read {card_path}: {exc!r}", flush=True)
            continue
        if not isinstance(item, dict):
            continue
        if args.source_site and item.get("source_site") != args.source_site:
            continue
        if wanted and item.get("unique_key") not in wanted:
            continue
        job_dir = card_path.parent
        render_paths = sorted(str(p) for p in (job_dir / "renders").glob("*__view_*.png"))
        asset_glb = job_dir / "output" / "asset.trellis.glb"
        trellis_report_path = job_dir / "output" / "trellis.report.json"
        used_image_path = ""
        input_image_paths: list[str] = []
        if trellis_report_path.exists():
            try:
                report = read_json(trellis_report_path)
                if isinstance(report, dict):
                    used_image_path = str(report.get("used_image_path") or "")
                    raw_image_paths = report.get("image_paths")
                    if isinstance(raw_image_paths, list):
                        input_image_paths = [str(x) for x in raw_image_paths if x]
            except Exception as exc:
                print(f"[warn] failed to read {trellis_report_path}: {exc!r}", flush=True)
        item.setdefault("extra", {})
        if not isinstance(item["extra"], dict):
            item["extra"] = {"previous_extra": item["extra"]}
        item["extra"]["trellis_remote"] = {
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "card_normalized_json": str(card_path),
            "trellis_report_json": str(trellis_report_path) if trellis_report_path.exists() else "",
            "used_image_path": used_image_path,
            "input_image_paths": input_image_paths,
            "asset_glb": str(asset_glb) if asset_glb.exists() else "",
            "render_pngs": render_paths,
            "render_png_count": len(render_paths),
        }
        item["generated_asset_status"] = "ready" if asset_glb.exists() else "missing"
        item["generated_asset_format"] = "glb" if asset_glb.exists() else ""
        item["generated_asset_remote_path"] = str(asset_glb) if asset_glb.exists() else ""
        item["generated_render_remote_paths"] = render_paths
        selected.append(item)
    if args.offset:
        selected = selected[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def render_paths_for_item(item: dict[str, Any], view: str) -> list[str]:
    paths = list(item.get("generated_render_remote_paths") or [])
    if not paths:
        trellis = item.get("extra", {}).get("trellis_remote", {})
        paths = list(trellis.get("render_pngs") or [])
    if view == "all":
        return paths
    needle = f"__view_{view}.png"
    preferred = [p for p in paths if needle in p]
    return preferred[:1] or paths[:1]


def view_name_from_render_path(path: str) -> str:
    match = re.search(r"__view_([^.]+)\.png$", path)
    return match.group(1) if match else "unknown"


def reference_candidates(item: dict[str, Any], max_rank: int) -> list[str]:
    trellis = item.get("extra", {}).get("trellis_remote", {})
    job_dir = str(trellis.get("job_dir") or "")
    out: list[str] = []
    used_image = str(trellis.get("used_image_path") or "")
    if used_image:
        out.append(used_image)
    for image_path in trellis.get("input_image_paths") or []:
        image_path = str(image_path)
        if image_path and image_path not in out:
            out.append(image_path)
    if job_dir:
        for rank in range(1, max(1, max_rank) + 1):
            for candidate in (f"{job_dir}/images/image_{rank:02d}.jpg", f"{job_dir}/images/image_{rank:02d}.png"):
                if candidate not in out:
                    out.append(candidate)
    return out


def scp_remote(remote_path: str, local_path: Path, args: argparse.Namespace) -> bool:
    direct = Path(remote_path)
    if direct.exists() and direct.is_file():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if direct.resolve() == local_path.resolve():
            return direct.stat().st_size > 0
        shutil.copy2(direct, local_path)
        return local_path.exists() and local_path.stat().st_size > 0
    if local_path.exists() and local_path.stat().st_size > 0:
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{args.remote_user}@{args.remote_host}:{remote_path}"
    cmd = [
        "scp",
        "-P",
        str(args.remote_port),
        "-i",
        str(args.ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        remote,
        str(local_path),
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        if local_path.exists():
            local_path.unlink()
        return False
    return local_path.exists() and local_path.stat().st_size > 0


def cache_remote_first(candidates: list[str], local_dir: Path, prefix: str, args: argparse.Namespace) -> tuple[Path | None, str]:
    for idx, remote_path in enumerate(candidates, start=1):
        suffix = Path(remote_path).suffix.lower() or ".jpg"
        local_path = local_dir / f"{prefix}_{idx:02d}{suffix}"
        if scp_remote(remote_path, local_path, args):
            return local_path, remote_path
    return None, ""


def cache_remote_exact(remote_path: str, local_dir: Path, filename: str, args: argparse.Namespace) -> Path | None:
    suffix = Path(remote_path).suffix.lower() or ".png"
    local_path = local_dir / f"{filename}{suffix}"
    if scp_remote(remote_path, local_path, args):
        return local_path
    return None


def fit_panel(path: Path, size: int, label: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (size, size - 42), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = 42 + (size - 42 - img.height) // 2
    panel.paste(img, (x, y))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, size, 42), fill=(28, 28, 28))
    draw.text((14, 10), label, fill="white", font=font)
    return panel


def build_composite(reference_path: Path, render_path: Path, out_path: Path, size: int) -> None:
    left = fit_panel(reference_path, size, "REFERENCE PHOTO")
    right = fit_panel(render_path, size, "GENERATED GLB RENDER")
    composite = Image.new("RGB", (size * 2, size), (240, 240, 240))
    composite.paste(left, (0, 0))
    composite.paste(right, (size, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, quality=92)


def build_multiview_composite(reference_path: Path, render_paths: dict[str, Path], out_path: Path, size: int) -> None:
    panel = max(360, min(size, 640))
    labels = [
        ("reference", "REFERENCE PHOTO", reference_path),
        ("front", "RENDER FRONT", render_paths.get("front")),
        ("left", "RENDER LEFT", render_paths.get("left")),
        ("right", "RENDER RIGHT", render_paths.get("right")),
        ("three_quarter", "RENDER THREE QUARTER", render_paths.get("three_quarter")),
    ]
    panels: list[Image.Image] = []
    for _, label, path in labels:
        if path is None:
            blank = Image.new("RGB", (panel, panel), "white")
            draw = ImageDraw.Draw(blank)
            draw.rectangle((0, 0, panel, 42), fill=(28, 28, 28))
            draw.text((14, 10), label, fill="white")
            draw.text((14, 70), "MISSING", fill=(160, 0, 0))
            panels.append(blank)
        else:
            panels.append(fit_panel(path, panel, label))

    composite = Image.new("RGB", (panel * 3, panel * 2), (240, 240, 240))
    positions = [(0, 0), (panel, 0), (panel * 2, 0), (0, panel), (panel, panel)]
    for img, pos in zip(panels, positions):
        composite.paste(img, pos)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, quality=92)


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def prompt_for_item(item: dict[str, Any]) -> str:
    return f"""
You are describing a generated 3D furniture asset against the exact input image used to create it.
The image has two panels: LEFT is the TRELLIS source input image, RIGHT is the generated GLB render.
The target product is the item named in Product metadata, but the LEFT image is the source of truth for
visual similarity, materials, colors, geometry, and completeness.

Product metadata:
- unique_key: {item.get('unique_key')}
- title: {item.get('title')}
- expected category: {item.get('category_norm') or item.get('category_raw')}
- description: {item.get('description')}
- color: {item.get('color')}
- materials: {item.get('materials')}
- dimensions_cm: {json.dumps(item.get('dimensions_cm'), ensure_ascii=False)}

Do not assign numeric scores. Describe visible facts only.
Evaluate only the target furniture object. Ignore the room, floor, wall, lamps, plants, decor, other furniture,
photo crop, lighting setup, camera angle, and perspective differences unless they hide the target object.
The most important question is whether the generated object itself is the same kind of item with the same
main shape, construction, proportions, and visible parts.

Return only valid JSON with detailed observations. Use short but specific English.
Be especially explicit about:
- what is good about the RIGHT render as a standalone 3D asset;
- visible defects: holes, missing surfaces, broken shells, melted parts, floating fragments;
- whether geometry is intact or broken;
- proportions and physical plausibility;
- material/texture logic, including blurry, noisy, torn, or missing texture;
- color differences: distinguish major hue mismatch from small brightness/saturation/lightness differences;
- small details such as handles, legs, seams, cushions, shelves, doors, arms, backrests;
- comparison to LEFT for object identity, shape, materials, colors, and important parts.
{{
  "target_object": "what the target item is",
  "comparison": {{
    "category": "same or different object class",
    "shape": "main shape/silhouette match or mismatch",
    "proportions": "proportion match or mismatch",
    "materials": "material match or mismatch",
    "colors": "major hue match/mismatch; mention if only brightness/saturation/lightness differs",
    "small_details": "handles/legs/seams/cushions/etc.",
    "completeness": "important parts present or missing"
  }},
  "render_quality": {{
    "standalone_strengths": ["visible strength"],
    "standalone_defects": ["visible defect"],
    "geometry_defect_severity": "none | minor | moderate | severe",
    "shape_plausibility": "RIGHT object shape by itself",
    "proportion_plausibility": "RIGHT proportions by itself",
    "geometry_integrity": "specific geometry evidence; say no holes only if no holes are visible",
    "material_texture_logic": "RIGHT material and texture quality",
    "small_detail_quality": "RIGHT small details quality",
    "completeness": "RIGHT required parts present or missing",
    "overall_usability": "usable as 3D asset or not"
  }}
}}
""".strip()


def prompt_for_multiview_item(item: dict[str, Any], views: list[str]) -> str:
    return f"""
You are describing generated 3D furniture renders against the exact input image used to create the asset.
The image is a grid: one REFERENCE PHOTO plus generated GLB renders for these views: {", ".join(views)}.
All render panels are different views of the same generated 3D asset. Judge them consistently.

Product metadata:
- unique_key: {item.get('unique_key')}
- title: {item.get('title')}
- expected category: {item.get('category_norm') or item.get('category_raw')}
- description: {item.get('description')}
- color: {item.get('color')}
- materials: {item.get('materials')}
- dimensions_cm: {json.dumps(item.get('dimensions_cm'), ensure_ascii=False)}

Do not assign numeric scores. Describe visible facts only.
Evaluate only the target furniture object. Ignore the room, floor, wall, lamps, plants, decor, other furniture,
photo crop, lighting setup, camera angle, and perspective differences unless they hide the target object.
Because all render panels are the same asset, do not suddenly call one view a different object class unless that
view truly lacks the required object parts. Focus on the same object identity, shape, proportions, materials,
visible holes/missing surfaces, and whether defects are view-specific or global.
Be soft and precise about color: major hue swaps such as green vs red are important, but dark brown vs lighter
brown, beige vs cream, or small brightness/saturation differences are only moderate differences.

Return only valid JSON with this structure. Use short but specific English.
{{
  "target_object": "what the target item is, using title/category when helpful",
  "global_assessment": "what is consistent across the render views",
  "views": {{
    "front": {{
      "comparison": {{
        "category": "same or different object class",
        "shape": "main shape/silhouette match or mismatch",
        "proportions": "proportion match or mismatch",
        "materials": "material match or mismatch",
        "colors": "major hue match/mismatch; mention if only brightness/saturation/lightness differs",
        "small_details": "handles/legs/seams/cushions/etc.",
        "completeness": "important parts present or missing"
      }},
      "render_quality": {{
        "standalone_strengths": ["visible strength"],
        "standalone_defects": ["visible defect"],
        "geometry_defect_severity": "none | minor | moderate | severe",
        "shape_plausibility": "object shape by itself",
        "proportion_plausibility": "proportions by itself",
        "geometry_integrity": "specific geometry evidence; say no holes only if no holes are visible",
        "material_texture_logic": "material and texture quality",
        "small_detail_quality": "small details quality",
        "completeness": "required parts present or missing",
        "overall_usability": "usable as 3D asset or not"
      }}
    }}
  }}
}}

Include exactly these view keys under "views": {json.dumps(views, ensure_ascii=False)}.
""".strip()


def scoring_prompt_for_observations(item: dict[str, Any], observations: dict[str, Any]) -> str:
    return f"""
You are a strict QA scoring layer for generated 3D assets. Convert VLM visual observations into numeric scores.
Do not inspect images. Use only the observations JSON and this rubric.

Your job is calibration, not politeness. Avoid the default middle range. A generated asset can be recognizable
and still score 1-4 for standalone render quality if the mesh is visibly broken, hollow, melted, or missing surfaces.

Product metadata:
- unique_key: {item.get('unique_key')}
- title: {item.get('title')}
- expected category: {item.get('category_norm') or item.get('category_raw')}

Important: the VLM can be wrong about category comparison. Use the original title and expected category above as
strong context. If the VLM says "wrong object class" but also describes the same kind of object or the image clearly
contains the named category, do not collapse category_correctness/visual_similarity to 1. Penalize actual missing
parts and geometry defects separately.

VLM observations:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Score 1-10. Use the full range. Scores must match the written observations and visible-defect wording.

General scale:
- 9-10: production-ready, clean, accurate, only tiny imperfections.
- 7-8: good/usable, clear object identity, only minor defects. No big holes, broken shells, or missing structural surfaces.
- 5-6: mediocre, recognizable but visibly flawed. Moderate artifacts, weak textures, or partial detail loss.
- 3-4: poor, major defects, incomplete/broken parts, obvious holes, melted areas, or missing surfaces.
- 1-2: unusable, severe broken geometry, many holes, hollow shell, large missing surfaces, or wrong object.

Hard gate rules. These override all softer positive wording:
- If observations conflict, use the worse explicit defect. Example: "severe" plus "usable" still means severe geometry.
- geometry_defect_severity=none -> render_geometry_integrity 8-10.
- geometry_defect_severity=minor -> render_geometry_integrity 6-7.
- geometry_defect_severity=moderate -> render_geometry_integrity 4-5.
- geometry_defect_severity=severe -> render_geometry_integrity 1-3.
- If standalone_defects mention broken shells, floating fragments, severe holes, many holes, hollow interior, torn mesh, or melted/missing surfaces -> render_geometry_integrity 1-3.
- If observations say no holes, no missing surfaces, intact, and severity is none -> render_geometry_integrity 8-10.
- Do not assign high geometry scores when holes or missing surfaces are explicitly observed.
- render_overall_quality should track actual usability: 8-10 only clean production assets, 5-6 for usable but flawed, 1-4 for broken/poor assets.
- If render_geometry_integrity is 1-3, render_overall_quality should usually be 1-4.
- If render_geometry_integrity is 4-5, render_overall_quality should usually be 4-6.
- render_completeness must be 1-4 if required parts are missing/incomplete or large surfaces are absent.
- render_shape_plausibility can be higher than geometry only when the silhouette is good, but broken geometry must still lower render_overall_quality.
- Color scoring: strong hue mismatch is a real problem, e.g. green vs red, blue vs yellow, white vs black -> color_match 1-4.
- Color scoring: same hue/material family but different brightness, saturation, lighting, or slightly warmer/cooler tone -> color_match usually 6-8, not 1-4.
- Color scoring: dark brown vs lighter brown, beige vs cream, grey vs slightly darker grey are moderate differences unless the product color is explicitly contradicted.
- Render-quality scores are standalone RIGHT-render scores. Do not use LEFT except for object type.
- Comparison scores are LEFT-vs-RIGHT target-object similarity scores. Ignore background, room, crop, and camera angle.
- Similar object identity does not imply high render quality. Keep comparison and standalone scores separate.
- Use varied scores when evidence differs. Do not mechanically repeat one score. Repeating mostly 6, 7, or 8 is a calibration failure.

Calibration examples:
- Same wardrobe, intact simple rectangular mesh, blurry wood texture, handles present: comparison may be 7-9, standalone render quality about 6-8.
- Same armchair silhouette, but side/back has large holes or torn/missing upholstery surfaces: comparison shape may be 6-8, but render_geometry_integrity 1-3 and render_overall_quality 2-4.
- Wrong object class: category_correctness 1-3 and visual_similarity usually 1-4.
- Correct object but weak small details/textures with intact mesh: geometry may be 7-9, material/detail scores 4-6.
- Same brown/beige/grey family but slightly darker or lighter than reference: color_match around 6-8 depending on severity.

Comparison criteria order:
0 category_correctness, 1 visual_similarity, 2 size_proportions, 3 material_correctness, 4 color_match, 5 geometry_quality, 6 completeness.
Standalone render-quality order:
0 render_shape_plausibility, 1 render_proportion_plausibility, 2 render_geometry_integrity, 3 render_material_texture_logic, 4 render_completeness, 5 render_overall_quality.

Return only valid JSON keyed by criterion names. Do not use arrays. Every score below is a placeholder;
replace every placeholder with your calibrated score.
{{
  "category_correctness": {{"score": 1, "reason": "specific evidence from observations"}},
  "visual_similarity": {{"score": 1, "reason": "specific evidence from observations"}},
  "size_proportions": {{"score": 1, "reason": "specific evidence from observations"}},
  "material_correctness": {{"score": 1, "reason": "specific evidence from observations"}},
  "color_match": {{"score": 1, "reason": "specific evidence from observations"}},
  "geometry_quality": {{"score": 1, "reason": "specific evidence from observations"}},
  "completeness": {{"score": 1, "reason": "specific evidence from observations"}},
  "render_shape_plausibility": {{"score": 1, "reason": "specific evidence from observations"}},
  "render_proportion_plausibility": {{"score": 1, "reason": "specific evidence from observations"}},
  "render_geometry_integrity": {{"score": 1, "reason": "specific evidence from observations"}},
  "render_material_texture_logic": {{"score": 1, "reason": "specific evidence from observations"}},
  "render_completeness": {{"score": 1, "reason": "specific evidence from observations"}},
  "render_overall_quality": {{"score": 1, "reason": "specific evidence from observations"}}
}}
""".strip()


def scoring_prompt_for_multiview_observations(item: dict[str, Any], observations: dict[str, Any], views: list[str]) -> str:
    return f"""
You are a strict QA scoring layer for generated 3D assets. Convert multi-view VLM observations into numeric scores.
Do not inspect images. Use only the observations JSON, product metadata, and this rubric.

Product metadata:
- unique_key: {item.get('unique_key')}
- title: {item.get('title')}
- expected category: {item.get('category_norm') or item.get('category_raw')}

Important: the VLM can be wrong when comparing categories between views. The title and expected category are strong
context. All render views are the same generated 3D asset. Keep object identity consistent across views unless a view
truly lacks the required object parts. Do not assign 9 to one view and 1 to a very similar adjacent view just because
the VLM wording changed; score actual visible differences such as holes, missing surfaces, wrong proportions, or bad textures.

VLM observations:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Score each view independently, but calibrated consistently across the same product. Use the full 1-10 range.

General scale:
- 9-10: production-ready, clean, accurate, only tiny imperfections.
- 7-8: good/usable, clear object identity, only minor defects. No big holes, broken shells, or missing structural surfaces.
- 5-6: mediocre, recognizable but visibly flawed. Moderate artifacts, weak textures, or partial detail loss.
- 3-4: poor, major defects, incomplete/broken parts, obvious holes, melted areas, or missing surfaces.
- 1-2: unusable, severe broken geometry, many holes, hollow shell, large missing surfaces, or actually wrong object.

Hard gate rules for standalone render quality:
- geometry_defect_severity=none -> render_geometry_integrity 8-10.
- geometry_defect_severity=minor -> render_geometry_integrity 6-7.
- geometry_defect_severity=moderate -> render_geometry_integrity 4-5.
- geometry_defect_severity=severe -> render_geometry_integrity 1-3.
- If defects mention broken shells, floating fragments, severe holes, many holes, hollow interior, torn mesh, or melted/missing surfaces -> render_geometry_integrity 1-3.
- If render_geometry_integrity is 1-3, render_overall_quality should usually be 1-4.
- Similar object identity does not imply high render quality. Keep comparison and standalone scores separate.
- Comparison scores should focus on the target object, not camera angle, background, or crop.
- Color scoring: strong hue mismatch is a real problem, e.g. green vs red, blue vs yellow, white vs black -> color_match 1-4.
- Color scoring: same hue/material family but different brightness, saturation, lighting, or slightly warmer/cooler tone -> color_match usually 6-8.
- Color scoring: dark brown vs lighter brown, beige vs cream, grey vs slightly darker grey are moderate differences unless the product color is explicitly contradicted.

Return only valid JSON. The top-level object must contain "views". Include exactly these view keys:
{json.dumps(views, ensure_ascii=False)}

For every view return all criterion names:
{{
  "views": {{
    "front": {{
      "category_correctness": {{"score": 1, "reason": "specific evidence"}},
      "visual_similarity": {{"score": 1, "reason": "specific evidence"}},
      "size_proportions": {{"score": 1, "reason": "specific evidence"}},
      "material_correctness": {{"score": 1, "reason": "specific evidence"}},
      "color_match": {{"score": 1, "reason": "specific evidence"}},
      "geometry_quality": {{"score": 1, "reason": "specific evidence"}},
      "completeness": {{"score": 1, "reason": "specific evidence"}},
      "render_shape_plausibility": {{"score": 1, "reason": "specific evidence"}},
      "render_proportion_plausibility": {{"score": 1, "reason": "specific evidence"}},
      "render_geometry_integrity": {{"score": 1, "reason": "specific evidence"}},
      "render_material_texture_logic": {{"score": 1, "reason": "specific evidence"}},
      "render_completeness": {{"score": 1, "reason": "specific evidence"}},
      "render_overall_quality": {{"score": 1, "reason": "specific evidence"}}
    }}
  }}
}}
""".strip()


def call_ollama_json(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    image_path: Path | None,
    temperature: float,
    timeout_sec: int,
    num_predict: int,
    normalize: bool = True,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "user",
        "content": prompt,
    }
    if image_path is not None:
        message["images"] = [image_to_base64(image_path)]
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": int(num_predict)},
        "messages": [message],
    }
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ollama HTTP {exc.code}: {detail}") from exc
    data = json.loads(raw)
    content = str((data.get("message") or {}).get("content") or "").replace("\u00ad", "").strip()
    try:
        parsed = json.loads(extract_json(content))
    except Exception as exc:
        parsed = extract_compact_scores_reasons(content)
        if parsed is None:
            raise RuntimeError(f"failed to parse VLM JSON: {exc}; content={content[:1000]!r}") from exc
    return {"parsed": normalize_scores(parsed) if normalize else parsed, "raw_response": data}


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError("no JSON object found")


def extract_compact_scores_reasons(text: str) -> dict[str, Any] | None:
    """Recover compact VLM output when the model emits valid s/r then corrupts the JSON tail."""
    decoder = json.JSONDecoder()

    def raw_array_after(key: str) -> Any:
        match = re.search(rf'"{re.escape(key)}"\s*:', text)
        if not match:
            return None
        pos = match.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        try:
            value, _ = decoder.raw_decode(text[pos:])
            return value if isinstance(value, list) else None
        except Exception:
            return None

    def recover_score_array(key: str, count: int) -> list[int] | None:
        score_match = re.search(rf'"{re.escape(key)}"\s*:\s*\[([^\]]+)', text, flags=re.S)
        if score_match:
            values = [int(x) for x in re.findall(r"\b(?:10|[1-9])\b", score_match.group(1))[:count]]
            return values if len(values) >= count else None
        return None

    def recover_reason_array(key: str, count: int) -> list[str] | None:
        reason_match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*)', text, flags=re.S)
        if reason_match:
            values = [
                bytes(m.group(1), "utf-8").decode("unicode_escape", errors="replace")
                for m in re.finditer(r'"((?:\\.|[^"\\])*)"', reason_match.group(1))
            ][:count]
            return values if len(values) >= count else None
        return None

    comparison_scores = raw_array_after("c_s")
    comparison_reasons = raw_array_after("c_r")
    render_scores = raw_array_after("q_s")
    render_reasons = raw_array_after("q_r")
    if not isinstance(comparison_scores, list):
        comparison_scores = recover_score_array("c_s", len(COMPARISON_SCORE_FIELDS))
    if not isinstance(comparison_reasons, list):
        comparison_reasons = recover_reason_array("c_r", len(COMPARISON_SCORE_FIELDS))
    if not isinstance(render_scores, list):
        render_scores = recover_score_array("q_s", len(RENDER_SCORE_FIELDS))
    if not isinstance(render_reasons, list):
        render_reasons = recover_reason_array("q_r", len(RENDER_SCORE_FIELDS))

    old_scores = raw_array_after("s")
    old_reasons = raw_array_after("r")
    if not isinstance(comparison_scores, list) and isinstance(old_scores, list):
        comparison_scores = old_scores[: len(COMPARISON_SCORE_FIELDS)]
    if not isinstance(comparison_reasons, list) and isinstance(old_reasons, list):
        comparison_reasons = old_reasons[: len(COMPARISON_SCORE_FIELDS)]

    if (
        isinstance(comparison_scores, list)
        and isinstance(comparison_reasons, list)
        and isinstance(render_scores, list)
        and isinstance(render_reasons, list)
        and len(comparison_scores) >= len(COMPARISON_SCORE_FIELDS)
        and len(comparison_reasons) >= len(COMPARISON_SCORE_FIELDS)
        and len(render_scores) >= len(RENDER_SCORE_FIELDS)
        and len(render_reasons) >= len(RENDER_SCORE_FIELDS)
    ):
        return {
            "c_s": comparison_scores[: len(COMPARISON_SCORE_FIELDS)],
            "c_r": comparison_reasons[: len(COMPARISON_SCORE_FIELDS)],
            "q_s": render_scores[: len(RENDER_SCORE_FIELDS)],
            "q_r": render_reasons[: len(RENDER_SCORE_FIELDS)],
        }
    return None


def score_value(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("score")
    try:
        score = int(round(float(value)))
    except Exception:
        return 1
    return max(1, min(10, score))


def normalize_scores(parsed: dict[str, Any]) -> dict[str, Any]:
    out = dict(parsed)
    comparison_scores: list[int] = []
    render_scores: list[int] = []
    compact_scores = out.get("scores") if isinstance(out.get("scores"), dict) else {}
    compact_reasons = out.get("reasons") if isinstance(out.get("reasons"), dict) else {}
    comparison_array_scores = out.get("c_s") if isinstance(out.get("c_s"), list) else out.get("s") if isinstance(out.get("s"), list) else []
    comparison_array_reasons = out.get("c_r") if isinstance(out.get("c_r"), list) else out.get("r") if isinstance(out.get("r"), list) else []
    render_array_scores = out.get("q_s") if isinstance(out.get("q_s"), list) else []
    render_array_reasons = out.get("q_r") if isinstance(out.get("q_r"), list) else []

    def fill_fields(fields: tuple[str, ...], array_scores: list[Any], array_reasons: list[Any], score_bucket: list[int]) -> dict[str, dict[str, Any]]:
        filled: dict[str, dict[str, Any]] = {}
        for idx, field in enumerate(fields):
            if idx < len(array_scores):
                value = array_scores[idx]
            else:
                value = compact_scores.get(field) if field in compact_scores else out.get(field)
            if not isinstance(value, dict):
                value = {"score": score_value(value), "reason": ""}
            value["score"] = score_value(value)
            if idx < len(array_reasons):
                reason = array_reasons[idx]
            else:
                reason = compact_reasons.get(field) if field in compact_reasons else value.get("reason")
            value["reason"] = str(reason or "")[:500]
            out[field] = value
            filled[field] = value
            score_bucket.append(value["score"])
        return filled

    comparison_filled = fill_fields(COMPARISON_SCORE_FIELDS, comparison_array_scores, comparison_array_reasons, comparison_scores)
    render_filled = fill_fields(RENDER_SCORE_FIELDS, render_array_scores, render_array_reasons, render_scores)
    if len(comparison_scores) != len(COMPARISON_SCORE_FIELDS) or len(render_scores) != len(RENDER_SCORE_FIELDS):
        raise ValueError("VLM response did not include complete comparison and standalone render-quality scores")

    out["comparison_scores"] = comparison_filled
    out["render_quality_scores"] = render_filled
    out["comparison_overall_score"] = round(sum(comparison_scores) / len(comparison_scores), 2) if comparison_scores else 0
    out["render_quality_overall_score"] = round(sum(render_scores) / len(render_scores), 2) if render_scores else 0
    all_scores = comparison_scores + render_scores
    out["overall_score"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    out["pass"] = out["comparison_overall_score"] >= 7.0 and out["render_quality_overall_score"] >= 6.0 and min(all_scores or [0]) >= 4
    if not isinstance(out.get("main_failures"), list):
        out["main_failures"] = []
    out["main_failures"] = [str(x)[:300] for x in out["main_failures"][:8]]
    if not out["main_failures"]:
        out["main_failures"] = [
            value["reason"]
            for field in SCORE_FIELDS
            for value in [out[field]]
            if value["score"] <= 4 and value["reason"]
        ][:4]
    out["brief_summary"] = str(out.get("brief_summary") or out.get("overall_reason") or out.get("summary") or "")[:700]
    out.pop("scores", None)
    out.pop("reasons", None)
    for key in ("c_s", "c_r", "q_s", "q_r", "s", "r", "overall_reason", "summary"):
        out.pop(key, None)
    return out


def normalize_multiview_scores(parsed: dict[str, Any], views: list[str]) -> dict[str, dict[str, Any]]:
    raw_views = parsed.get("views") if isinstance(parsed.get("views"), dict) else {}
    normalized: dict[str, dict[str, Any]] = {}
    for view in views:
        raw = raw_views.get(view)
        if not isinstance(raw, dict):
            raw = parsed.get(view) if isinstance(parsed.get(view), dict) else {}
        if not raw:
            raise ValueError(f"multi-view score response missing view {view!r}")
        normalized[view] = normalize_scores(raw)
    return normalized


def observations_text(observations: dict[str, Any]) -> str:
    return json.dumps(observations, ensure_ascii=False).lower()


def has_negated_defect(text: str) -> bool:
    negated_patterns = (
        "no holes",
        "no visible holes",
        "no severe holes",
        "without holes",
        "no missing surfaces",
        "no visible missing surfaces",
        "without missing surfaces",
        "holes are not visible",
        "holes not visible",
        "not present",
        "none visible",
    )
    return any(pattern in text for pattern in negated_patterns)


def apply_geometry_consistency(scores: dict[str, Any], observations: dict[str, Any]) -> dict[str, Any]:
    text = observations_text(observations)
    rq = observations.get("render_quality") if isinstance(observations.get("render_quality"), dict) else {}
    severity = str(rq.get("geometry_defect_severity") or "").strip().lower()
    geom_text = " ".join(
        str(rq.get(k) or "")
        for k in ("geometry_defect_severity", "geometry_integrity", "standalone_defects", "overall_usability")
    ).lower()
    if not geom_text:
        geom_text = text

    negated_defect = has_negated_defect(geom_text)
    severe = severity == "severe" or any(x in geom_text for x in ("severe", "large holes", "many holes", "broken shell", "broken shells", "floating fragment", "floating fragments", "unusable"))
    moderate = severity == "moderate"
    minor = severity == "minor" or any(x in geom_text for x in ("minor", "small", "slight", "some imperfections"))
    intact = severity == "none" or any(x in geom_text for x in ("no holes", "no missing surfaces", "intact", "no severe holes", "mostly intact"))
    broken = not negated_defect and (
        moderate
        or severe
        or any(x in geom_text for x in ("hole", "holes", "missing surface", "missing surfaces", "melted", "broken", "torn", "open shell"))
    )
    missing_parts = any(x in text for x in ("missing important parts", "missing parts", "missing legs", "missing arm", "missing arms", "missing backrest", "missing cushion", "incomplete"))
    not_usable = any(x in text for x in ("not usable", "unusable", "cannot be used", "not suitable as a 3d asset"))

    target: int | None = None
    reason = ""
    if severe:
        target = 2
        reason = "Observation mentions severe broken geometry: holes, missing surfaces, broken shells, or floating fragments."
    elif moderate:
        target = 4
        reason = "Observation classifies geometry defects as moderate."
    elif broken:
        target = 6 if minor else 3
        reason = "Observation mentions visible holes, missing surfaces, melted parts, or broken geometry."
    elif minor:
        target = 6
        reason = "Observation mentions only minor geometry imperfections."
    elif intact:
        target = 8
        reason = "Observation says geometry is intact or has no holes or missing surfaces."

    adjustments = scores.setdefault("score_adjustments", [])
    if target is not None:
        field = scores.get("render_geometry_integrity") or {}
        old = score_value(field.get("score"))
        if (broken or severe) and old > target:
            field["score"] = target
            field["reason"] = reason
            scores["render_geometry_integrity"] = field
            adjustments.append({"field": "render_geometry_integrity", "old_score": old, "new_score": target, "reason": reason})
        elif intact and not broken and old < target:
            field["score"] = target
            field["reason"] = reason
            scores["render_geometry_integrity"] = field
            adjustments.append({"field": "render_geometry_integrity", "old_score": old, "new_score": target, "reason": reason})

    # Keep comparison geometry consistent with visible render geometry, but less aggressively:
    # it is still a comparison criterion, so object similarity can remain useful while broken mesh is penalized.
    if severe or broken:
        field = scores.get("geometry_quality") or {}
        old = score_value(field.get("score"))
        cap = 3 if severe else 5
        if old > cap:
            field["score"] = cap
            field["reason"] = "Visible render geometry defects affect comparison geometry quality."
            scores["geometry_quality"] = field
            adjustments.append({"field": "geometry_quality", "old_score": old, "new_score": cap, "reason": field["reason"]})

    if missing_parts:
        field = scores.get("render_completeness") or {}
        old = score_value(field.get("score"))
        cap = 4 if severe or broken else 6
        if old > cap:
            field["score"] = cap
            field["reason"] = "Observation mentions missing or incomplete required parts."
            scores["render_completeness"] = field
            adjustments.append({"field": "render_completeness", "old_score": old, "new_score": cap, "reason": field["reason"]})

    if severe or not_usable or broken:
        field = scores.get("render_overall_quality") or {}
        old = score_value(field.get("score"))
        cap = 3 if not_usable or severe else 5
        if old > cap:
            field["score"] = cap
            field["reason"] = "Standalone usability is limited by observed geometry defects."
            scores["render_overall_quality"] = field
            adjustments.append({"field": "render_overall_quality", "old_score": old, "new_score": cap, "reason": field["reason"]})

    recompute_score_aggregates(scores)
    return scores


def recompute_score_aggregates(scores: dict[str, Any]) -> None:
    comparison = [score_value((scores.get(field) or {}).get("score")) for field in COMPARISON_SCORE_FIELDS]
    render = [score_value((scores.get(field) or {}).get("score")) for field in RENDER_SCORE_FIELDS]
    scores["comparison_scores"] = {field: scores[field] for field in COMPARISON_SCORE_FIELDS if field in scores}
    scores["render_quality_scores"] = {field: scores[field] for field in RENDER_SCORE_FIELDS if field in scores}
    scores["comparison_overall_score"] = round(sum(comparison) / len(comparison), 2) if comparison else 0
    scores["render_quality_overall_score"] = round(sum(render) / len(render), 2) if render else 0
    all_scores = comparison + render
    scores["overall_score"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    scores["pass"] = scores["comparison_overall_score"] >= 7.0 and scores["render_quality_overall_score"] >= 6.0 and min(all_scores or [0]) >= 4


def merge_results_into_catalog(catalog_path: Path, results_path: Path) -> dict[str, Any]:
    catalog = read_json(catalog_path)
    items = catalog.get("items", [])
    by_key = {x.get("unique_key"): x for x in items if isinstance(x, dict)}
    merged = 0
    latest: dict[str, dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("unique_key"):
                latest[str(row["unique_key"])] = row
    for key, row in latest.items():
        item = by_key.get(key)
        if not item:
            continue
        item["vlm_render_quality"] = {
            "status": "ok",
            "model": row.get("model"),
            "evaluated_at_unix": row.get("evaluated_at_unix"),
            "evaluation_id": row.get("evaluation_id"),
            "view": row.get("view"),
            "composite_path": row.get("composite_path"),
            "scores": row.get("scores"),
        }
        merged += 1
    catalog.setdefault("meta", {})["mebelru_vlm_render_quality_merge"] = {
        "results_path": str(results_path),
        "merged": merged,
        "merged_at_unix": time.time(),
    }
    backup = catalog_path.with_suffix(catalog_path.suffix + f".bak_vlm_render_quality_{int(time.time())}")
    shutil.copy2(catalog_path, backup)
    write_json(catalog_path, catalog)
    return {"merged": merged, "backup": str(backup)}


def write_summary(out_dir: Path, results_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    failures = [r for r in rows if r.get("status") != "ok"]
    score_sums = {field: 0 for field in SCORE_FIELDS}
    for row in ok:
        scores = row.get("scores") or {}
        for field in SCORE_FIELDS:
            score_sums[field] += score_value((scores.get(field) or {}).get("score"))
    summary = {
        "rows": len(rows),
        "ok": len(ok),
        "failed": len(failures),
        "average_scores": {
            field: round(score_sums[field] / len(ok), 3) if ok else 0 for field in SCORE_FIELDS
        },
        "average_overall": round(
            sum(float((r.get("scores") or {}).get("overall_score") or 0) for r in ok) / len(ok),
            3,
        )
        if ok
        else 0,
    }
    write_json(out_dir / "summary.json", summary)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "unique_key",
            "title",
            "view",
            "status",
            "overall_score",
            "comparison_overall_score",
            "render_quality_overall_score",
            "pass",
            *SCORE_FIELDS,
            "brief_summary",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            scores = row.get("scores") or {}
            writer.writerow(
                {
                    "unique_key": row.get("unique_key"),
                    "title": row.get("title"),
                    "view": row.get("view"),
                    "status": row.get("status"),
                    "overall_score": scores.get("overall_score"),
                    "comparison_overall_score": scores.get("comparison_overall_score"),
                    "render_quality_overall_score": scores.get("render_quality_overall_score"),
                    "pass": scores.get("pass"),
                    **{field: (scores.get(field) or {}).get("score") for field in SCORE_FIELDS},
                    "brief_summary": scores.get("brief_summary"),
                }
            )


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog)
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    composites_dir = out_dir / "composites"
    results_path = out_dir / "results.jsonl"
    failures_path = out_dir / "failures.jsonl"

    if args.jobs_root:
        selected = load_items_from_jobs_root(Path(args.jobs_root), args)
        catalog = {"items": selected}
    else:
        catalog = read_json(catalog_path)
        selected = select_items(catalog, args)
    done = load_done(results_path) if args.resume else set()
    source_label = f"jobs_root={args.jobs_root}" if args.jobs_root else f"catalog={catalog_path}"
    print(f"[input] {source_label} selected={len(selected)} model={args.model} view={args.view}", flush=True)

    processed = 0
    for item_idx, item in enumerate(selected, start=1):
        key = str(item.get("unique_key") or "")
        item_slug = slug(key.replace("::", "_"))
        render_paths = render_paths_for_item(item, args.view)
        if not render_paths:
            append_jsonl(failures_path, {"status": "missing_render", "unique_key": key, "title": item.get("title")})
            continue
        ref_path, ref_remote = cache_remote_first(
            reference_candidates(item, args.max_reference_rank),
            cache_dir / item_slug,
            "reference",
            args,
        )
        if ref_path is None:
            append_jsonl(failures_path, {"status": "missing_reference", "unique_key": key, "title": item.get("title")})
            continue

        if args.product_multiview and args.view == "all":
            view_to_remote: dict[str, str] = {}
            for remote_render in render_paths:
                view = view_name_from_render_path(remote_render)
                if view not in view_to_remote:
                    view_to_remote[view] = remote_render
            ordered_views = [view for view in VIEW_ORDER if view in view_to_remote]
            ordered_views.extend(sorted(view for view in view_to_remote if view not in ordered_views))
            todo_views = [view for view in ordered_views if f"{key}::{view}" not in done]
            if not todo_views:
                continue

            local_renders: dict[str, Path] = {}
            missing = False
            for view in ordered_views:
                local_render = cache_remote_exact(view_to_remote[view], cache_dir / item_slug, f"render_{view}", args)
                if local_render is None:
                    append_jsonl(
                        failures_path,
                        {"status": "missing_render_scp", "unique_key": key, "view": view, "render_remote_path": view_to_remote[view]},
                    )
                    missing = True
                    continue
                local_renders[view] = local_render
            if missing and not local_renders:
                continue

            composite_path = composites_dir / f"{item_slug}__multiview.jpg"
            if not composite_path.exists():
                build_multiview_composite(ref_path, local_renders, composite_path, int(args.composite_size))

            if args.prepare_only:
                for view in todo_views:
                    if view not in local_renders:
                        continue
                    append_jsonl(
                        results_path,
                        {
                            "evaluation_id": f"{key}::{view}",
                            "unique_key": key,
                            "external_id": item.get("external_id"),
                            "title": item.get("title"),
                            "category_norm": item.get("category_norm"),
                            "category_raw": item.get("category_raw"),
                            "view": view,
                            "reference_remote_path": ref_remote,
                            "render_remote_path": view_to_remote[view],
                            "reference_local_path": str(ref_path),
                            "render_local_path": str(local_renders[view]),
                            "composite_path": str(composite_path),
                            "model": args.model,
                            "status": "prepared",
                            "evaluated_at_unix": time.time(),
                        },
                    )
                    processed += 1
                    print(f"[prepared] {item_idx}/{len(selected)} {key}::{view}", flush=True)
                continue

            try:
                last_exc: Exception | None = None
                observation_response: dict[str, Any] | None = None
                score_response: dict[str, Any] | None = None
                normalized_by_view: dict[str, dict[str, Any]] | None = None
                for attempt in range(1, max(1, int(args.max_attempts)) + 1):
                    try:
                        observation_response = call_ollama_json(
                            ollama_url=args.ollama_url,
                            model=args.model,
                            prompt=prompt_for_multiview_item(item, ordered_views),
                            image_path=composite_path,
                            temperature=float(args.temperature),
                            timeout_sec=int(args.timeout_sec),
                            num_predict=int(args.num_predict),
                            normalize=False,
                        )
                        score_response = call_ollama_json(
                            ollama_url=args.ollama_url,
                            model=args.score_model or args.model,
                            prompt=scoring_prompt_for_multiview_observations(item, observation_response["parsed"], ordered_views),
                            image_path=None,
                            temperature=float(args.temperature),
                            timeout_sec=int(args.timeout_sec),
                            num_predict=int(args.score_num_predict),
                            normalize=False,
                        )
                        normalized_by_view = normalize_multiview_scores(score_response["parsed"], ordered_views)
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt >= max(1, int(args.max_attempts)):
                            raise
                        time.sleep(1.0)
                if observation_response is None or score_response is None or normalized_by_view is None:
                    raise RuntimeError(f"no multi-view VLM response: {last_exc!r}")

                parsed_observations = observation_response["parsed"]
                observation_views = parsed_observations.get("views") if isinstance(parsed_observations.get("views"), dict) else {}
                for view in todo_views:
                    if view not in local_renders:
                        continue
                    evaluation_id = f"{key}::{view}"
                    view_observations = observation_views.get(view) if isinstance(observation_views.get(view), dict) else parsed_observations
                    append_jsonl(
                        results_path,
                        {
                            "evaluation_id": evaluation_id,
                            "unique_key": key,
                            "external_id": item.get("external_id"),
                            "title": item.get("title"),
                            "category_norm": item.get("category_norm"),
                            "category_raw": item.get("category_raw"),
                            "view": view,
                            "reference_remote_path": ref_remote,
                            "render_remote_path": view_to_remote[view],
                            "reference_local_path": str(ref_path),
                            "render_local_path": str(local_renders[view]),
                            "composite_path": str(composite_path),
                            "model": args.model,
                            "status": "ok",
                            "vlm_observations": view_observations,
                            "vlm_observations_multiview": parsed_observations,
                            "scores": normalized_by_view[view],
                            "raw_vlm_response": observation_response["raw_response"],
                            "raw_score_response": score_response["raw_response"],
                            "evaluated_at_unix": time.time(),
                        },
                    )
                    done.add(evaluation_id)
                    processed += 1
                    print(
                        f"[ok] {item_idx}/{len(selected)} {evaluation_id} overall={normalized_by_view[view].get('overall_score')}",
                        flush=True,
                    )
            except Exception as exc:
                append_jsonl(
                    failures_path,
                    {
                        "unique_key": key,
                        "title": item.get("title"),
                        "status": "multiview_vlm_error",
                        "error": repr(exc),
                        "composite_path": str(composite_path),
                        "evaluated_at_unix": time.time(),
                    },
                )
                print(f"[fail] {item_idx}/{len(selected)} {key} multiview {exc!r}", flush=True)
            continue

        for remote_render in render_paths:
            view = view_name_from_render_path(remote_render)
            evaluation_id = f"{key}::{view}"
            if evaluation_id in done:
                continue
            local_render = cache_remote_exact(remote_render, cache_dir / item_slug, f"render_{view}", args)
            if local_render is None:
                append_jsonl(failures_path, {"status": "missing_render_scp", "unique_key": key, "render_remote_path": remote_render})
                continue
            composite_path = composites_dir / f"{item_slug}__{view}.jpg"
            if not composite_path.exists():
                build_composite(ref_path, local_render, composite_path, int(args.composite_size))

            row_base = {
                "evaluation_id": evaluation_id,
                "unique_key": key,
                "external_id": item.get("external_id"),
                "title": item.get("title"),
                "category_norm": item.get("category_norm"),
                "category_raw": item.get("category_raw"),
                "view": view,
                "reference_remote_path": ref_remote,
                "render_remote_path": remote_render,
                "reference_local_path": str(ref_path),
                "render_local_path": str(local_render),
                "composite_path": str(composite_path),
                "model": args.model,
            }
            if args.prepare_only:
                append_jsonl(results_path, {**row_base, "status": "prepared", "evaluated_at_unix": time.time()})
                processed += 1
                print(f"[prepared] {item_idx}/{len(selected)} {evaluation_id}", flush=True)
                continue
            try:
                last_exc: Exception | None = None
                observation_response: dict[str, Any] | None = None
                score_response: dict[str, Any] | None = None
                for attempt in range(1, max(1, int(args.max_attempts)) + 1):
                    try:
                        observation_response = call_ollama_json(
                            ollama_url=args.ollama_url,
                            model=args.model,
                            prompt=prompt_for_item(item),
                            image_path=composite_path,
                            temperature=float(args.temperature),
                            timeout_sec=int(args.timeout_sec),
                            num_predict=int(args.num_predict),
                            normalize=False,
                        )
                        score_response = call_ollama_json(
                            ollama_url=args.ollama_url,
                            model=args.score_model or args.model,
                            prompt=scoring_prompt_for_observations(item, observation_response["parsed"]),
                            image_path=None,
                            temperature=float(args.temperature),
                            timeout_sec=int(args.timeout_sec),
                            num_predict=int(args.score_num_predict),
                            normalize=True,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt >= max(1, int(args.max_attempts)):
                            raise
                        time.sleep(1.0)
                if observation_response is None or score_response is None:
                    raise RuntimeError(f"no VLM response: {last_exc!r}")
                append_jsonl(
                    results_path,
                    {
                        **row_base,
                        "status": "ok",
                        "vlm_observations": observation_response["parsed"],
                        "scores": score_response["parsed"],
                        "raw_vlm_response": observation_response["raw_response"],
                        "raw_score_response": score_response["raw_response"],
                        "evaluated_at_unix": time.time(),
                    },
                )
                done.add(evaluation_id)
                processed += 1
                print(
                    f"[ok] {item_idx}/{len(selected)} {evaluation_id} overall={score_response['parsed'].get('overall_score')}",
                    flush=True,
                )
            except Exception as exc:
                append_jsonl(failures_path, {**row_base, "status": "vlm_error", "error": repr(exc), "evaluated_at_unix": time.time()})
                print(f"[fail] {item_idx}/{len(selected)} {evaluation_id} {exc!r}", flush=True)

    write_summary(out_dir, results_path)
    if args.merge_catalog and not args.prepare_only:
        merge_info = merge_results_into_catalog(catalog_path, results_path)
        print(f"[merge] {merge_info}", flush=True)
    print(f"[done] processed={processed} results={results_path} failures={failures_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
