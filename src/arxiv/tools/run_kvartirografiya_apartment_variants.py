#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data/input/example/prompts"
DEFAULT_OUT_DIR = REPO_ROOT / "data/output/kvartirografiya_apartment_variants"

ROOM_TYPE_ALIASES = {
    "bedroom": "bedroom",
    "living_room": "living_room",
    "living": "living_room",
    "studio": "living_room",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "toilet": "toilet",
    "joint_bathroom": "bathroom",
    "hall": "hallway",
    "hallway": "hallway",
    "corridor": "hallway",
}

VARIANT_PROFILES = {
    "cheapest": {
        "selection_mode": "cheapest",
        "selection_strategy": "cheapest",
        "prompt_suffix": "Use a cost-conscious furniture and material set while keeping the room functional and coherent.",
    },
    "optimal": {
        "selection_mode": "optimal",
        "selection_strategy": "balanced",
        "prompt_suffix": "Use a balanced furniture and material set with practical circulation and durable everyday items.",
    },
    "best_match": {
        "selection_mode": "best_match",
        "selection_strategy": "style",
        "prompt_suffix": "Prioritize the best style match and visual consistency while keeping all required room functions.",
    },
}

FALLBACK_PROMPTS = {
    "hallway": "Design a compact modern hallway. Keep the route from the entrance to all room doors clear. Required items: wardrobe, shoe cabinet, mirror.",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_apartment_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("apt_"):
        suffix = text.removeprefix("apt_")
    else:
        suffix = text
    if suffix.isdigit():
        return f"apt_{int(suffix):04d}"
    return text


def normalize_room_type(raw: str | None) -> str:
    text = str(raw or "").strip().lower()
    return ROOM_TYPE_ALIASES.get(text, text or "living_room")


def room_size_class(area_m2: float | None) -> str:
    if area_m2 is None:
        return "medium"
    if area_m2 <= 8.0:
        return "small"
    if area_m2 <= 18.0:
        return "medium"
    return "large"


def room_payload(room_json: Path) -> dict[str, Any]:
    data = read_json(room_json)
    room = data.get("room") if isinstance(data, dict) else {}
    if not isinstance(room, dict):
        room = {}
    return room


def choose_prompt(room_type: str, area_m2: float | None, prompts_dir: Path) -> tuple[str, str | None, str]:
    normalized = normalize_room_type(room_type)
    if normalized in FALLBACK_PROMPTS:
        return FALLBACK_PROMPTS[normalized], None, "fallback_builtin"

    size = room_size_class(area_m2)
    candidates = sorted(prompts_dir.glob(f"{normalized}_{size}_*.prompt.txt"))
    if not candidates:
        candidates = sorted(prompts_dir.glob(f"{normalized}_*.prompt.txt"))
    if not candidates and normalized == "bathroom":
        candidates = sorted(prompts_dir.glob("toilet_*.prompt.txt"))
    if not candidates:
        candidates = sorted(prompts_dir.glob("living_room_*.prompt.txt"))

    if not candidates:
        return "Design a modern functional room using the provided real room geometry.", None, "fallback_builtin"

    path = candidates[0]
    return path.read_text(encoding="utf-8").strip(), str(path.resolve()), "prompt_file"


def prompt_room_type(room: dict[str, Any], room_entry: dict[str, Any], normalized_room_type: str) -> str:
    source = normalize_room_type(str(room.get("source_room_type") or room_entry.get("source_room_type") or ""))
    if source in {"toilet", "wc", "restroom"}:
        return "toilet"
    return normalized_room_type


def find_bundle(input_path: Path, apartment: str | None) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    wanted = normalize_apartment_id(apartment)
    if (input_path / "manifest.json").is_file() and (input_path / "rooms").is_dir():
        manifest = read_json(input_path / "manifest.json")
        found = normalize_apartment_id(str(manifest.get("apartment_id", "")))
        if wanted is None or wanted == found or wanted == input_path.name:
            return input_path

    if wanted is None:
        raise ValueError("--apartment is required unless --input-path points directly to an apartment bundle")

    direct_matches = [
        input_path / wanted,
        input_path / "apartment_bundles" / wanted,
    ]
    for candidate in direct_matches:
        if (candidate / "manifest.json").is_file() and (candidate / "rooms").is_dir():
            return candidate.resolve()

    index_path = input_path / "index.json"
    if index_path.is_file():
        index = read_json(index_path)
        for manifest_ref in index.get("floor_manifests", []) if isinstance(index, dict) else []:
            floor_manifest_path = Path(manifest_ref)
            if not floor_manifest_path.is_absolute():
                floor_manifest_path = (input_path / floor_manifest_path).resolve()
            if not floor_manifest_path.is_file():
                continue
            floor_manifest = read_json(floor_manifest_path)
            for apt in floor_manifest.get("apartments", []):
                if not isinstance(apt, dict):
                    continue
                apt_id = normalize_apartment_id(str(apt.get("apartment_id", "")))
                if apt_id == wanted:
                    bundle_ref = apt.get("bundle_manifest")
                    if bundle_ref:
                        bundle_manifest = Path(bundle_ref)
                        if not bundle_manifest.is_absolute():
                            bundle_manifest = (floor_manifest_path.parent / bundle_manifest).resolve()
                        return bundle_manifest.parent.resolve()

    for manifest_path in input_path.rglob("apartment_bundles/*/manifest.json"):
        bundle = manifest_path.parent
        if normalize_apartment_id(bundle.name) == wanted:
            return bundle.resolve()
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if normalize_apartment_id(str(manifest.get("apartment_id", ""))) == wanted:
            return bundle.resolve()

    raise FileNotFoundError(f"Apartment bundle {wanted} was not found under {input_path}")


def shell_command(args: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in args)


def collect_supplier_summary(run_dir: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(run_dir.glob("supplier_replacements*.summary.json")):
        try:
            data = read_json(summary_path)
        except Exception as exc:
            summaries.append({"summary_json": str(summary_path), "error": str(exc)})
            continue
        if isinstance(data, dict):
            data = dict(data)
            data["summary_json"] = str(summary_path.resolve())
            stem = summary_path.name.removesuffix(".summary.json")
            data["short_md"] = str((summary_path.parent / f"{stem}.short.md").resolve())
            data["full_md"] = str((summary_path.parent / f"{stem}.full.md").resolve())
            data["html"] = str((summary_path.parent / f"{stem}.html").resolve())
            data["selected_items"] = selected_supplier_items(data)
            summaries.append(data)
    return {
        "run_dir": str(run_dir.resolve()),
        "summary_count": len(summaries),
        "summaries": summaries,
    }


def selected_supplier_items(summary: dict[str, Any]) -> list[dict[str, Any]]:
    bindings_path = summary.get("bindings_path")
    if not bindings_path:
        return []
    try:
        bindings_data = read_json(Path(str(bindings_path)))
    except Exception:
        return []
    bindings = bindings_data.get("bindings") if isinstance(bindings_data, dict) else []
    if not isinstance(bindings, list):
        return []

    items: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        status = str(binding.get("selection_status") or "").strip()
        if status not in {"selected", "selected_with_real_asset", "selected_without_real_asset"}:
            continue
        candidate = binding.get("chosen_candidate")
        if not isinstance(candidate, dict):
            candidates = [x for x in binding.get("top_candidates") or [] if isinstance(x, dict)]
            candidate_id = str(binding.get("chosen_candidate_id") or "").strip()
            candidate = next((x for x in candidates if str(x.get("candidate_id") or x.get("id") or "") == candidate_id), {})
        if not isinstance(candidate, dict) or not candidate:
            continue
        product_url = ""
        for key in ("product_url", "model_page_url", "model_vendor_url", "source_url", "model_download_landing_url"):
            if str(candidate.get(key) or "").strip():
                product_url = str(candidate.get(key)).strip()
                break
        items.append(
            {
                "target_id": binding.get("target_id"),
                "category": binding.get("category"),
                "title": candidate.get("title") or candidate.get("name") or candidate.get("model_name") or candidate.get("candidate_id"),
                "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
                "source_site": candidate.get("source_site"),
                "price_value": candidate.get("price_value"),
                "price_currency": candidate.get("price_currency") or "RUB",
                "product_url": product_url,
            }
        )
    return items

def estimated_items_total(items: list[dict[str, Any]]) -> tuple[float | None, str]:
    total = 0.0
    currency = "RUB"
    found = False
    for item in items:
        value = item.get("price_value")
        if value is None or value == "":
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
        currency = str(item.get("price_currency") or currency)
        found = True
    return (total if found else None), currency


def build_report(plan: dict[str, Any], executed: bool) -> str:
    lines = [
        f"# Kvartirografiya apartment variants: {plan['apartment_id']}",
        "",
        f"- Bundle: `{plan['bundle_dir']}`",
        f"- Rooms: {len(plan['rooms'])}",
        f"- Variants per room: {', '.join(plan['variant_modes'])}",
        f"- Executed: {'yes' if executed else 'no'}",
        "",
    ]
    for room in plan["rooms"]:
        lines.extend(
            [
                f"## {room['room_id']} ({room['room_type']})",
                f"- Area: {room.get('area_m2') if room.get('area_m2') is not None else 'unknown'} m2",
                f"- Prompt source: {room['prompt_source']}",
                f"- Prompt file: `{room['prompt_file']}`" if room.get("prompt_file") else "- Prompt file: builtin fallback",
            ]
        )
        for run in room["runs"]:
            lines.append(f"- {run['variant_mode']}: `{run['run_dir']}`")
            if executed:
                summary = run.get("supplier_summary", {})
                lines.append(f"  - Supplier summaries: {summary.get('summary_count', 0)}")
                for item in summary.get("summaries", [])[:3]:
                    selected_items = item.get("selected_items") or []
                    total, currency = estimated_items_total(selected_items)
                    if total is not None:
                        lines.append(f"  - Estimated item total: {total:.0f} {currency}")
                    if item.get("summary_json"):
                        lines.append(f"  - Summary JSON: `{item['summary_json']}`")
                    if item.get("full_md"):
                        lines.append(f"  - Full cost/link report: `{item['full_md']}`")
                    for selected in selected_items[:10]:
                        title = str(selected.get("title") or selected.get("candidate_id") or "item")
                        price = selected.get("price_value")
                        currency = selected.get("price_currency") or ""
                        url = selected.get("product_url") or ""
                        price_text = f", {price} {currency}".rstrip() if price is not None else ""
                        if url:
                            lines.append(f"  - [{title}]({url}){price_text}")
                        else:
                            lines.append(f"  - {title}{price_text}")
            else:
                lines.append(f"  - Command: `{run['command']}`")
        lines.append("")
    return "\n".join(lines)


def resolve_bundle_with_optional_adapter(args: argparse.Namespace, out_dir: Path) -> Path:
    input_path = Path(args.input_path).expanduser().resolve()
    try:
        return find_bundle(input_path, args.apartment)
    except (FileNotFoundError, ValueError):
        if not args.auto_adapt:
            raise

    adapter_out_dir = Path(args.adapter_out_dir).expanduser().resolve() if args.adapter_out_dir else out_dir / "_adapter"
    cmd = [
        sys.executable,
        "src/tools/kvartirografiya_adapter.py",
        "--input-dir",
        str(input_path),
        "--out-dir",
        str(adapter_out_dir),
    ]
    print("[adapter] " + shell_command(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return find_bundle(adapter_out_dir, args.apartment)


def build_plan(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    bundle_dir = resolve_bundle_with_optional_adapter(args, out_dir)
    manifest = read_json(bundle_dir / "manifest.json")
    apt_id = normalize_apartment_id(str(manifest.get("apartment_id") or bundle_dir.name)) or bundle_dir.name

    run_root = out_dir / apt_id
    prompts_out = run_root / "_prompts"
    prompts_out.mkdir(parents=True, exist_ok=True)

    variant_modes = [mode.strip() for mode in str(args.variant_modes).replace(";", ",").split(",") if mode.strip()]
    variant_modes = [mode for mode in variant_modes if mode in VARIANT_PROFILES]
    if not variant_modes:
        raise ValueError("--variant-modes must include at least one of: cheapest, optimal, best_match")

    plan: dict[str, Any] = {
        "input_path": str(Path(args.input_path).expanduser().resolve()),
        "bundle_dir": str(bundle_dir),
        "apartment_id": apt_id,
        "apartment_json": manifest.get("apartment_json"),
        "source_manifest": str((bundle_dir / "manifest.json").resolve()),
        "variant_modes": variant_modes,
        "with_infinigen": bool(args.with_infinigen),
        "rooms": [],
    }
    pipeline_placer = args.placer or ("infinigen_clean" if args.with_infinigen else None)

    for room_entry in manifest.get("rooms", []):
        if not isinstance(room_entry, dict):
            continue
        room_json = Path(str(room_entry.get("room_json", ""))).expanduser()
        if not room_json.is_absolute():
            room_json = (bundle_dir / room_json).resolve()
        if not room_json.is_file():
            continue

        room = room_payload(room_json)
        room_id = str(room.get("id") or room_entry.get("room_id") or room_json.stem)
        raw_type = str(room.get("room_type") or room.get("type") or room_entry.get("room_type") or "")
        room_type = normalize_room_type(raw_type)
        area_m2 = room.get("area_m2")
        try:
            area_float = float(area_m2) if area_m2 is not None else None
        except (TypeError, ValueError):
            area_float = None

        prompt_type = prompt_room_type(room, room_entry, room_type)
        base_prompt, prompt_file, prompt_source = choose_prompt(prompt_type, area_float, Path(args.prompts_dir).expanduser().resolve())
        room_plan: dict[str, Any] = {
            "room_id": room_id,
            "room_type": room_type,
            "source_room_type": room.get("source_room_type") or room_entry.get("source_room_type"),
            "prompt_room_type": prompt_type,
            "room_json": str(room_json.resolve()),
            "area_m2": area_float,
            "prompt_source": prompt_source,
            "prompt_file": prompt_file,
            "runs": [],
        }

        for mode in variant_modes:
            profile = VARIANT_PROFILES[mode]
            run_dir = run_root / room_id / mode
            prompt_out = prompts_out / f"{room_id}.{mode}.prompt.txt"
            prompt_text = "\n\n".join([base_prompt, profile["prompt_suffix"]]).strip() + "\n"
            prompt_out.write_text(prompt_text, encoding="utf-8")

            cmd = [
                sys.executable,
                "src/run_pipeline.py",
                "--room",
                str(room_json.resolve()),
                "--prompt-file",
                str(prompt_out.resolve()),
                "--run-dir",
                str(run_dir.resolve()),
                "--keep-tmp",
                "--supplier-selection-mode",
                profile["selection_mode"],
                "--supplier-selection-strategy",
                profile["selection_strategy"],
                "--supplier-top-k",
                str(args.supplier_top_k),
                "--supplier-llm-provider",
                args.supplier_llm_provider,
            ]
            if pipeline_placer:
                cmd.extend(["--placer", pipeline_placer])
            if args.modes:
                cmd.extend(["--modes", args.modes])
            if args.infinigen_fast_small:
                cmd.append("--infinigen-fast-small")
            if args.infinigen_no_pose_cameras:
                cmd.append("--infinigen-no-pose-cameras")
            if args.infinigen_solve_steps_large is not None:
                cmd.extend(["--infinigen-solve-steps-large", str(args.infinigen_solve_steps_large)])
            if args.infinigen_solve_steps_medium is not None:
                cmd.extend(["--infinigen-solve-steps-medium", str(args.infinigen_solve_steps_medium)])
            if args.infinigen_solve_steps_small is not None:
                cmd.extend(["--infinigen-solve-steps-small", str(args.infinigen_solve_steps_small)])
            if args.skip_blender:
                cmd.append("--skip-blender")
            else:
                cmd.extend(["--blender-output", args.blender_output])
                cmd.append("--build-supplier-blend")
                if args.keep_blend:
                    cmd.append("--keep-blend")
                if args.headless:
                    cmd.append("--headless")
                if args.blender:
                    cmd.extend(["--blender", args.blender])
                cmd.extend(["--blender-gif-frames", str(args.blender_gif_frames)])
                cmd.extend(["--supplier-gif-frames", str(args.supplier_gif_frames)])
                if args.skip_supplier_gif:
                    cmd.append("--skip-supplier-gif")
            if args.heuristic_llm_stages:
                cmd.extend(
                    [
                        "--chooser-llm-provider",
                        "none",
                        "--style-llm-provider",
                        "none",
                        "--flooring-llm-provider",
                        "none",
                        "--wall-llm-provider",
                        "none",
                    ]
                )
            if args.no_flooring:
                cmd.append("--no-flooring")
            if args.no_wall_material:
                cmd.append("--no-wall-material")
            for catalog in args.supplier_catalog_json or []:
                cmd.extend(["--supplier-catalog-json", catalog])

            room_plan["runs"].append(
                {
                    "variant_mode": mode,
                    "run_dir": str(run_dir.resolve()),
                    "prompt_file": str(prompt_out.resolve()),
                    "command_args": cmd,
                    "command": shell_command(cmd),
                }
            )
        plan["rooms"].append(room_plan)

    if not plan["rooms"]:
        raise RuntimeError(f"No room JSON files found in apartment bundle: {bundle_dir}")

    return run_root, plan


def execute_plan(plan: dict[str, Any]) -> int:
    failures = 0
    for room in plan["rooms"]:
        for run in room["runs"]:
            cmd = list(run["command_args"])
            print(f"[run] {room['room_id']} / {run['variant_mode']}", flush=True)
            completed = subprocess.run(cmd, cwd=REPO_ROOT)
            run["returncode"] = completed.returncode
            run["supplier_summary"] = collect_supplier_summary(Path(run["run_dir"]))
            if completed.returncode != 0:
                failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run supplier variants for every room in a Kvartirografiya apartment bundle.")
    parser.add_argument("--input-path", required=True, help="Adapter output root, floor dir, apartment_bundles dir, or direct apt_XXXX bundle dir.")
    parser.add_argument("--apartment", default=None, help="Apartment number/id, e.g. 1, 0001, apt_0001. Optional for direct bundle input.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for run plan, prompts, reports, and pipeline run dirs.")
    parser.add_argument("--adapter-out-dir", default=None, help="Where to write auto-adapted Kvartirografiya bundles when input is a raw source folder.")
    parser.add_argument("--auto-adapt", action=argparse.BooleanOptionalAction, default=True, help="Run kvartirografiya_adapter.py automatically if no apartment bundle is found.")
    parser.add_argument("--prompts-dir", default=str(DEFAULT_PROMPTS_DIR), help="Directory with default room prompts.")
    parser.add_argument("--variant-modes", default="cheapest,optimal,best_match", help="Comma-separated supplier variants.")
    parser.add_argument("--supplier-top-k", type=int, default=5)
    parser.add_argument("--supplier-catalog-json", action="append", default=[])
    parser.add_argument("--supplier-llm-provider", choices=["none", "ollama"], default="none")
    parser.add_argument("--with-infinigen", action="store_true", help="Generate room placement with Infinigen by passing --placer infinigen_clean to run_pipeline.py unless --placer is set explicitly.")
    parser.add_argument("--placer", default=None, help="run_pipeline.py placer, e.g. infinigen_clean")
    parser.add_argument("--modes", default=None, help="Optional run_pipeline.py --modes override")
    parser.add_argument("--infinigen-fast-small", action="store_true", help="Pass --infinigen-fast-small to run_pipeline.py")
    parser.add_argument("--infinigen-no-pose-cameras", action="store_true", help="Pass --infinigen-no-pose-cameras to run_pipeline.py")
    parser.add_argument("--infinigen-solve-steps-large", type=int, default=None)
    parser.add_argument("--infinigen-solve-steps-medium", type=int, default=None)
    parser.add_argument("--infinigen-solve-steps-small", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Run src/run_pipeline.py for every room/variant. Without this only writes the plan.")
    parser.add_argument("--skip-blender", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blender-output", choices=["render", "gif", "both"], default="both")
    parser.add_argument("--keep-blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--blender-gif-frames", type=int, default=36)
    parser.add_argument("--supplier-gif-frames", type=int, default=36)
    parser.add_argument("--skip-supplier-gif", action="store_true")
    parser.add_argument("--heuristic-llm-stages", action=argparse.BooleanOptionalAction, default=True, help="Disable chooser/style/floor/wall Ollama stages for reproducible batch runs.")
    parser.add_argument("--no-flooring", action="store_true")
    parser.add_argument("--no-wall-material", action="store_true")
    args = parser.parse_args()

    run_root, plan = build_plan(args)
    failures = execute_plan(plan) if args.execute else 0

    write_json(run_root / "run_plan.json", plan)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(str(REPO_ROOT))]
    for room in plan["rooms"]:
        for run in room["runs"]:
            commands.append(run["command"])
    (run_root / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (run_root / "report.md").write_text(build_report(plan, args.execute), encoding="utf-8")

    print(f"Apartment bundle: {plan['bundle_dir']}")
    print(f"Rooms: {len(plan['rooms'])}")
    print(f"Variants per room: {', '.join(plan['variant_modes'])}")
    print(f"Plan: {run_root / 'run_plan.json'}")
    print(f"Commands: {run_root / 'commands.sh'}")
    print(f"Report: {run_root / 'report.md'}")
    if failures:
        print(f"Failed runs: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
