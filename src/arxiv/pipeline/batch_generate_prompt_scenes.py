from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return list(data or [])
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"unsupported prompts file: {path}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch prompt scene generation")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume-failed-only", action="store_true")
    parser.add_argument("--llm-provider", default="none")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="gpt-oss:20b")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    prompts = _load_prompts(Path(args.prompts).expanduser().resolve())
    root = Path(args.out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    prior = {row.get("scene_id"): row for row in manifest}
    updated: list[dict[str, Any]] = []
    for index, row in enumerate(prompts, start=1):
        prompt = str(row.get("prompt") or row.get("text") or "").strip()
        if not prompt:
            continue
        scene_id = str(row.get("scene_id") or f"scene_{index:03d}")
        if args.resume_failed_only and prior.get(scene_id, {}).get("status") == "done":
            updated.append(prior[scene_id])
            continue
        scene_dir = root / scene_id
        cmd = [
            "python3",
            "-m",
            "src.pipeline.generate_prompt_scene",
            "--prompt",
            prompt,
            "--out-dir",
            str(scene_dir),
            "--llm-provider",
            args.llm_provider,
            "--ollama-url",
            args.ollama_url,
            "--ollama-model",
            args.ollama_model,
        ]
        import subprocess

        completed = subprocess.run(cmd, check=False)
        updated.append(
            {
                "scene_id": scene_id,
                "prompt": prompt,
                "out_dir": str(scene_dir),
                "status": "done" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
            }
        )
        manifest_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
