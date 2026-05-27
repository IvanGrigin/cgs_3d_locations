#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="src/arxiv/archive_manifest_unused_run_pipeline_infinigen_procedural_20260526_130635.json"
python3 - "$ROOT/$MANIFEST" "$ROOT" <<'PY_RESTORE'
import json, shutil, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
errors = []
restored = 0
already_restored = 0
for entry in manifest.get("entries", []):
    src = root / entry["archived_path"]
    dst = root / entry["original_path"]
    if not src.is_file():
        if dst.is_file():
            already_restored += 1
            continue
        errors.append(f"missing archived file: {entry['archived_path']}")
        continue
    if dst.exists():
        errors.append(f"restore destination exists: {entry['original_path']}")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    restored += 1
if errors:
    raise SystemExit("restore failed:\n" + "\n".join(errors[:100]))
print(f"restored {restored} files; already_restored={already_restored}")
PY_RESTORE
