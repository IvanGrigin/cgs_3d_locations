#!/usr/bin/env zsh
set -euo pipefail

ROOT=""
MODE="optimal"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
CORNER_WIDTH="960"
CORNER_HEIGHT="720"
CORNER_SAMPLES="16"
OVERVIEW_WIDTH="1400"
OVERVIEW_HEIGHT="1000"
OVERVIEW_SAMPLES="16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --blender)
      BLENDER="$2"
      shift 2
      ;;
    --corner-width)
      CORNER_WIDTH="$2"
      shift 2
      ;;
    --corner-height)
      CORNER_HEIGHT="$2"
      shift 2
      ;;
    --corner-samples)
      CORNER_SAMPLES="$2"
      shift 2
      ;;
    --overview-width)
      OVERVIEW_WIDTH="$2"
      shift 2
      ;;
    --overview-height)
      OVERVIEW_HEIGHT="$2"
      shift 2
      ;;
    --overview-samples)
      OVERVIEW_SAMPLES="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [[ -n "$ROOT" ]]; then
        echo "Only one root argument is supported" >&2
        exit 2
      fi
      ROOT="$1"
      shift
      ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  echo "Usage: $0 <apartment-or-project-root> [--mode optimal] [--blender /path/to/Blender]" >&2
  exit 2
fi

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_ABS="$(cd "$ROOT" && pwd)"

if [[ ! -x "$BLENDER" ]]; then
  echo "Blender executable not found or not executable: $BLENDER" >&2
  exit 2
fi

typeset -a APT_DIRS=()
if [[ -f "$ROOT_ABS/manifest.json" && -f "$ROOT_ABS/apartment.json" ]]; then
  APT_DIRS+=("$ROOT_ABS")
else
  while IFS= read -r manifest; do
    apt_dir="$(dirname "$manifest")"
    if [[ -f "$apt_dir/apartment.json" ]]; then
      APT_DIRS+=("$apt_dir")
    fi
  done < <(find "$ROOT_ABS" -maxdepth 3 -type f -name manifest.json | sort)
fi

if [[ "${#APT_DIRS}" -eq 0 ]]; then
  echo "No apartment folders with manifest.json and apartment.json found under $ROOT_ABS" >&2
  exit 1
fi

json_room_type() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit
room = data.get("room") if isinstance(data.get("room"), dict) else {}
print(str(room.get("room_type") or room.get("type") or "").lower())
PY
}

run_blender_room() {
  local room_id="$1"
  local room_type="$2"
  local scene_json="$3"
  local reference_blend="$4"
  local save_blend="$5"
  local build_report="$6"
  local builder="$PROJECT_ROOT/src/Plasement/blender_scene_builder.py"

  echo "[finalize] rebuild room $room_id"
  if [[ -f "$reference_blend" ]]; then
    "$BLENDER" --factory-startup "$reference_blend" -b \
      --python "$builder" -- \
      --json "$scene_json" \
      --project-root "$PROJECT_ROOT/src" \
      --save-blend "$save_blend" \
      --build-report "$build_report" \
      --no-pack-assets \
      --reference-blend "$reference_blend" || {
        if [[ -f "$save_blend" ]]; then
          echo "[finalize] warning: rebuild failed for $room_id; reusing existing $save_blend" >&2
        else
          echo "[finalize] error: rebuild failed for $room_id and no fallback blend exists" >&2
          exit 1
        fi
      }
  else
    "$BLENDER" --factory-startup -b \
      --python "$builder" -- \
      --json "$scene_json" \
      --project-root "$PROJECT_ROOT/src" \
      --save-blend "$save_blend" \
      --build-report "$build_report" \
      --no-pack-assets || {
        if [[ -f "$save_blend" ]]; then
          echo "[finalize] warning: rebuild failed for $room_id; reusing existing $save_blend" >&2
        else
          echo "[finalize] error: rebuild failed for $room_id and no fallback blend exists" >&2
          exit 1
        fi
      }
  fi
}

write_final_report() {
  local apt_dir="$1"
  local out_dir="$apt_dir/apartment_pipeline/$MODE"
  local report="$out_dir/report_requirements.md"
  cat > "$report" <<EOF
# Apartment requirements final report

- apartment: \`$apt_dir\`
- mode: \`$MODE\`
- final blend: \`$out_dir/scene_apartment.requirements.blend\`
- overview render: \`$out_dir/render_apartment.requirements.png\`
- room corner renders: \`$out_dir/room_corner_renders.report.md\`
- cost report: \`$out_dir/renovation_cost_report.md\`

## Room Corner Renders

See \`room_corner_renders.report.md\` for four upper-corner views per room.
EOF
  python3 - "$apt_dir" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

apt = Path(sys.argv[1])
mode = sys.argv[2]
out = apt / "apartment_pipeline" / mode
payload = {
    "apartment_dir": str(apt),
    "mode": mode,
    "outputs": {
        "apartment_scene": str(out / "scene_apartment.requirements.v1.json"),
        "apartment_blend": str(out / "scene_apartment.requirements.blend"),
        "overview_render": str(out / "render_apartment.requirements.png"),
        "corner_report_md": str(out / "room_corner_renders.report.md"),
        "corner_report_json": str(out / "room_corner_renders.report.json"),
        "cost_report_md": str(out / "renovation_cost_report.md"),
        "final_report_md": str(out / "report_requirements.md"),
    },
}
(out / "finalize_requirements.report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

for APT_DIR in "${APT_DIRS[@]}"; do
  OUT_DIR="$APT_DIR/apartment_pipeline/$MODE"
  mkdir -p "$OUT_DIR"

  echo "[finalize] apartment $APT_DIR"
  python3 "$PROJECT_ROOT/src/tools/ensure_apartment_requirements.py" \
    "$APT_DIR" \
    --mode "$MODE" \
    --out-summary "$OUT_DIR/finalize_requirements.ensure_summary.json"

  for ROOM_DIR in "$APT_DIR"/rooms/*; do
    [[ -d "$ROOM_DIR" ]] || continue
    ROOM_ID="$(basename "$ROOM_DIR")"
    PIPE_DIR="$ROOM_DIR/pipeline/$MODE"
    SCENE_JSON="$PIPE_DIR/scene_requirements.v1.json"
    [[ -f "$SCENE_JSON" ]] || continue
    ROOM_TYPE="$(json_room_type "$SCENE_JSON")"
    if [[ "$ROOM_TYPE" == "kitchen" || "$ROOM_ID" == *"kitchen"* ]]; then
      SAVE_BLEND="$PIPE_DIR/scene_kitchen_requirements.blend"
      BUILD_REPORT="$PIPE_DIR/scene_kitchen_requirements.build_report.json"
      REF_BLEND=""
    else
      SAVE_BLEND="$PIPE_DIR/scene_infinigen_clean_supplier.requirements.blend"
      BUILD_REPORT="$PIPE_DIR/scene_infinigen_clean_supplier.requirements.build_report.json"
      REF_BLEND="$PIPE_DIR/infinigen_clean_scene.blend"
    fi
    run_blender_room "$ROOM_ID" "$ROOM_TYPE" "$SCENE_JSON" "$REF_BLEND" "$SAVE_BLEND" "$BUILD_REPORT"
  done

  echo "[finalize] assemble apartment"
  "$BLENDER" --factory-startup -b \
    --python "$PROJECT_ROOT/src/tools/assemble_apartment_blend.py" -- \
    --apt-dir "$APT_DIR" \
    --mode "$MODE" \
    --apartment-scene "$OUT_DIR/scene_apartment.requirements.v1.json" \
    --save-blend "$OUT_DIR/scene_apartment.requirements.blend" \
    --render "$OUT_DIR/render_apartment.requirements.png" \
    --build-report "$OUT_DIR/scene_apartment.requirements.build_report.json" \
    --width "$OVERVIEW_WIDTH" \
    --height "$OVERVIEW_HEIGHT" \
    --samples "$OVERVIEW_SAMPLES"

  echo "[finalize] summarize renovation cost"
  python3 "$PROJECT_ROOT/src/tools/summarize_apartment_cost.py" \
    "$APT_DIR" \
    --mode "$MODE" \
    --out-json "$OUT_DIR/renovation_cost_report.json" \
    --out-md "$OUT_DIR/renovation_cost_report.md"

  echo "[finalize] render room corner views"
  "$BLENDER" --factory-startup "$OUT_DIR/scene_apartment.requirements.blend" -b \
    --python "$PROJECT_ROOT/src/tools/render_apartment_room_corner_views.py" -- \
    --apt-dir "$APT_DIR" \
    --mode "$MODE" \
    --apartment-scene "$OUT_DIR/scene_apartment.requirements.v1.json" \
    --out-dir "$OUT_DIR/room_corner_renders" \
    --report-json "$OUT_DIR/room_corner_renders.report.json" \
    --report-md "$OUT_DIR/room_corner_renders.report.md" \
    --width "$CORNER_WIDTH" \
    --height "$CORNER_HEIGHT" \
    --samples "$CORNER_SAMPLES"

  write_final_report "$APT_DIR"
done

python3 - "$ROOT_ABS" "$MODE" "${APT_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
apts = [Path(x) for x in sys.argv[3:]]
payload = {
    "root": str(root),
    "mode": mode,
    "count": len(apts),
    "apartments": [str(x) for x in apts],
}
(root / "finalize_requirements.summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
