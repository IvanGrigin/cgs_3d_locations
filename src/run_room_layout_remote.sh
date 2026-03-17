#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# src/run_room_layout_remote.sh

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <room.json> <objects_for_server.json> <run_name>" >&2
  exit 2
fi

LOCAL_ROOM_JSON="$1"
LOCAL_OBJECTS_JSON="$2"
RUN_NAME="$3"

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
err() {
  echo "[ERROR] $*" >&2
}

info() {
  echo "[INFO] $*"
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    err "File not found: $p"
    exit 2
  fi
}

yaml_get_py() {
  local yaml_path="$1"
  local dotted_key="$2"
  python3 - "$yaml_path" "$dotted_key" <<'PY'
import sys
from pathlib import Path

yaml_path = Path(sys.argv[1]).expanduser().resolve()
dotted_key = sys.argv[2]

try:
    import yaml
except Exception:
    print("")
    raise SystemExit(0)

if not yaml_path.is_file():
    print("")
    raise SystemExit(0)

data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
cur = data
for part in dotted_key.split("."):
    if not isinstance(cur, dict) or part not in cur:
        print("")
        raise SystemExit(0)
    cur = cur[part]

if cur is None:
    print("")
elif isinstance(cur, (dict, list)):
    print("")
else:
    print(str(cur))
PY
}

pick_value() {
  local current="$1"
  local yaml_path="$2"
  local dotted_key="$3"
  if [[ -n "${current:-}" ]]; then
    printf '%s\n' "$current"
    return
  fi
  yaml_get_py "$yaml_path" "$dotted_key"
}

# ------------------------------------------------------------
# validate local inputs
# ------------------------------------------------------------
require_file "$LOCAL_ROOM_JSON"
require_file "$LOCAL_OBJECTS_JSON"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PATHS_CONFIG="${DIFFUSCENE_PATHS_CONFIG:-$PROJECT_ROOT/config/paths.yaml}"
if [[ ! -f "$PATHS_CONFIG" ]]; then
  info "paths config not found at $PATHS_CONFIG, will rely on env/defaults"
fi

# ------------------------------------------------------------
# resolve ssh config
# ------------------------------------------------------------
REMOTE_HOST="$(pick_value "${DIFFUSCENE_REMOTE_HOST:-}" "$PATHS_CONFIG" "remote.ssh.host")"
REMOTE_PORT="$(pick_value "${DIFFUSCENE_REMOTE_PORT:-}" "$PATHS_CONFIG" "remote.ssh.port")"
REMOTE_USER="$(pick_value "${DIFFUSCENE_REMOTE_USER:-}" "$PATHS_CONFIG" "remote.ssh.user")"
REMOTE_KEY="$(pick_value "${DIFFUSCENE_REMOTE_KEY:-}" "$PATHS_CONFIG" "remote.ssh.key")"

REMOTE_PROJECT_ROOT="$(pick_value "${DIFFUSCENE_REMOTE_PROJECT_ROOT:-}" "$PATHS_CONFIG" "remote.project.root")"
REMOTE_INPUT_ROOT="$(pick_value "${DIFFUSCENE_REMOTE_INPUT_ROOT:-}" "$PATHS_CONFIG" "remote.dirs.input_root")"
REMOTE_RUNS_ROOT="$(pick_value "${DIFFUSCENE_REMOTE_RUNS_ROOT:-}" "$PATHS_CONFIG" "remote.dirs.runs_root")"

REMOTE_CONDA_ENV="$(pick_value "${DIFFUSCENE_REMOTE_CONDA_ENV:-}" "$PATHS_CONFIG" "remote.env.conda_env")"
REMOTE_NLTK_DATA="$(pick_value "${DIFFUSCENE_REMOTE_NLTK_DATA:-}" "$PATHS_CONFIG" "remote.env.nltk_data")"

REMOTE_THREED_FUTURE="$(pick_value "${DIFFUSCENE_REMOTE_THREED_FUTURE:-}" "$PATHS_CONFIG" "remote.data.threed_future")"

DOMAIN="${DIFFUSCENE_DOMAIN:-bedroom}"
case "$DOMAIN" in
  bedroom)
    REMOTE_CFG="$(pick_value "${DIFFUSCENE_REMOTE_CFG:-}" "$PATHS_CONFIG" "remote.config.rearrange_bedrooms")"
    REMOTE_WEIGHT="$(pick_value "${DIFFUSCENE_REMOTE_WEIGHT:-}" "$PATHS_CONFIG" "remote.weights.bedrooms_rearrange")"
    ;;
  livingroom)
    REMOTE_CFG="$(pick_value "${DIFFUSCENE_REMOTE_CFG:-}" "$PATHS_CONFIG" "remote.config.rearrange_livingrooms")"
    REMOTE_WEIGHT="$(pick_value "${DIFFUSCENE_REMOTE_WEIGHT:-}" "$PATHS_CONFIG" "remote.weights.livingrooms_rearrange")"
    ;;
  diningroom)
    REMOTE_CFG="$(pick_value "${DIFFUSCENE_REMOTE_CFG:-}" "$PATHS_CONFIG" "remote.config.rearrange_diningrooms")"
    # в yaml у тебя нет diningrooms_rearrange, поэтому здесь нужен fallback
    REMOTE_WEIGHT="${DIFFUSCENE_REMOTE_WEIGHT:-}"
    ;;
  *)
    err "Unsupported domain: $DOMAIN"
    exit 2
    ;;
esac

REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-diffuscene}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/workspace/projects/DiffuScene}"
REMOTE_INPUT_ROOT="${REMOTE_INPUT_ROOT:-/workspace/room_inputs}"
REMOTE_RUNS_ROOT="${REMOTE_RUNS_ROOT:-$REMOTE_PROJECT_ROOT/scripts/room_runs}"
REMOTE_NLTK_DATA="${REMOTE_NLTK_DATA:-/root/nltk_data}"
REMOTE_THREED_FUTURE="${REMOTE_THREED_FUTURE:-/workspace/data/models/3D-FUTURE-model-processed/3D-FUTURE-model}"

if [[ -z "$REMOTE_HOST" ]]; then
  err "REMOTE_HOST is empty"
  exit 2
fi

if [[ -z "$REMOTE_KEY" ]]; then
  err "REMOTE_KEY is empty"
  exit 2
fi

REMOTE_KEY="${REMOTE_KEY/#\~/$HOME}"

if [[ ! -f "$REMOTE_KEY" ]]; then
  err "SSH key not found: $REMOTE_KEY"
  exit 2
fi

if [[ -z "${REMOTE_CFG:-}" ]]; then
  err "REMOTE_CFG is empty for domain=$DOMAIN"
  exit 2
fi

if [[ -z "${REMOTE_WEIGHT:-}" ]]; then
  err "REMOTE_WEIGHT is empty for domain=$DOMAIN"
  exit 2
fi

REMOTE_HOST_SPEC="${REMOTE_USER}@${REMOTE_HOST}"

REMOTE_INPUT_DIR="${REMOTE_INPUT_ROOT%/}/${RUN_NAME}"
REMOTE_RUN_DIR="${REMOTE_RUNS_ROOT%/}/${RUN_NAME}"

REMOTE_ROOM_JSON="${REMOTE_INPUT_DIR}/room.json"
REMOTE_OBJECTS_JSON="${REMOTE_INPUT_DIR}/objects_for_server.json"

LOCAL_RESULTS_DIR="$(cd "$(dirname "$LOCAL_ROOM_JSON")" && pwd)"

SSH_OPTS=(
  -i "$REMOTE_KEY"
  -p "$REMOTE_PORT"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=120
)

SCP_OPTS=(
  -i "$REMOTE_KEY"
  -P "$REMOTE_PORT"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

# ------------------------------------------------------------
# debug print
# ------------------------------------------------------------
info "REMOTE_HOST      = $REMOTE_HOST"
info "REMOTE_PORT      = $REMOTE_PORT"
info "REMOTE_USER      = $REMOTE_USER"
info "REMOTE_PROJECT   = $REMOTE_PROJECT_ROOT"
info "REMOTE_INPUT_DIR = $REMOTE_INPUT_DIR"
info "REMOTE_RUN_DIR   = $REMOTE_RUN_DIR"
info "DOMAIN           = $DOMAIN"
info "REMOTE_CFG       = $REMOTE_CFG"
info "REMOTE_WEIGHT    = $REMOTE_WEIGHT"

# ------------------------------------------------------------
# quick ssh check
# ------------------------------------------------------------
info "Checking SSH connectivity..."
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST_SPEC" 'echo "[REMOTE] ssh ok: $(hostname)"'

# ------------------------------------------------------------
# prepare remote dirs
# ------------------------------------------------------------
info "Preparing remote directories..."
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST_SPEC" "
  mkdir -p '$REMOTE_INPUT_DIR'
  mkdir -p '$REMOTE_RUN_DIR'
"

# ------------------------------------------------------------
# upload inputs
# ------------------------------------------------------------
info "Uploading room/object files..."
scp "${SCP_OPTS[@]}" "$LOCAL_ROOM_JSON"    "${REMOTE_HOST_SPEC}:${REMOTE_ROOM_JSON}"
scp "${SCP_OPTS[@]}" "$LOCAL_OBJECTS_JSON" "${REMOTE_HOST_SPEC}:${REMOTE_OBJECTS_JSON}"

# ------------------------------------------------------------
# run remote inference
# ------------------------------------------------------------
info "Running remote DiffuScene inference..."

read -r -d '' REMOTE_SCRIPT <<EOF || true
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true

conda activate "$REMOTE_CONDA_ENV"

cd "$REMOTE_PROJECT_ROOT"

export PYTHONPATH="$REMOTE_PROJECT_ROOT:$REMOTE_PROJECT_ROOT/ChamferDistancePytorch:\$PYTHONPATH"
export NLTK_DATA="$REMOTE_NLTK_DATA"

python scripts/infer_room.py \
  --room_json "$REMOTE_ROOM_JSON" \
  --objects_json "$REMOTE_OBJECTS_JSON" \
  --run_dir "$REMOTE_RUN_DIR" \
  --cfg "$REMOTE_CFG" \
  --threed_future "$REMOTE_THREED_FUTURE" \
  --weight "$REMOTE_WEIGHT" \
  --n_sequences 1

test -f "$REMOTE_RUN_DIR/placements_room.json"
EOF

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST_SPEC" "$REMOTE_SCRIPT"

# ------------------------------------------------------------
# download outputs
# ------------------------------------------------------------
info "Downloading result files..."

download_if_exists() {
  local remote_file="$1"
  local local_file="$2"
  if ssh "${SSH_OPTS[@]}" "$REMOTE_HOST_SPEC" "test -f '$remote_file'"; then
    scp "${SCP_OPTS[@]}" "${REMOTE_HOST_SPEC}:${remote_file}" "$local_file"
  fi
}

download_if_exists "$REMOTE_RUN_DIR/placements_room.json"       "$LOCAL_RESULTS_DIR/placements_room.json"
download_if_exists "$REMOTE_RUN_DIR/placements_room_check.json" "$LOCAL_RESULTS_DIR/placements_room_check.json"
download_if_exists "$REMOTE_RUN_DIR/pred_bbox.json"             "$LOCAL_RESULTS_DIR/pred_bbox.json"
download_if_exists "$REMOTE_RUN_DIR/pred_bbox_metric.json"      "$LOCAL_RESULTS_DIR/pred_bbox_metric.json"

echo "Done. Results in: $LOCAL_RESULTS_DIR"