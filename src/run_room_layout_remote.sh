#!/usr/bin/env bash
set -euo pipefail

ROOM_JSON="${1:?need room json}"
OBJECTS_JSON="${2:?need objects json}"
RUN_NAME="${3:-run_$(date +%s)}"

SERVER_USER="root"
SERVER_HOST="207.102.87.207"
SERVER_PORT="40527"
SSH_KEY="${HOME}/.ssh/id_ed25519"

REMOTE_BASE="/workspace/projects/DiffuScene/room_inbox_remote"
REMOTE_RUN_DIR="${REMOTE_BASE}/${RUN_NAME}"

# Локальные результаты кладём не в $(pwd), а в ту же папку, где лежит ROOM_JSON,
# то есть в текущий run_dir пайплайна.
LOCAL_OUT_DIR="$(cd "$(dirname "${ROOM_JSON}")" && pwd)"
mkdir -p "${LOCAL_OUT_DIR}"

echo "==> create remote dir"
ssh -T -i "${SSH_KEY}" -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_HOST}" \
  "mkdir -p '${REMOTE_RUN_DIR}'"

echo "==> upload inputs"
scp -i "${SSH_KEY}" -P "${SERVER_PORT}" \
  "${ROOM_JSON}" \
  "${OBJECTS_JSON}" \
  "${SERVER_USER}@${SERVER_HOST}:${REMOTE_RUN_DIR}/"

echo "==> run server pipeline"
ssh -T -i "${SSH_KEY}" -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_HOST}" <<EOF
set -euo pipefail
cd /workspace/projects/DiffuScene
source /opt/miniforge3/etc/profile.d/conda.sh || true
conda activate diffuscene || true

python scripts/objects_json_to_model_input.py \
  --in_json "${REMOTE_RUN_DIR}/$(basename "${OBJECTS_JSON}")" \
  --out_json "${REMOTE_RUN_DIR}/objects_model_input.json" \
  --allow-missing-objfeats

python scripts/infer_room.py \
  --room_json "${REMOTE_RUN_DIR}/$(basename "${ROOM_JSON}")" \
  --objects_json "${REMOTE_RUN_DIR}/objects_model_input.json" \
  --run_dir "${REMOTE_RUN_DIR}/run" \
  --cfg /workspace/projects/DiffuScene/config/rearrange/diffusion_bedrooms_instancond_lat32_v_rearrange.yaml \
  --threed_future /workspace/data/assets/datasets/3d_front_processed/threed_future_model_bedroom.pkl \
  --weight /workspace/data/assets/checkpoints/pretrained_diffusion/bedrooms_rearrange/model_17000 \
  --n_sequences 1

python - <<PY
import json, math
from pathlib import Path

run_dir = Path("${REMOTE_RUN_DIR}/run")
pred = json.loads((run_dir / "model_out/seq_0000/pred_bbox.json").read_text(encoding="utf-8"))
room_meta = json.loads((run_dir / "room_meta.json").read_text(encoding="utf-8"))
room = json.loads(Path(room_meta["room_json"]).read_text(encoding="utf-8"))

CENTROIDS_MIN = [-2.7625005, 0.045, -2.75275]
CENTROIDS_MAX = [2.77844175, 3.6248396, 2.81854277]
SIZES_MIN = [0.03998288, 0.02000002, 0.012772]
SIZES_MAX = [2.8682, 1.770065, 1.698315]

def descale(v, mn, mx):
    return [((x + 1.0) / 2.0) * (b - a) + a for x, a, b in zip(v, mn, mx)]

def lerp(a, b, t):
    return a + (b - a) * t

def remap(v, src_min, src_max, dst_min, dst_max):
    if abs(src_max - src_min) < 1e-12:
        return (dst_min + dst_max) / 2.0
    t = (v - src_min) / (src_max - src_min)
    return lerp(dst_min, dst_max, t)

poly = room["room"]["floor_polygon"]
room_xs = [float(p["x"]) for p in poly]
room_ys = [float(p["y"]) for p in poly]
room_xmin, room_xmax = min(room_xs), max(room_xs)
room_ymin, room_ymax = min(room_ys), max(room_ys)

MODEL_X_MIN, MODEL_X_MAX = -2.7625005, 2.77844175
MODEL_Z_MIN, MODEL_Z_MAX = -2.75275, 2.81854277

items = []
for it in pred["items"]:
    t_metric = descale(it["translation"], CENTROIDS_MIN, CENTROIDS_MAX)
    s_metric = descale(it["size"], SIZES_MIN, SIZES_MAX)
    yaw = math.atan2(it["angle"][1], it["angle"][0])

    room_x = remap(t_metric[0], MODEL_X_MIN, MODEL_X_MAX, room_xmin, room_xmax)
    room_y = remap(t_metric[2], MODEL_Z_MIN, MODEL_Z_MAX, room_ymin, room_ymax)

    items.append({
        "i": it["i"],
        "class_name": it["class_name"],
        "position_room_xy_m": [room_x, room_y],
        "z_floor_m": 0.0,
        "size_m": s_metric,
        "yaw_rad": yaw,
        "yaw_deg": yaw * 180.0 / math.pi
    })

out = {"items": items}
(run_dir / "model_out/seq_0000/placements_room.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("saved placements_room.json")
PY
EOF

echo "==> download results"
scp -i "${SSH_KEY}" -P "${SERVER_PORT}" \
  "${SERVER_USER}@${SERVER_HOST}:${REMOTE_RUN_DIR}/run/model_out/seq_0000/placements_room.json" \
  "${LOCAL_OUT_DIR}/"

scp -i "${SSH_KEY}" -P "${SERVER_PORT}" \
  "${SERVER_USER}@${SERVER_HOST}:${REMOTE_RUN_DIR}/run/model_out/seq_0000/pred_bbox.json" \
  "${LOCAL_OUT_DIR}/" || true

echo "Done. Results in: ${LOCAL_OUT_DIR}"