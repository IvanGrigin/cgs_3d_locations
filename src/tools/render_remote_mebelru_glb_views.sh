#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-84.2.13.196}"
PORT="${PORT:-28553}"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
ROOT="${ROOT:-/workspace/trellis2_supplier_jobs_mebelru_all}"
RES="${RES:-512}"
LOG="${LOG:-out/glb_creator_mebelru_remote_render_views.log}"

mkdir -p "$(dirname "$LOG")"

scp -P "$PORT" -i "$SSH_KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  src/tools/glb_creator_render_glb_blender.py \
  "$REMOTE_USER@$HOST:$ROOT/glb_creator_render_glb_blender.py"

ssh -p "$PORT" -i "$SSH_KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "$REMOTE_USER@$HOST" \
  ROOT="$ROOT" RES="$RES" 'bash -s' <<'REMOTE_RENDER' 2>&1 | tee "$LOG"
set -euo pipefail

SCRIPT="$ROOT/glb_creator_render_glb_blender.py"

find "$ROOT" -mindepth 2 -path "*/output/asset.trellis.glb" -type f | sort > /tmp/mebelru_remote_glbs.txt
TOTAL=$(wc -l < /tmp/mebelru_remote_glbs.txt | tr -d " ")
DONE=0
SKIP=0
FAIL=0

echo "[render] total_glb=$TOTAL root=$ROOT resolution=$RES"

while IFS= read -r GLB; do
  JOB_DIR=$(dirname "$(dirname "$GLB")")
  JOB_ID=$(basename "$JOB_DIR")
  RENDERS="$JOB_DIR/renders"
  LOG_DIR="$JOB_DIR/logs"
  mkdir -p "$RENDERS" "$LOG_DIR"

  EXISTING=$(find "$RENDERS" -maxdepth 1 -type f -name "*__view_*.png" | wc -l | tr -d " ")
  if [ "$EXISTING" -ge 4 ]; then
    SKIP=$((SKIP + 1))
    DONE=$((DONE + 1))
    echo "[render][skip] $DONE/$TOTAL $JOB_ID existing=$EXISTING"
    continue
  fi

  echo "[render][start] $((DONE + 1))/$TOTAL $JOB_ID"

  if blender --background --python "$SCRIPT" -- \
      --glb "$GLB" \
      --out-dir "$RENDERS" \
      --resolution "$RES" \
      > "$LOG_DIR/remote_render_stdout.log" 2>&1; then

    (
      cd "$RENDERS"
      cp render_front.png "${JOB_ID}__view_front.png" 2>/dev/null || true
      cp render_left.png "${JOB_ID}__view_left.png" 2>/dev/null || true
      cp render_right.png "${JOB_ID}__view_right.png" 2>/dev/null || true
      cp render_three_quarter.png "${JOB_ID}__view_three_quarter.png" 2>/dev/null || true
    )

    DONE=$((DONE + 1))
    COUNT=$(find "$RENDERS" -maxdepth 1 -type f -name "*__view_*.png" | wc -l | tr -d " ")
    echo "[render][done] $DONE/$TOTAL $JOB_ID png=$COUNT"
  else
    FAIL=$((FAIL + 1))
    DONE=$((DONE + 1))
    echo "[render][fail] $DONE/$TOTAL $JOB_ID log=$LOG_DIR/remote_render_stdout.log"
  fi
done < /tmp/mebelru_remote_glbs.txt

echo "[render][summary] total=$TOTAL done=$DONE skipped=$SKIP failed=$FAIL"
echo "[render][count_named_png] $(find "$ROOT" -path "*/renders/*__view_*.png" -type f | wc -l | tr -d " ")"
echo "[render][count_manifest] $(find "$ROOT" -path "*/renders/render_manifest.json" -type f | wc -l | tr -d " ")"
REMOTE_RENDER
