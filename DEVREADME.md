## Dev Runbook

### Локальное окружение
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### SSH на remote
```bash
ssh -p 32172 root@1.208.108.242
```

### Ollama tunnel с remote
```bash
ssh -p 32172 -N -L 11435:127.0.0.1:11434 root@1.208.108.242
curl http://127.0.0.1:11435/api/tags
```

## Основной pipeline

### M3DLayout autoregressive
```bash
python3 src/run_pipeline.py \
  --placer m3dlayout_ar \
  --room data/input/room.json \
  --prompt "The room has a double bed and two nightstands." \
  --remote-host 1.208.108.242 \
  --remote-port 32172 \
  --remote-user root \
  --remote-key ~/.ssh/id_ed25519 \
  --remote-conda-env m3dlayout \
  --skip-blender \
  --keep-tmp
```

### M3DLayout diffusion
```bash
python3 src/run_pipeline.py \
  --placer m3dlayout_diffusion \
  --room data/input/room.json \
  --prompt "The room has a double bed and two nightstands." \
  --remote-host 1.208.108.242 \
  --remote-port 32172 \
  --remote-user root \
  --remote-key ~/.ssh/id_ed25519 \
  --remote-conda-env m3dlayout \
  --skip-blender \
  --keep-tmp
```

### Infinigen clean
```bash
python3 src/run_pipeline.py \
  --placer infinigen_clean \
  --room data/input/room.json \
  --prompt "placeholder" \
  --remote-host 1.208.108.242 \
  --remote-port 32172 \
  --remote-user root \
  --remote-key ~/.ssh/id_ed25519 \
  --remote-conda-env infinigen \
  --remote-infinigen-src /workspace/infinigen/src \
  --skip-blender \
  --keep-tmp
```

## Benchmark

### Benchmark для M3DLayout и Infinigen
```bash
python3 src/run_mode_benchmark.py \
  --room data/input/room.json \
  --prompt "The room has a double bed and two nightstands." \
  --modes m3dlayout_ar,m3dlayout_diffusion,infinigen_clean \
  --count-per-mode 5 \
  --remote-host 1.208.108.242 \
  --remote-port 32172 \
  --remote-user root \
  --remote-key ~/.ssh/id_ed25519 \
  --keep-tmp
```

## Quality Search

### Quality search по M3DLayout и Infinigen
```bash
python3 src/run_quality_search.py \
  --room data/input/room.json \
  --prompt "The room has a double bed and two nightstands." \
  --stage-sequence m3dlayout_ar:m3dlayout_ar,m3dlayout_diffusion:m3dlayout_diffusion,infinigen_clean:infinigen_clean \
  --remote-host 1.208.108.242 \
  --remote-port 32172 \
  --remote-user root \
  --remote-key ~/.ssh/id_ed25519 \
  --skip-blender \
  --keep-tmp
```

## Примечания

- `run_pipeline.py`, `run_mode_benchmark.py` и `run_quality_search.py` больше не требуют GLB-комнаты для Blender-сборки сцены.
- `--no-import-glb` и `--glb` оставлены только для совместимости и в рабочих командах не нужны.
- Если remote defaults уже записаны в `config/paths.yaml`, можно не передавать `--remote-host`, `--remote-port`, `--remote-user`, `--remote-key`, `--remote-conda-env`, `--remote-infinigen-src` вручную.
- Для `infinigen_clean` prompt сейчас технический и на сам placer не влияет.


python3 src/run_pipeline.py \
  --placer infinigen_clean \
  --modes infinigen_clean \
  --room data/input/custom_rooms/bedroom_supplier_fallback_20260416.json \
  --prompt-file data/input/custom_rooms/bedroom_supplier_fallback_20260416.prompt.txt \
  --ollama-url http://127.0.0.1:11435 \
  --ollama-model gpt-oss:20b \
  --supplier-catalog-json data/sourse/suppliers/supplier_product_full.json \
  --supplier-top-k 12 \
  --supplier-llm-provider ollama \
  --supplier-llm-top-n 6 \
  --supplier-require-local-asset \
  --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --headless \
  --no-bbox-fallback \
  --run-dir out/custom_rooms/bedroom_supplier_fallback_20260416/run_01


ssh -p 32172 -i /Users/a01/.ssh/id_ed25519 root@1.208.108.242

ls data/sourse/suppliers/supplier_catalog_canonical.json


cd /Users/a01/Desktop/ITMO/sem_7/cgs_3d_locations

OUT="out/procedural_bedroom_supplier_trellis_box_33_fallback_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

cat > "$OUT/room_bedroom_6x4_7.v1.json" <<'JSON'
{
  "schema": "scene.v1",
  "room": {
    "id": "room_001",
    "name": "Bedroom",
    "type": "bedroom",
    "ceiling_height": 2.8,
    "floor_polygon": [
      {"x": 0.0, "y": 0.0},
      {"x": 6.2, "y": 0.0},
      {"x": 6.2, "y": 4.7},
      {"x": 0.0, "y": 4.7}
    ],
    "walls": [
      {"id": "w0", "from_vertex": 0, "to_vertex": 1},
      {"id": "w1", "from_vertex": 1, "to_vertex": 2},
      {"id": "w2", "from_vertex": 2, "to_vertex": 3},
      {"id": "w3", "from_vertex": 3, "to_vertex": 0}
    ],
    "doors": [
      {
        "id": "door_0",
        "wall_id": "w0",
        "s": 2.65,
        "width": 0.9,
        "z0": 0.0,
        "height": 2.05,
        "swing": {"hinge": "left", "direction": "in"}
      }
    ],
    "windows": [
      {
        "id": "win_0",
        "wall_id": "w2",
        "s": 2.3,
        "width": 1.6,
        "z0": 0.9,
        "height": 1.2,
        "glazing": "double"
      }
    ]
  },
  "placements": []
}
JSON

python3 src/tools/run_procedural_room_supplier.py \
  --room "$OUT/room_bedroom_6x4_7.v1.json" \
  --out-dir "$OUT" \
  --prompt "large realistic modern bedroom with queen bed, nightstands, wardrobe, dresser, desk, chair, floor lamp, ceiling light, wall lights, plant, rug, bedding and decorative objects" \
  --density very_high \
  --policy always \
  --replace-existing \
  --seed 1 \
  --supplier-catalog-json data/sourse/suppliers/supplier_catalog_canonical.json \
  --supplier-selection-mode optimal \
  --supplier-selection-strategy balanced \
  --top-k 80 \
  --trellis-generate-missing-assets \
  --trellis-max-assets 33 \
  --trellis-max-failures-per-candidate 2 \
  --trellis-progress-log \
  --trellis-server-host 1.208.108.242 \
  --trellis-server-port 32172 \
  --trellis-server-user root \
  --trellis-ssh-key ~/.ssh/id_ed25519 \
  --trellis-remote-root /workspace/trellis_supplier_jobs \
  --trellis-remote-trellis-root /workspace/TRELLIS-BOX \
  --trellis-remote-model-dir JeffreyXiang/TRELLIS-image-large \
  --trellis-remote-runner-path /workspace/TRELLIS-BOX/trellis_box_lowvram_remote_runner_retry14.py \
  --trellis-remote-cuda-visible-devices 0 \
  --trellis-multi-mode stochastic \
  --trellis-max-images 1 \
  --trellis-oom-retry-max-images 1 \
  --no-trellis-disable-after-oom \
  --trellis-seed 1 \
  --trellis-sparse-steps 4 \
  --trellis-slat-steps 4 \
  --trellis-image-size 224 \
  --trellis-texture-size 128 \
  --trellis-simplify 0.985 \
  --trellis-fill-holes-resolution 128 \
  --trellis-fill-holes-num-views 32 \
  --trellis-vlm-single-object-filter \
  --trellis-vlm-provider ollama \
  --trellis-vlm-ollama-url http://127.0.0.1:11435 \
  --trellis-vlm-model llama3.2-vision:11b \
  --trellis-vlm-timeout 120 \
  --trellis-vlm-unload-after-filter \
  --build-blend \
  --scene-builder-script src/Plasement/blender_scene_builder.py \
  --out-blend final_supplier_procedural.blend \
  --out-png final_supplier_procedural.png \
  2>&1 | tee "$OUT/run.log"

echo
echo "OUT=$OUT"
echo "PNG=$OUT/final_supplier_procedural.png"
echo "BLEND=$OUT/final_supplier_procedural.blend"
echo "REPORT=$OUT/procedural_room_supplier_report.json"
echo "GLB files:"
find "$OUT/trellis_missing_assets" -name "asset.trellis.glb" -print 2>/dev/null


Для RTX 3090
ssh -i ~/.ssh/id_ed25519 -p 28553 root@84.2.13.196


TRELLIS.2 на RTX 3090: 84.2.13.196:28553
LLM/VLM Ollama на RTX 3060: tunnel 127.0.0.1:11435 -> 1.208.108.242:11434


