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
