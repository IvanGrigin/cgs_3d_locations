import csv
import json
from pathlib import Path

from types import SimpleNamespace

import pytest

pytest.skip("legacy test for archived module src.pipeline.batch_generate_prompt_scenes", allow_module_level=True)

from src.pipeline.batch_generate_prompt_scenes import _load_prompts
from src.pipeline.batch_generate_prompt_scenes import main as batch_main


def test_load_prompts_reads_jsonl_yaml_csv(tmp_path: Path):
    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"scene_id": "a", "prompt": "first"}),
                json.dumps({"prompt": "second"}),
            ]
        ),
        encoding="utf-8",
    )
    assert len(_load_prompts(jsonl)) == 2

    yaml_path = tmp_path / "prompts.yaml"
    yaml_path.write_text("- prompt: one\n- prompt: two\n", encoding="utf-8")
    assert len(_load_prompts(yaml_path)) == 2

    csv_path = tmp_path / "prompts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scene_id", "prompt"])
        w.writeheader()
        w.writerow({"scene_id": "c", "prompt": "three"})
    assert len(_load_prompts(csv_path)) == 1


def test_load_prompts_rejects_unsupported(tmp_path: Path):
    with pytest.raises(ValueError):
        _load_prompts(tmp_path / "x.txt")


def test_batch_main_writes_manifest(monkeypatch, tmp_path: Path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")

    out_dir = tmp_path / "runs"
    calls = []

    class FakeResult:
        returncode = 0

    def fake_run(cmd, check=False):
        calls.append(cmd)
        return FakeResult()

    import sys as _sys

    monkeypatch.setattr(_sys.modules["subprocess"], "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "batch_generate_prompt_scenes.py",
            "--prompts",
            str(prompts),
            "--out-dir",
            str(out_dir),
            "--llm-provider",
            "none",
        ],
    )

    batch_main()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["status"] == "done"
    assert calls and calls[0][0] == "python3"
