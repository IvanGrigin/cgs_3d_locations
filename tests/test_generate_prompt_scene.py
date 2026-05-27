import json
import types
import sys
from pathlib import Path

import pytest

pytest.skip("legacy test for archived module src.pipeline.generate_prompt_scene", allow_module_level=True)

from src.pipeline import generate_prompt_scene as gps
from src.pipeline.generate_prompt_scene import (
    _build_llm_client,
    _candidate_score,
    _copy_if_exists,
    _is_valid_final_candidate,
    _materialize_final,
    _screen_and_score,
    _select_best,
    _write_run_status,
)
from src.prompt_compiler.schemas import GateResult, JudgeResult
from src.prompt_compiler.llm_client import OllamaJSONLLMClient


def test_no_valid_candidate_status_written(tmp_path: Path) -> None:
    gate = GateResult(
        passed=False,
        rule_score=0.0,
        hard_failures=["missing_required_bed", "empty_scene_generated"],
        soft_failures=[],
    )
    judge = JudgeResult(
        passed=False,
        total_score=0.0,
        functionality_score=0.0,
        prompt_match_score=0.0,
        style_match_score=0.0,
        composition_score=0.0,
    )
    row = {"gate": gate, "judge": judge, "candidate_dir": tmp_path / "seed_000"}
    assert not _is_valid_final_candidate(row, min_judge_score=6.0)
    _write_run_status(tmp_path, status="no_valid_candidate", selected_candidate="seed_000", reason="all_failed")
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "no_valid_candidate"
    assert status["selected_candidate"] == "seed_000"


def test_build_llm_client_behavior():
    assert _build_llm_client("none", "m", "http://localhost") is None
    assert isinstance(_build_llm_client("ollama", "gpt", "http://localhost"), OllamaJSONLLMClient)
    try:
        _build_llm_client("unknown", "m", "http://localhost")
    except ValueError as exc:
        assert "unsupported llm provider" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_candidate_score_and_selection_filters():
    gate = GateResult(passed=True, rule_score=2.0)
    judge = JudgeResult(
        passed=True,
        total_score=8.0,
        functionality_score=1.0,
        prompt_match_score=1.0,
        style_match_score=1.0,
        composition_score=1.0,
    )
    assert _candidate_score(gate, None) == gate.rule_score
    assert _candidate_score(gate, judge) == 2.0 * 0.55 + 8.0 * 0.45

    pass_row = {
        "gate": GateResult(passed=True, rule_score=4.0),
        "judge": JudgeResult(passed=True, total_score=7.5, functionality_score=0, prompt_match_score=0, style_match_score=0, composition_score=0),
        "combined_score": 4.0,
    }
    fail_row = {"gate": GateResult(passed=False, rule_score=9.0), "judge": None, "combined_score": 9.0}
    assert _select_best([fail_row, pass_row], min_judge_score=7.0)["gate"].passed

    all_fail = _select_best([fail_row], min_judge_score=7.0)
    assert all_fail is not None
    assert all_fail["gate"] is fail_row["gate"]


def test_copy_and_materialize_helpers(tmp_path: Path):
    src_file = tmp_path / "src.txt"
    src_file.write_text("data", encoding="utf-8")
    dst_file = tmp_path / "dst.txt"
    _copy_if_exists(src_file, dst_file)
    assert dst_file.read_text(encoding="utf-8") == "data"
    missing = tmp_path / "missing.txt"
    _copy_if_exists(missing, tmp_path / "ignored.txt")
    assert not (tmp_path / "ignored.txt").exists()

    best = {
        "candidate_dir": tmp_path / "candidate",
        "candidate_dir2": tmp_path / "candidate",
    }
    candidate = best["candidate_dir"]
    candidate.mkdir()
    (candidate / "infinigen_clean_scene.blend").write_text("blend", encoding="utf-8")
    (candidate / "render.png").write_text("png", encoding="utf-8")
    (candidate / "placement.json").write_text("{}", encoding="utf-8")
    (candidate / "inventory_summary.json").write_text("{}", encoding="utf-8")
    (candidate / "rule_gate.json").write_text("{}", encoding="utf-8")

    run_root = tmp_path / "run"
    _materialize_final(run_root, best)
    assert (run_root / "final" / "selected_candidate.txt").read_text(encoding="utf-8") == "candidate"


def test_valid_candidate_check():
    gate = GateResult(
        passed=True,
        rule_score=1.0,
    )
    judge = JudgeResult(
        passed=True,
        total_score=6.0,
        functionality_score=1.0,
        prompt_match_score=1.0,
        style_match_score=1.0,
        composition_score=1.0,
    )
    assert _is_valid_final_candidate({"gate": gate, "judge": judge}, min_judge_score=5.0)
    assert _is_valid_final_candidate({"gate": gate, "judge": None}, min_judge_score=6.5)


def test_screen_and_score_calls_scoring_pipeline(monkeypatch, tmp_path: Path):
    candidates = [tmp_path / "candidate_a", tmp_path / "candidate_b"]
    for cand in candidates:
        cand.mkdir()

    def fake_run_screening(*_, **__):
        return None

    recorded = {"eval": 0, "judge": 0, "writes": 0}

    def fake_eval(compiled, candidate_dir):
        recorded["eval"] += 1
        return GateResult(passed=True, rule_score=float(candidate_dir.name.endswith("a")))

    def fake_judge(compiled, candidate_dir, client):
        recorded["judge"] += 1
        score = 10.0 if candidate_dir.name.endswith("a") else 3.0
        return JudgeResult(passed=True, total_score=score, functionality_score=0, prompt_match_score=0, style_match_score=0, composition_score=0)

    def fake_write_gate_result(gate, path):
        recorded["writes"] += 1
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(gps, "run_screening_from_compiled_policy", fake_run_screening)
    monkeypatch.setattr(gps, "evaluate_candidate", fake_eval)
    monkeypatch.setattr(gps, "run_judge", fake_judge)
    monkeypatch.setattr(gps, "write_gate_result", fake_write_gate_result)

    results = _screen_and_score(
        compiled_policy=types.SimpleNamespace(),
        run_root=tmp_path / "run",
        llm_client=None,
        screening_base_dir=tmp_path,
        seeds=[1, 2, 3],
        skip_judge=False,
        remote_kwargs={},
    )

    assert len(results) == 2
    assert recorded["eval"] == 2
    assert recorded["judge"] == 2
    assert recorded["writes"] == 2


def test_main_valid_candidate_path(monkeypatch, tmp_path: Path):
    policy = types.SimpleNamespace(
        acceptance_policy=types.SimpleNamespace(min_judge_score=6.0),
        scene_id="",
        artifacts={},
    )

    gate = GateResult(
        passed=True,
        rule_score=1.0,
    )
    judge = JudgeResult(
        passed=True,
        total_score=8.0,
        functionality_score=1.0,
        prompt_match_score=1.0,
        style_match_score=1.0,
        composition_score=1.0,
    )
    row = {
        "gate": gate,
        "judge": judge,
        "candidate_dir": tmp_path / "candidate_ok",
        "combined_score": _candidate_score(gate, judge),
    }
    (row["candidate_dir"]).mkdir()

    recorded = {}

    monkeypatch.setattr(gps, "load_policies", lambda *_: {})
    monkeypatch.setattr(gps, "extract_intent", lambda *_, **__: {})
    monkeypatch.setattr(gps, "save_intent_trace", lambda *_, **__: None)
    monkeypatch.setattr(gps, "compile_prompt_intent", lambda *_: policy)
    monkeypatch.setattr(gps, "write_compiled_artifacts", lambda *_, **__: None)
    monkeypatch.setattr(gps, "_screen_and_score", lambda *_, **__: [row])
    monkeypatch.setattr(gps, "_materialize_final", lambda run_root, best: recorded.setdefault("materialize", True))
    monkeypatch.setattr(gps, "_write_run_status", lambda run_root, *, status, selected_candidate="", reason="": recorded.setdefault("status", status))
    monkeypatch.setattr(gps, "build_scene_report", lambda *_: recorded.setdefault("report", True))

    monkeypatch.setattr(sys, "argv", [
        "generate_prompt_scene.py",
        "--prompt", "small bedroom with bed",
        "--out-dir", str(tmp_path / "run_ok"),
        "--llm-provider", "none",
        "--skip-judge",
    ])

    gps.main()
    assert recorded["status"] == "ok"
    assert recorded["materialize"] is True
    assert recorded["report"] is True


def test_main_no_valid_candidate_path(monkeypatch, tmp_path: Path):
    policy = types.SimpleNamespace(
        acceptance_policy=types.SimpleNamespace(min_judge_score=6.0),
        scene_id="",
        artifacts={},
    )

    gate = GateResult(
        passed=False,
        rule_score=1.0,
    )
    judge = JudgeResult(
        passed=False,
        total_score=1.0,
        functionality_score=1.0,
        prompt_match_score=1.0,
        style_match_score=1.0,
        composition_score=1.0,
    )
    row = {
        "gate": gate,
        "judge": judge,
        "candidate_dir": tmp_path / "candidate_bad",
        "combined_score": _candidate_score(gate, judge),
    }
    row["candidate_dir"].mkdir()

    recorded = {}

    monkeypatch.setattr(gps, "load_policies", lambda *_: {})
    monkeypatch.setattr(gps, "extract_intent", lambda *_, **__: {})
    monkeypatch.setattr(gps, "save_intent_trace", lambda *_, **__: None)
    monkeypatch.setattr(gps, "compile_prompt_intent", lambda *_: policy)
    monkeypatch.setattr(gps, "write_compiled_artifacts", lambda *_, **__: None)
    monkeypatch.setattr(gps, "_screen_and_score", lambda *_, **__: [row])
    monkeypatch.setattr(gps, "_materialize_final", lambda *_, **__: recorded.setdefault("materialize", True))
    monkeypatch.setattr(gps, "_write_run_status", lambda run_root, *, status, selected_candidate="", reason="": recorded.setdefault("status", status))
    monkeypatch.setattr(gps, "build_scene_report", lambda *_: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_prompt_scene.py",
            "--prompt",
            "small bedroom with bed",
            "--out-dir",
            str(tmp_path / "run_fail"),
            "--llm-provider",
            "none",
            "--skip-judge",
            "--max-repair-rounds",
            "0",
        ],
    )

    gps.main()
    assert recorded["status"] == "no_valid_candidate"
    assert "materialize" not in recorded
