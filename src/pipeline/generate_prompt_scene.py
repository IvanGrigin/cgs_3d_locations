from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.Plasement.run_infinigen_clean import run_screening_from_compiled_policy
from src.prompt_compiler.compile_to_infinigen import compile_prompt_intent, write_compiled_artifacts
from src.prompt_compiler.llm_client import BaseLLMClient, OllamaJSONLLMClient, StubLLMClient
from src.prompt_compiler.policies import load_policies
from src.prompt_compiler.prompt_to_intent import extract_intent, save_intent_trace
from src.prompt_compiler.schemas import CompiledPolicy, GateResult, JudgeResult
from src.scene_quality.judge_runner import run_judge
from src.scene_quality.quality_gate import evaluate_candidate, write_gate_result
from src.scene_quality.repair_loop import apply_repair_plan, build_repair_plan
from src.scene_quality.report_builder import build_scene_report


def _build_llm_client(provider: str, model: str, base_url: str) -> BaseLLMClient | None:
    if provider == "none":
        return None
    if provider == "ollama":
        return OllamaJSONLLMClient(base_url=base_url, model=model)
    raise ValueError(f"unsupported llm provider: {provider}")


def _candidate_score(gate: GateResult, judge: JudgeResult | None) -> float:
    if judge is None:
        return gate.rule_score
    return gate.rule_score * 0.55 + judge.total_score * 0.45


def _screen_and_score(
    compiled_policy: CompiledPolicy,
    run_root: Path,
    llm_client: BaseLLMClient | None,
    *,
    screening_base_dir: Path,
    seeds: list[int],
    skip_judge: bool,
    remote_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    compiled_policy_path = run_root / "compiled_policy.active.json"
    run_screening_from_compiled_policy(
        compiled_policy_path=compiled_policy_path,
        screening_base_dir=screening_base_dir,
        seeds=seeds,
        **remote_kwargs,
    )
    results: list[dict[str, Any]] = []
    for candidate_dir in sorted(screening_base_dir.iterdir()):
        if not candidate_dir.is_dir():
            continue
        gate = evaluate_candidate(compiled_policy, candidate_dir)
        write_gate_result(gate, candidate_dir / "rule_gate.json")
        judge = None if skip_judge else run_judge(compiled_policy, candidate_dir, llm_client)
        results.append(
            {
                "candidate_dir": candidate_dir,
                "gate": gate,
                "judge": judge,
                "combined_score": _candidate_score(gate, judge),
            }
        )
    return results


def _select_best(results: list[dict[str, Any]], min_judge_score: float) -> dict[str, Any] | None:
    if not results:
        return None
    passing = []
    for row in results:
        gate: GateResult = row["gate"]
        judge: JudgeResult | None = row["judge"]
        judge_ok = judge is None or judge.total_score >= min_judge_score
        if gate.passed and judge_ok:
            passing.append(row)
    pool = passing or results
    return max(pool, key=lambda row: row["combined_score"])


def _is_valid_final_candidate(row: dict[str, Any] | None, min_judge_score: float) -> bool:
    if row is None:
        return False
    gate: GateResult = row["gate"]
    judge: JudgeResult | None = row["judge"]
    judge_ok = judge is None or judge.total_score >= min_judge_score
    return bool(gate.passed and judge_ok)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _materialize_final(run_root: Path, best: dict[str, Any]) -> None:
    final_dir = run_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir: Path = best["candidate_dir"]
    (final_dir / "selected_candidate.txt").write_text(candidate_dir.name, encoding="utf-8")
    _copy_if_exists(candidate_dir / "infinigen_clean_scene.blend", final_dir / "scene.blend")
    _copy_if_exists(candidate_dir / "render.png", final_dir / "render.png")
    _copy_if_exists(candidate_dir / "placement.json", final_dir / "placement.json")
    _copy_if_exists(candidate_dir / "inventory_summary.json", final_dir / "inventory_summary.json")
    _copy_if_exists(candidate_dir / "rule_gate.json", final_dir / "rule_gate.json")
    _copy_if_exists(candidate_dir / "judge.json", final_dir / "judge.json")


def _write_run_status(run_root: Path, *, status: str, selected_candidate: str = "", reason: str = "") -> None:
    payload = {
        "status": status,
        "selected_candidate": selected_candidate,
        "reason": reason,
    }
    (run_root / "run_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-driven Infinigen scene generation pipeline")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--screening-seeds", default="0,1,2,3")
    parser.add_argument("--final-seeds", default="11,12")
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--llm-provider", default="none")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="gpt-oss:20b")
    parser.add_argument("--policies", default="config/scene_policies.yaml")
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-port", type=int, default=22)
    parser.add_argument("--remote-user", default=None)
    parser.add_argument("--remote-key", default=None)
    parser.add_argument("--remote-conda-env", default=None)
    parser.add_argument("--remote-infinigen-src", default="/workspace/infinigen/src")
    parser.add_argument("--infinigen-src", default=None)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    run_root = Path(args.out_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    llm_client = _build_llm_client(args.llm_provider, args.ollama_model, args.ollama_url)
    policies = load_policies(args.policies)

    intent = extract_intent(args.prompt, llm_client or StubLLMClient())
    save_intent_trace(intent, run_root)

    compiled_policy = compile_prompt_intent(intent, policies)
    compiled_policy.scene_id = run_root.name
    write_compiled_artifacts(compiled_policy, run_root)

    remote_kwargs = {
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "remote_user": args.remote_user,
        "remote_key": args.remote_key,
        "remote_conda_env": args.remote_conda_env,
        "remote_infinigen_src": args.remote_infinigen_src,
        "infinigen_src": args.infinigen_src,
    }
    screening_seeds = [int(part.strip()) for part in str(args.screening_seeds).split(",") if part.strip()]

    current_policy = compiled_policy
    best_row: dict[str, Any] | None = None
    for round_index in range(args.max_repair_rounds + 1):
        round_root = run_root if round_index == 0 else (run_root / f"repaired_round_{round_index}")
        round_root.mkdir(parents=True, exist_ok=True)
        if round_index > 0:
            current_policy.artifacts["compiled_policy"] = str((round_root / f"compiled_policy.repaired.v{round_index}.json").resolve())
            (round_root / f"compiled_policy.repaired.v{round_index}.json").write_text(
                current_policy.model_dump_json_pretty(),
                encoding="utf-8",
            )
            (run_root / "compiled_policy.active.json").write_text(current_policy.model_dump_json_pretty(), encoding="utf-8")
        screening_dir = round_root / "screening"
        results = _screen_and_score(
            current_policy,
            run_root,
            llm_client,
            screening_base_dir=screening_dir,
            seeds=screening_seeds,
            skip_judge=args.skip_judge,
            remote_kwargs=remote_kwargs,
        )
        best_row = _select_best(results, current_policy.acceptance_policy.min_judge_score)
        if best_row is None:
            break
        gate: GateResult = best_row["gate"]
        judge: JudgeResult | None = best_row["judge"]
        judge_ok = judge is None or judge.total_score >= current_policy.acceptance_policy.min_judge_score
        if gate.passed and judge_ok:
            break
        if args.skip_repair or round_index >= args.max_repair_rounds:
            break
        repair_plan = build_repair_plan(current_policy, gate, judge)
        (round_root / "repair_plan.json").write_text(repair_plan.model_dump_json_pretty(), encoding="utf-8")
        current_policy = apply_repair_plan(current_policy, repair_plan)

    if _is_valid_final_candidate(best_row, current_policy.acceptance_policy.min_judge_score):
        _materialize_final(run_root, best_row)
        _write_run_status(
            run_root,
            status="ok",
            selected_candidate=str(best_row["candidate_dir"].name),
        )
    else:
        _write_run_status(
            run_root,
            status="no_valid_candidate",
            selected_candidate=str(best_row["candidate_dir"].name) if best_row is not None else "",
            reason="all_screening_candidates_failed_hard_or_below_threshold",
        )
    build_scene_report(run_root)


if __name__ == "__main__":
    main()
