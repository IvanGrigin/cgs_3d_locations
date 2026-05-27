"""
Screening и финализация только для пайплайна llm_vlm_layout_refinement.
Не меняет run_infinigen_clean.run_from_compiled_policy: полный Namespace + Infinigen task собираются здесь.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from time import perf_counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.Plasement.run_infinigen_clean import run_local, run_remote
from src.prompt_compiler.compile_to_infinigen import build_room_json, build_style_profile
from src.prompt_compiler.llm_client import BaseLLMClient, OllamaJSONLLMClient, StubLLMClient
from src.prompt_compiler.policies import ScenePolicies
from src.prompt_compiler.schemas import CompiledPolicy, GateResult, JudgeResult
from src.scene_quality.quality_gate import evaluate_candidate, write_gate_result


_STALE_ARTIFACTS = (
    "early_failure.json",
    "run.log",
    "inventory.json",
    "inventory_summary.json",
    "solver_summary.json",
    "candidate_pool.json",
    "infinigen_clean_scene.blend",
    "infinigen_clean_meta.json",
    "placement.json",
    "render.png",
    "rule_gate.json",
    "judge_result.json",
    "infinigen_failure.json",
)


def _clean_stale_artifacts(out_dir: Path) -> list[str]:
    """Удаляет артефакты прошлых запусков в out_dir, чтобы при сетевом сбое
    мы не путали старый run.log/early_failure.json с реальной причиной нового падения.
    """
    removed: list[str] = []
    for name in _STALE_ARTIFACTS:
        p = out_dir / name
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                removed.append(name)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed.append(name + "/")
        except Exception:
            pass
    return removed


_SSH_NETWORK_RETCODES: tuple[int, ...] = (255,)


def _is_ssh_network_failure(exc: subprocess.CalledProcessError) -> bool:
    """Эвристика: returncode=255 от ssh — разрыв канала / сетевая ошибка,
    а не падение Infinigen. Имеет смысл повторить попытку."""
    return int(exc.returncode or 0) in _SSH_NETWORK_RETCODES


def _build_scp_command(
    *,
    remote_host: str,
    remote_user: str,
    remote_port: int | None,
    remote_key: str | None,
    remote_path: str,
    local_path: Path,
) -> list[str]:
    """Команда scp для тихого бинарного скачивания, минуя base64/TTY."""
    cmd = [
        "scp",
        "-q",
        "-B",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "TCPKeepAlive=yes",
        "-o", "StrictHostKeyChecking=no",
    ]
    if remote_port:
        cmd += ["-P", str(int(remote_port))]
    if remote_key:
        cmd += ["-i", str(Path(remote_key).expanduser())]
    cmd += [f"{remote_user}@{remote_host}:{remote_path}", str(local_path)]
    return cmd


def _ensure_blend_via_scp(
    out_dir: Path,
    *,
    remote_host: str | None,
    remote_user: str | None,
    remote_port: int | None,
    remote_key: str | None,
    log_run_root: Path | None,
    max_retries: int = 2,
    retry_sleep_sec: float = 3.0,
) -> bool:
    """Если infinigen_clean_scene.blend локально отсутствует или пуст —
    пытается забрать его через scp, используя путь из infinigen_clean_meta.json.

    Это компенсирует баг в run_infinigen_clean.py / maybe_download_remote_artifact,
    где большой бинарный blend качается через base64-pipe и теряется на vast.ai
    из-за TTY-banner-ов SSH (`Welcome to vast.ai...`), а ошибка тихо проглатывается.

    Returns True, если blend в итоге существует и не пустой; иначе False.
    """
    blend_path = out_dir / "infinigen_clean_scene.blend"
    if blend_path.is_file() and blend_path.stat().st_size > 0:
        return True
    meta_path = out_dir / "infinigen_clean_meta.json"
    if not meta_path.is_file():
        emit_llm_vlm_log(
            log_run_root,
            f"  scp-fallback for blend skipped in {out_dir.name}: "
            "infinigen_clean_meta.json absent",
        )
        return False
    if not (remote_host and remote_user):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        emit_llm_vlm_log(
            log_run_root,
            f"  scp-fallback for blend failed to parse meta in {out_dir.name}: {exc!r}",
        )
        return False
    remote_blend = str(meta.get("scene_blend") or "").strip()
    if not remote_blend:
        emit_llm_vlm_log(
            log_run_root,
            f"  scp-fallback for blend in {out_dir.name}: meta.scene_blend missing",
        )
        return False
    cmd = _build_scp_command(
        remote_host=remote_host,
        remote_user=remote_user,
        remote_port=remote_port,
        remote_key=remote_key,
        remote_path=remote_blend,
        local_path=blend_path,
    )
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, attempts + 1):
        emit_llm_vlm_log(
            log_run_root,
            f"  scp-fallback for blend in {out_dir.name} attempt={attempt}/{attempts}: "
            f"{' '.join(shlex.quote(c) for c in cmd[:1])} ... {remote_blend} -> {blend_path}",
        )
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            emit_llm_vlm_log(
                log_run_root,
                f"  scp-fallback for blend timed out in {out_dir.name} (10 min); retrying" if attempt < attempts
                else f"  scp-fallback for blend timed out and exhausted retries in {out_dir.name}",
            )
            if attempt < attempts:
                time.sleep(retry_sleep_sec)
                continue
            return False
        if completed.returncode == 0 and blend_path.is_file() and blend_path.stat().st_size > 0:
            emit_llm_vlm_log(
                log_run_root,
                f"  scp-fallback success in {out_dir.name}: blend size={blend_path.stat().st_size} bytes",
            )
            return True
        err = (completed.stderr or "").strip()[:400]
        emit_llm_vlm_log(
            log_run_root,
            f"  scp-fallback for blend failed in {out_dir.name} attempt={attempt}/{attempts}: "
            f"returncode={completed.returncode} stderr={err!r}",
        )
        try:
            if blend_path.is_file() and blend_path.stat().st_size == 0:
                blend_path.unlink()
        except Exception:
            pass
        if attempt < attempts:
            time.sleep(retry_sleep_sec)
    return False


def _apply_postfilter_to_inventory(
    out_dir: Path,
    compiled_policy: CompiledPolicy,
    *,
    log_run_root: Path | None = None,
) -> dict[str, Any] | None:
    """Удаляет из inventory.json объекты с factory_name из factory_blacklist и режет
    по max_counts (per-semantic), затем пересчитывает inventory_summary.json.
    Это компенсирует то, что Infinigen-сольвер игнорирует style_profile blacklist
    в decor/secondary стадиях (родная функция home_asset_usage патчится только
    в одном месте; populator-ы добавляют объекты независимо).

    Возвращает delta-сводку или None, если не было правок.
    """
    inv_path = out_dir / "inventory.json"
    summary_path = out_dir / "inventory_summary.json"
    if not inv_path.is_file() or not summary_path.is_file():
        return None
    blacklist = set(compiled_policy.program.factory_blacklist or [])
    max_counts = dict(compiled_policy.program.max_counts or {})
    if not blacklist and not max_counts:
        return None
    try:
        items = json.loads(inv_path.read_text(encoding="utf-8"))
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        emit_llm_vlm_log(log_run_root, f"  postfilter inventory parse failed in {out_dir.name}: {exc!r}")
        return None
    if not isinstance(items, list):
        return None

    def _f(item: Any) -> str:
        return str((item or {}).get("factory_name") or (item or {}).get("factory") or "")

    def _s(item: Any) -> str:
        return str((item or {}).get("semantic") or "")

    removed_blacklist: list[dict[str, Any]] = []
    after_blacklist: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if blacklist and _f(it) in blacklist:
            removed_blacklist.append(it)
        else:
            after_blacklist.append(it)

    # Если у фабрики semantic шире нашей разговорной категории, в max_counts может стоять
    # ограничение на узкое имя (FloorLamp/CeilingLight/...) — учитываем это маппингом.
    _FACTORY_TO_CAP_KEY: dict[str, str] = {
        "FloorLampFactory": "FloorLamp",
        "CeilingLightFactory": "CeilingLight",
        "TableLampFactory": "TableLamp",
        "WallLampFactory": "WallLamp",
        "DesignerLampFactory": "FloorLamp",
        "FloorTowerLampFactory": "FloorLamp",
    }

    def _effective_cap(it: dict[str, Any], sem: str) -> tuple[int | None, str]:
        """Возвращает (cap, count_key) для max_counts. cap is None — лимита нет."""
        factory_cap_key = _FACTORY_TO_CAP_KEY.get(_f(it))
        if factory_cap_key and max_counts.get(factory_cap_key) is not None:
            return int(max_counts[factory_cap_key]), factory_cap_key
        if sem and max_counts.get(sem) is not None:
            return int(max_counts[sem]), sem
        return None, sem

    removed_overflow: list[dict[str, Any]] = []
    if max_counts:
        seen: dict[str, int] = {}
        kept: list[dict[str, Any]] = []
        for it in after_blacklist:
            sem = _s(it)
            cap, count_key = _effective_cap(it, sem)
            if cap is None:
                kept.append(it)
                continue
            if int(seen.get(count_key, 0)) < int(cap):
                seen[count_key] = int(seen.get(count_key, 0)) + 1
                kept.append(it)
            else:
                removed_overflow.append(it)
    else:
        kept = list(after_blacklist)

    if not removed_blacklist and not removed_overflow:
        return None

    factory_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    factory_to_semantic: dict[str, str] = {}
    for it in kept:
        f = _f(it)
        s = _s(it)
        if f:
            factory_counts[f] = int(factory_counts.get(f, 0)) + 1
            if s and f not in factory_to_semantic:
                factory_to_semantic[f] = s
        if s:
            semantic_counts[s] = int(semantic_counts.get(s, 0)) + 1

    old_core_factory_keys = set((old_summary.get("core_factory_counts") or {}).keys())
    core_factory_counts = {f: c for f, c in factory_counts.items() if f in old_core_factory_keys}
    core_semantic_counts: dict[str, int] = {}
    for f, c in core_factory_counts.items():
        s = factory_to_semantic.get(f)
        if s:
            core_semantic_counts[s] = int(core_semantic_counts.get(s, 0)) + int(c)

    removed_in_core_blacklist = sum(1 for it in removed_blacklist if _f(it) in old_core_factory_keys)
    removed_in_core_overflow = sum(1 for it in removed_overflow if _f(it) in old_core_factory_keys)
    old_real = int(old_summary.get("real_object_count", 0) or 0)
    new_real = max(0, old_real - removed_in_core_blacklist - removed_in_core_overflow)

    new_summary = dict(old_summary)
    new_summary["raw_real_object_count"] = len(kept)
    new_summary["real_object_count"] = int(new_real)
    new_summary["factory_counts"] = factory_counts
    new_summary["semantic_counts"] = semantic_counts
    new_summary["core_factory_counts"] = core_factory_counts
    new_summary["core_semantic_counts"] = core_semantic_counts
    new_summary["postfilter"] = {
        "removed_blacklist_count": len(removed_blacklist),
        "removed_overflow_count": len(removed_overflow),
        "removed_blacklist_factories": sorted({_f(it) for it in removed_blacklist}),
        "removed_overflow_semantics": sorted({_s(it) for it in removed_overflow if _s(it)}),
    }

    try:
        inv_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(new_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        emit_llm_vlm_log(log_run_root, f"  postfilter write failed in {out_dir.name}: {exc!r}")
        return None

    emit_llm_vlm_log(
        log_run_root,
        f"  postfilter applied in {out_dir.name}: "
        f"-{len(removed_blacklist)} blacklist {sorted({_f(it) for it in removed_blacklist})!r}, "
        f"-{len(removed_overflow)} overflow {sorted({_s(it) for it in removed_overflow if _s(it)})!r}; "
        f"raw_real_object_count {old_summary.get('raw_real_object_count')} -> {len(kept)}, "
        f"real_object_count {old_real} -> {new_real}",
    )
    return new_summary["postfilter"]


def emit_llm_vlm_log(run_root: Path | None, message: str) -> None:
    """Печать в stdout и дозапись в run_root/llm_vlm_run.log (если run_root задан)."""
    line = f"[llm_vlm] {message}"
    print(line, flush=True)
    if run_root is not None:
        p = Path(run_root).expanduser().resolve() / "llm_vlm_run.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def truncate_for_log(obj: Any, max_len: int = 1600) -> str:
    if isinstance(obj, str):
        s = obj
    else:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) <= max_len:
        return s
    return s[: max_len - 24] + "\n… [truncated] …"


def format_timing_dur(dt: float) -> str:
    """Длительность для [timing]-логов: меньше 1 с — миллисекунды, иначе секунды (3 знака)."""
    x = max(0.0, float(dt))
    if x < 1.0:
        return f"{x * 1000.0:.1f}ms"
    return f"{x:.3f}s"


def default_policies_path() -> Path:
    return Path(__file__).resolve().parent / "llm_vlm_scene_policies.yaml"


def load_policies_llm_vlm(path: str | Path | None = None) -> ScenePolicies:
    p = Path(path).expanduser().resolve() if path else default_policies_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return ScenePolicies.model_validate(data)


def build_llm_client(provider: str, model: str, base_url: str) -> BaseLLMClient | None:
    if provider == "none":
        return None
    if provider == "ollama":
        return OllamaJSONLLMClient(base_url=base_url, model=model)
    raise ValueError(f"unsupported llm provider: {provider}")


_PLACEMENT_DIR = Path(__file__).resolve().parent.parent / "Plasement"
_BUILDER_SCRIPT = _PLACEMENT_DIR / "blender_scene_builder.py"


def run_vlm_fast_preview_render(
    candidate_dir: str | Path,
    *,
    blender_executable: str,
    resolution_pct: int = 42,
    render_engine: str = "eevee",
    log_run_root: Path | None = None,
    timeout_sec: int = 900,
) -> bool:
    """Локальный headless-рендер `render.png` из placement + infinigen blend (для VLM).

    Использует EEVEE или облегчённый Cycles, пониженное ``resolution_percentage`` —
    заметно быстрее полноразмерного финального рендера.
    """
    out_dir = Path(candidate_dir).expanduser().resolve()
    placement = out_dir / "placement.json"
    blend = out_dir / "infinigen_clean_scene.blend"
    out_png = out_dir / "render.png"
    if not placement.is_file() or not blend.is_file():
        emit_llm_vlm_log(
            log_run_root,
            f"vlm_fast_render skip in {out_dir.name}: missing placement.json or infinigen_clean_scene.blend",
        )
        return False
    if not _BUILDER_SCRIPT.is_file():
        emit_llm_vlm_log(log_run_root, f"vlm_fast_render skip: builder script missing {_BUILDER_SCRIPT}")
        return False

    eng = str(render_engine or "eevee").strip().lower()
    if eng not in ("eevee", "fast_cycles"):
        eng = "eevee"
    pct = max(5, min(100, int(resolution_pct)))

    project_root = str(_PLACEMENT_DIR.parent.resolve())
    cmd = [
        blender_executable,
        str(blend.resolve()),
        "-b",
        "--python",
        str(_BUILDER_SCRIPT.resolve()),
        "--",
        "--json",
        str(placement.resolve()),
        "--project-root",
        project_root,
        "--reference-blend",
        str(blend.resolve()),
        "--render",
        str(out_png.resolve()),
        "--render-resolution-pct",
        str(pct),
        "--render-engine",
        eng,
    ]
    emit_llm_vlm_log(
        log_run_root,
        f"vlm_fast_render: blender render_engine={eng} resolution_pct={pct} -> {out_png.name}",
    )
    t0 = perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired:
        emit_llm_vlm_log(
            log_run_root,
            f"[timing] vlm_fast_render wall={format_timing_dur(perf_counter() - t0)} (timeout {timeout_sec}s) in {out_dir.name}",
        )
        emit_llm_vlm_log(log_run_root, f"vlm_fast_render timeout after {timeout_sec}s in {out_dir.name}")
        try:
            if out_png.is_file():
                out_png.unlink()
        except Exception:
            pass
        return False
    except Exception as exc:
        emit_llm_vlm_log(
            log_run_root,
            f"[timing] vlm_fast_render wall={format_timing_dur(perf_counter() - t0)} (spawn error) in {out_dir.name}",
        )
        emit_llm_vlm_log(log_run_root, f"vlm_fast_render failed to spawn blender in {out_dir.name}: {exc!r}")
        return False

    wall = perf_counter() - t0
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-1200:]
        emit_llm_vlm_log(log_run_root, f"[timing] vlm_fast_render wall={format_timing_dur(wall)} exit={completed.returncode}")
        emit_llm_vlm_log(
            log_run_root,
            f"vlm_fast_render blender exit={completed.returncode} in {out_dir.name} tail={tail!r}",
        )
        return False
    if not out_png.is_file() or out_png.stat().st_size < 32:
        emit_llm_vlm_log(log_run_root, f"[timing] vlm_fast_render wall={format_timing_dur(wall)} (no PNG)")
        emit_llm_vlm_log(log_run_root, f"vlm_fast_render produced no usable PNG in {out_dir.name}")
        return False
    emit_llm_vlm_log(log_run_root, f"[timing] vlm_fast_render wall={format_timing_dur(wall)} OK")
    emit_llm_vlm_log(
        log_run_root,
        f"vlm_fast_render OK {out_dir.name}/{out_png.name} size={out_png.stat().st_size}",
    )
    return True


def run_one_candidate_from_compiled(
    compiled_policy_path: str | Path,
    output_dir: str | Path,
    seed: int,
    *,
    remote_host: str | None = None,
    remote_port: int = 22,
    remote_user: str | None = None,
    remote_key: str | None = None,
    remote_conda_env: str | None = None,
    remote_infinigen_src: str = "/workspace/infinigen/src",
    infinigen_src: str | None = None,
    infinigen_task: str = "coarse",
    infinigen_configs: list[str] | None = None,
    log_run_root: Path | None = None,
    ssh_max_retries: int = 2,
    ssh_retry_sleep_sec: float = 5.0,
) -> dict[str, Any]:
    t_wall0 = perf_counter()
    compiled_policy = CompiledPolicy.load(compiled_policy_path)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    removed = _clean_stale_artifacts(out_dir)
    if removed:
        emit_llm_vlm_log(
            log_run_root,
            f"  cleaned stale artifacts in {out_dir.name}: {removed}",
        )

    room_json_path = out_dir / "room.json"
    style_profile_path = out_dir / "style_profile.json"
    placement_path = out_dir / "placement.json"
    room_json_path.write_text(json.dumps(build_room_json(compiled_policy), ensure_ascii=False, indent=2), encoding="utf-8")
    style_profile_path.write_text(json.dumps(build_style_profile(compiled_policy), ensure_ascii=False, indent=2), encoding="utf-8")
    cfgs = list(infinigen_configs or ["singleroom.gin", "fast_solve.gin"])
    args = argparse.Namespace(
        room=str(room_json_path),
        seed=str(seed),
        out=str(placement_path),
        run_dir=str(out_dir),
        style_profile=str(style_profile_path),
        infinigen_src=infinigen_src,
        remote_host=remote_host,
        remote_port=remote_port,
        remote_user=remote_user,
        remote_key=remote_key,
        remote_conda_env=remote_conda_env,
        remote_infinigen_src=remote_infinigen_src,
        infinigen_task=infinigen_task,
        infinigen_configs=cfgs,
    )
    error: str | None = None
    network_attempts = 0
    max_attempts = max(1, int(ssh_max_retries) + 1)
    t_inf0 = perf_counter()
    while True:
        network_attempts += 1
        try:
            if remote_host and remote_user:
                run_remote(args)
            else:
                run_local(args)
            error = None
            break
        except subprocess.CalledProcessError as exc:
            if remote_host and _is_ssh_network_failure(exc) and network_attempts < max_attempts:
                emit_llm_vlm_log(
                    log_run_root,
                    f"  ssh network failure (returncode={exc.returncode}) attempt={network_attempts}/{max_attempts}; "
                    f"sleep {ssh_retry_sleep_sec:.1f}s and retry",
                )
                _clean_stale_artifacts(out_dir)
                time.sleep(max(0.0, float(ssh_retry_sleep_sec)))
                continue
            error = f"infinigen_subprocess_failed: returncode={exc.returncode}"
            break
        except Exception as exc:
            error = f"infinigen_failed: {exc!r}"
            break
    dt_infinigen = perf_counter() - t_inf0

    inv_path = out_dir / "inventory.json"
    inv_sum_path = out_dir / "inventory_summary.json"

    t_scp0 = perf_counter()
    if error is None and remote_host and remote_user:
        try:
            _ensure_blend_via_scp(
                out_dir,
                remote_host=remote_host,
                remote_user=remote_user,
                remote_port=remote_port,
                remote_key=remote_key,
                log_run_root=log_run_root,
            )
        except Exception as exc:
            emit_llm_vlm_log(log_run_root, f"  scp-fallback unexpected error in {out_dir.name}: {exc!r}")
    dt_scp = perf_counter() - t_scp0 if (remote_host and remote_user) else 0.0

    t_post0 = perf_counter()
    if error is None and inv_path.is_file() and inv_sum_path.is_file():
        try:
            _apply_postfilter_to_inventory(out_dir, compiled_policy, log_run_root=log_run_root)
        except Exception as exc:
            emit_llm_vlm_log(log_run_root, f"  postfilter unexpected error in {out_dir.name}: {exc!r}")
    dt_post = perf_counter() - t_post0

    if not inv_path.is_file():
        inv_path.write_text("[]", encoding="utf-8")
    if not inv_sum_path.is_file():
        inv_sum_path.write_text(
            json.dumps(
                {
                    "raw_real_object_count": 0,
                    "real_object_count": 0,
                    "factory_counts": {},
                    "semantic_counts": {},
                    "core_factory_counts": {},
                    "core_semantic_counts": {},
                    "infinigen_run_failed": bool(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if error is not None:
        (out_dir / "infinigen_failure.json").write_text(
            json.dumps(
                {
                    "error": error,
                    "seed": int(seed),
                    "ssh_attempts": network_attempts,
                    "remote_host": remote_host,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    early_failure: dict[str, Any] | None = None
    early_path = out_dir / "early_failure.json"
    if error is not None and early_path.is_file():
        try:
            early_path.unlink()
            emit_llm_vlm_log(
                log_run_root,
                "  removed early_failure.json after SSH/Infinigen subprocess failure: "
                "файл может быть от ПРЕДЫДУЩЕГО запуска (артефакты не успели скачаться).",
            )
        except Exception:
            pass
    if early_path.is_file():
        try:
            early_failure = json.loads(early_path.read_text(encoding="utf-8"))
        except Exception:
            early_failure = {"raw": early_path.read_text(encoding="utf-8", errors="replace")}
    dt_total = perf_counter() - t_wall0
    emit_llm_vlm_log(
        log_run_root,
        f"[timing] {out_dir.name} seed={seed} infinigen+ssh={format_timing_dur(dt_infinigen)} "
        f"scp_blend={format_timing_dur(dt_scp)} postfilter={format_timing_dur(dt_post)} total={format_timing_dur(dt_total)} err={error!r}",
    )
    return {
        "candidate_dir": str(out_dir),
        "placement": str(placement_path),
        "inventory": str(inv_path),
        "inventory_summary": str(inv_sum_path),
        "solver_summary": str(out_dir / "solver_summary.json"),
        "blend": str(out_dir / "infinigen_clean_scene.blend"),
        "run_log": str(out_dir / "run.log"),
        "infinigen_error": error,
        "ssh_attempts": network_attempts,
        "early_failure": early_failure,
    }


def run_screening_llm_vlm(
    compiled_policy_path: str | Path,
    screening_base_dir: str | Path,
    seeds: list[int],
    *,
    log_run_root: Path | None = None,
    ssh_max_retries: int = 2,
    ssh_retry_sleep_sec: float = 5.0,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    base_dir = Path(screening_base_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    emit_llm_vlm_log(
        log_run_root,
        f"screening start dir={base_dir} seeds={seeds} remote_host={kwargs.get('remote_host')!r} "
        f"infinigen_task={kwargs.get('infinigen_task')!r} infinigen_configs={kwargs.get('infinigen_configs')!r} "
        f"ssh_max_retries={ssh_max_retries}",
    )
    t_screen0 = perf_counter()
    results: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        candidate_dir = base_dir / f"seed_{index:03d}"
        emit_llm_vlm_log(log_run_root, f"  seed[{index}]={seed} -> {candidate_dir.name}")
        result = run_one_candidate_from_compiled(
            compiled_policy_path=compiled_policy_path,
            output_dir=candidate_dir,
            seed=int(seed),
            log_run_root=log_run_root,
            ssh_max_retries=ssh_max_retries,
            ssh_retry_sleep_sec=ssh_retry_sleep_sec,
            **kwargs,
        )
        if result.get("infinigen_error"):
            attempts = int(result.get("ssh_attempts") or 1)
            err_text = str(result["infinigen_error"])
            is_ssh_net = "returncode=255" in err_text
            is_disk_full = "REMOTE_DISK_FULL" in err_text
            emit_llm_vlm_log(
                log_run_root,
                f"  seed[{index}]={seed} infinigen_error={err_text!r} ssh_attempts={attempts} "
                f"{'[SSH network failure, не Infinigen!] ' if is_ssh_net else ''}"
                f"{'[REMOTE_DISK_FULL — не запускали Infinigen!] ' if is_disk_full else ''}"
                "(продолжаем со следующим seed)",
            )
            if is_disk_full:
                rh = kwargs.get("remote_host")
                ru = kwargs.get("remote_user")
                rp = kwargs.get("remote_port") or 22
                rk = kwargs.get("remote_key")
                ssh_cmd = "ssh"
                if rk:
                    ssh_cmd += f" -i {rk}"
                if rp and int(rp) != 22:
                    ssh_cmd += f" -p {rp}"
                ssh_cmd += f" {ru}@{rh}"
                emit_llm_vlm_log(
                    log_run_root,
                    "  hint: на удалённом хосте закончилось место. Очистите старые tmp-каталоги:\n"
                    f"    {ssh_cmd} 'du -sh /workspace/tmp/* 2>/dev/null | sort -h | tail -20'\n"
                    f"    {ssh_cmd} 'rm -rf /workspace/tmp/infinigen_clean_*'\n"
                    f"    {ssh_cmd} 'df -h /workspace'\n"
                    "  И перезапустите CLI. Можно также увеличить диск контейнера на vast.ai.",
                )
            elif is_ssh_net:
                emit_llm_vlm_log(
                    log_run_root,
                    "  hint: добавьте в ~/.ssh/config для этого хоста ControlMaster auto, "
                    "ControlPersist 10m, ControlPath ~/.ssh/cm-%r@%h:%p, ServerAliveInterval 30, "
                    "ServerAliveCountMax 6 — это резко уменьшит SSH-разрывы и ускорит работу.",
                )
            else:
                _dump_infinigen_failure_excerpt(candidate_dir, log_run_root)
        results.append(result)
    emit_llm_vlm_log(
        log_run_root,
        f"[timing] screening dir={base_dir.name} seeds_n={len(seeds)} wall={format_timing_dur(perf_counter() - t_screen0)}",
    )
    return results


def _dump_infinigen_failure_excerpt(candidate_dir: Path, log_run_root: Path | None) -> None:
    """Печатает в наш лог хвост run.log и содержимое early_failure.json/solver_summary.json,
    чтобы было видно реальную причину падения Infinigen."""
    early = candidate_dir / "early_failure.json"
    if early.is_file():
        try:
            payload = json.loads(early.read_text(encoding="utf-8"))
            emit_llm_vlm_log(
                log_run_root,
                f"  early_failure.json: {truncate_for_log(payload, 1200)}",
            )
        except Exception as exc:
            emit_llm_vlm_log(log_run_root, f"  early_failure.json read failed: {exc!r}")
    solver = candidate_dir / "solver_summary.json"
    if solver.is_file():
        try:
            payload = json.loads(solver.read_text(encoding="utf-8"))
            errors = payload.get("errors") or []
            if errors:
                emit_llm_vlm_log(
                    log_run_root,
                    f"  solver_summary.errors: {truncate_for_log(errors, 1500)}",
                )
        except Exception as exc:
            emit_llm_vlm_log(log_run_root, f"  solver_summary.json read failed: {exc!r}")
    log_path = candidate_dir / "run.log"
    if log_path.is_file():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail_lines = text.splitlines()[-60:]
            emit_llm_vlm_log(
                log_run_root,
                "  run.log tail:\n" + truncate_for_log("\n".join(tail_lines), 3500),
            )
        except Exception as exc:
            emit_llm_vlm_log(log_run_root, f"  run.log read failed: {exc!r}")


def candidate_score(gate: GateResult, judge: JudgeResult | None) -> float:
    if judge is None:
        return gate.rule_score
    return gate.rule_score * 0.55 + judge.total_score * 0.45


def _llm_vlm_judge_dir() -> Path:
    return Path(__file__).resolve().parent / "llm_vlm_judge"


def _image_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    with Image.open(path) as image:
        return {"exists": True, "width": image.width, "height": image.height, "mode": image.mode}


def _heuristic_judge(compiled_policy: CompiledPolicy, gate_result: GateResult, candidate_dir: Path) -> JudgeResult:
    inv = gate_result.inventory_summary or {}
    real_object_count = int(inv.get("real_object_count", 0) or 0)
    core_sem = dict(inv.get("core_semantic_counts") or {})
    required = list(compiled_policy.program.required_semantics or [])
    required_present = [s for s in required if int(core_sem.get(s, 0) or 0) > 0]
    required_missing = [s for s in required if int(core_sem.get(s, 0) or 0) <= 0]

    n_hard = len(gate_result.hard_failures or [])
    n_soft = len(gate_result.soft_failures or [])

    if real_object_count == 0:
        functionality = 0.0
    elif required_missing:
        functionality = max(0.5, 4.0 - 1.0 * len(required_missing))
    elif n_hard == 0:
        functionality = 7.5
    else:
        functionality = max(3.0, 6.5 - 0.5 * n_hard)

    if real_object_count == 0:
        prompt_match = 0.0
    elif required_missing:
        prompt_match = max(0.5, 3.5 - 1.0 * len(required_missing))
    else:
        prompt_match = max(2.0, 7.5 - 0.4 * n_hard - 0.2 * n_soft)

    style_drop = 0.0
    style_drop += 1.0 * sum(1 for f in gate_result.hard_failures if str(f).startswith("forbidden_factory:"))
    style_drop += 0.5 * sum(1 for f in gate_result.soft_failures if str(f).startswith("count_overflow:"))
    style_match = max(0.0, min(10.0, 7.5 - style_drop)) if real_object_count > 0 else 0.0

    composition = (
        0.0
        if real_object_count == 0
        else max(1.0, min(10.0, 7.0 - 0.3 * n_hard - 0.2 * n_soft))
    )

    if real_object_count > 0 and not required_missing:
        weights = (0.30, 0.25, 0.25, 0.20)
        total_score = (
            functionality * weights[0]
            + prompt_match * weights[1]
            + style_match * weights[2]
            + composition * weights[3]
        )
    else:
        total_score = (functionality + prompt_match + style_match + composition) / 4.0

    strengths: list[str] = []
    if required_present:
        strengths.append(f"required semantics present: {', '.join(required_present)}")
    if real_object_count > 0:
        strengths.append(f"non-empty scene with {real_object_count} real objects")

    weaknesses: list[str] = list(gate_result.hard_failures or []) + list(gate_result.soft_failures or [])
    if required_missing:
        weaknesses.insert(0, f"missing required semantics: {', '.join(required_missing)}")

    notes_bits: list[str] = []
    if real_object_count == 0:
        notes_bits.append("scene is empty; nothing to judge.")
    if required_missing:
        notes_bits.append(f"missing: {', '.join(required_missing)}.")
    if n_hard:
        notes_bits.append(f"{n_hard} hard rule_gate failures (forbidden factories or critical issues).")
    if n_soft:
        notes_bits.append(f"{n_soft} soft failures (mostly count overflows).")
    notes = " ".join(notes_bits) or "heuristic judge (llm_vlm bundle); scene OK by rule_gate."

    return JudgeResult(
        passed=(
            real_object_count > 0
            and not required_missing
            and not gate_result.hard_failures
            and total_score >= compiled_policy.acceptance_policy.min_judge_score
        ),
        total_score=round(total_score, 3),
        functionality_score=round(functionality, 3),
        prompt_match_score=round(prompt_match, 3),
        style_match_score=round(style_match, 3),
        composition_score=round(composition, 3),
        strengths=strengths,
        weaknesses=weaknesses,
        notes=notes,
        candidate_dir=str(candidate_dir),
        diagnostic_only=bool(gate_result.hard_failures),
    )


def run_judge_llm_vlm(
    compiled_policy: CompiledPolicy,
    candidate_dir: str | Path,
    llm_client: BaseLLMClient | None,
    *,
    log_run_root: Path | None = None,
) -> JudgeResult:
    """Как scene_quality.judge_runner.run_judge, но промпт/рубрика из src/pipeline/llm_vlm_judge/."""
    t_j0 = perf_counter()
    candidate_path = Path(candidate_dir).expanduser().resolve()
    gate_result = GateResult.load(candidate_path / "rule_gate.json")
    jdir = _llm_vlm_judge_dir()
    prompt_path = jdir / "room_judge_prompt.md"
    rubric_path = jdir / "room_judge_rubric.yaml"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    render_info = _image_summary(candidate_path / "render.png")
    user_prompt = json.dumps(
        {
            "original_prompt": compiled_policy.prompt_text,
            "room_type": compiled_policy.geometry.room_type.value,
            "area_sqm": compiled_policy.geometry.area_sqm,
            "area_bucket": compiled_policy.geometry.area_bucket.value,
            "style_label": compiled_policy.style_policy.style_label,
            "required_semantics": compiled_policy.program.required_semantics,
            "inventory_summary": gate_result.inventory_summary,
            "rule_gate": gate_result.model_dump(mode="json"),
            "render": render_info,
            "rubric": rubric,
        },
        ensure_ascii=False,
        indent=2,
    )
    emit_llm_vlm_log(
        log_run_root,
        f"judge candidate={candidate_path.name} gate_passed={gate_result.passed} rule_score={gate_result.rule_score:.3f} "
        f"hard={gate_result.hard_failures!r} soft_n={len(gate_result.soft_failures)} render={render_info}",
    )
    emit_llm_vlm_log(
        log_run_root,
        f"judge LLM input (truncated):\n{truncate_for_log(user_prompt, 2400)}",
    )
    schema = {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "total_score": {"type": "number"},
            "functionality_score": {"type": "number"},
            "prompt_match_score": {"type": "number"},
            "style_match_score": {"type": "number"},
            "composition_score": {"type": "number"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": [
            "passed",
            "total_score",
            "functionality_score",
            "prompt_match_score",
            "style_match_score",
            "composition_score",
            "strengths",
            "weaknesses",
            "notes",
        ],
        "additionalProperties": False,
    }
    if llm_client is None:
        emit_llm_vlm_log(log_run_root, f"judge mode=heuristic (no LLM client) candidate={candidate_path.name}")
        result = _heuristic_judge(compiled_policy, gate_result, candidate_path)
    else:
        try:
            emit_llm_vlm_log(log_run_root, f"judge mode=LLM candidate={candidate_path.name}")
            t_llm0 = perf_counter()
            raw = llm_client.complete_json(prompt_template, user_prompt, schema)
            emit_llm_vlm_log(
                log_run_root,
                f"[timing] judge_llm complete_json wall={format_timing_dur(perf_counter() - t_llm0)} candidate={candidate_path.name}",
            )
            emit_llm_vlm_log(
                log_run_root,
                f"judge LLM raw (truncated):\n{truncate_for_log(raw, 1200)}",
            )
            result = JudgeResult(
                passed=bool(raw.get("passed", False)),
                total_score=float(raw.get("total_score", 0.0)),
                functionality_score=float(raw.get("functionality_score", 0.0)),
                prompt_match_score=float(raw.get("prompt_match_score", 0.0)),
                style_match_score=float(raw.get("style_match_score", 0.0)),
                composition_score=float(raw.get("composition_score", 0.0)),
                strengths=list(raw.get("strengths") or []),
                weaknesses=list(raw.get("weaknesses") or []),
                notes=str(raw.get("notes") or ""),
                candidate_dir=str(candidate_path),
                diagnostic_only=bool(gate_result.hard_failures),
            )
            real_objs = int((gate_result.inventory_summary or {}).get("real_object_count", 0) or 0)
            all_zero = (
                result.total_score == 0.0
                and result.functionality_score == 0.0
                and result.prompt_match_score == 0.0
                and result.style_match_score == 0.0
                and result.composition_score == 0.0
            )
            empty_payload = (
                not result.notes
                and not result.strengths
                and not result.weaknesses
            )
            if all_zero and (real_objs > 0 or empty_payload):
                emit_llm_vlm_log(
                    log_run_root,
                    f"judge LLM degenerate response (all zeros, empty notes/strengths/weaknesses) "
                    f"for non-empty scene (real_object_count={real_objs}); falling back to heuristic. "
                    f"raw={truncate_for_log(raw, 600)}",
                )
                result = _heuristic_judge(compiled_policy, gate_result, candidate_path)
            else:
                emit_llm_vlm_log(
                    log_run_root,
                    f"judge LLM result total={result.total_score} func={result.functionality_score} "
                    f"prompt={result.prompt_match_score} style={result.style_match_score} "
                    f"comp={result.composition_score} passed={result.passed} "
                    f"strengths_n={len(result.strengths)} weaknesses_n={len(result.weaknesses)} "
                    f"notes={truncate_for_log(result.notes, 400)}",
                )
        except Exception as exc:
            emit_llm_vlm_log(log_run_root, f"judge LLM failed -> heuristic: {exc!r}")
            result = _heuristic_judge(compiled_policy, gate_result, candidate_path)
            emit_llm_vlm_log(
                log_run_root,
                f"judge heuristic result total={result.total_score} passed={result.passed}",
            )
    emit_llm_vlm_log(
        log_run_root,
        f"[timing] judge total wall={format_timing_dur(perf_counter() - t_j0)} candidate={candidate_path.name}",
    )
    emit_llm_vlm_log(
        log_run_root,
        f"judge saved candidate={candidate_path.name} total={result.total_score:.3f} passed={result.passed}",
    )
    result.save(candidate_path / "judge.json")
    return result


def screen_and_score(
    compiled_policy: CompiledPolicy,
    run_root: Path,
    llm_client: BaseLLMClient | None,
    *,
    screening_base_dir: Path,
    seeds: list[int],
    skip_judge: bool,
    remote_kwargs: dict[str, Any],
    judge_llm_client: BaseLLMClient | None = None,
) -> list[dict[str, Any]]:
    compiled_policy_path = run_root / "compiled_policy.active.json"
    emit_llm_vlm_log(run_root, f"screen_and_score policy={compiled_policy_path} screening={screening_base_dir}")
    emit_llm_vlm_log(
        run_root,
        f"compiled effective: required={compiled_policy.program.required_semantics!r} "
        f"factory_whitelist={truncate_for_log(compiled_policy.program.factory_whitelist, 600)} "
        f"factory_blacklist={truncate_for_log(compiled_policy.program.factory_blacklist, 600)} "
        f"max_counts={truncate_for_log(compiled_policy.program.max_counts, 600)}",
    )
    t_ss0 = perf_counter()
    expected_seed_names = {f"seed_{i:03d}" for i in range(len(seeds))}
    if screening_base_dir.is_dir():
        stale_seed_dirs: list[Path] = []
        for child in screening_base_dir.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith("seed_"):
                continue
            if child.name not in expected_seed_names:
                stale_seed_dirs.append(child)
        for stale in stale_seed_dirs:
            shutil.rmtree(stale, ignore_errors=True)
        if stale_seed_dirs:
            emit_llm_vlm_log(
                run_root,
                f"  removed stale seed dirs from previous run (not in current seeds): "
                f"{sorted(d.name for d in stale_seed_dirs)}",
            )
    run_screening_llm_vlm(
        compiled_policy_path=compiled_policy_path,
        screening_base_dir=screening_base_dir,
        seeds=seeds,
        log_run_root=run_root,
        **remote_kwargs,
    )
    t_after_screen = perf_counter()
    results: list[dict[str, Any]] = []
    for candidate_dir in sorted(screening_base_dir.iterdir()):
        if not candidate_dir.is_dir():
            continue
        if candidate_dir.name not in expected_seed_names:
            continue
        gate = evaluate_candidate(compiled_policy, candidate_dir)
        write_gate_result(gate, candidate_dir / "rule_gate.json")
        judge_client = judge_llm_client if judge_llm_client is not None else llm_client
        judge = None if skip_judge else run_judge_llm_vlm(compiled_policy, candidate_dir, judge_client, log_run_root=run_root)
        comb = candidate_score(gate, judge)
        if skip_judge:
            emit_llm_vlm_log(
                run_root,
                f"  scored {candidate_dir.name} gate_passed={gate.passed} rule={gate.rule_score:.3f} judge=skipped combined={comb:.3f}",
            )
        else:
            assert judge is not None
            emit_llm_vlm_log(
                run_root,
                f"  scored {candidate_dir.name} gate_passed={gate.passed} rule={gate.rule_score:.3f} "
                f"judge_total={judge.total_score:.3f} combined={comb:.3f}",
            )
        results.append(
            {
                "candidate_dir": candidate_dir,
                "gate": gate,
                "judge": judge,
                "combined_score": comb,
            }
        )
    dt_gate = perf_counter() - t_after_screen
    dt_all = perf_counter() - t_ss0
    emit_llm_vlm_log(
        run_root,
        f"[timing] screen_and_score screening_phase={format_timing_dur(t_after_screen - t_ss0)} "
        f"gate+judge_n={len(results)}={format_timing_dur(dt_gate)} total={format_timing_dur(dt_all)} dir={screening_base_dir.name}",
    )
    return results


def select_best(results: list[dict[str, Any]], min_judge_score: float) -> dict[str, Any] | None:
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


def is_valid_final_candidate(row: dict[str, Any] | None, min_judge_score: float) -> bool:
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


def materialize_final(run_root: Path, best: dict[str, Any]) -> None:
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


def write_run_status(run_root: Path, *, status: str, selected_candidate: str = "", reason: str = "") -> None:
    payload = {
        "status": status,
        "selected_candidate": selected_candidate,
        "reason": reason,
    }
    (run_root / "run_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "StubLLMClient",
    "build_llm_client",
    "candidate_score",
    "default_policies_path",
    "emit_llm_vlm_log",
    "format_timing_dur",
    "truncate_for_log",
    "is_valid_final_candidate",
    "load_policies_llm_vlm",
    "materialize_final",
    "run_judge_llm_vlm",
    "run_one_candidate_from_compiled",
    "run_screening_llm_vlm",
    "run_vlm_fast_preview_render",
    "screen_and_score",
    "select_best",
    "write_run_status",
]
