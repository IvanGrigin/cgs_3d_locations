#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _fmt_sec(sec: float | int | None) -> str:
    if sec is None:
        return "?:??:??"  # pragma: no cover
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class StageTimer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.monotonic()
        self.last = self.t0
        self.stages: list[dict[str, Any]] = []

    def mark(self, name: str, **extra: Any) -> None:
        now = time.monotonic()
        dt = now - self.last
        total = now - self.t0
        self.last = now
        row = {"stage": name, "dt_sec": round(dt, 3), "elapsed_sec": round(total, 3)}
        row.update(extra)
        self.stages.append(row)
        extra_txt = " ".join(f"{k}={v}" for k, v in extra.items())
        if extra_txt:
            extra_txt = " " + extra_txt
        print(
            f"[TIMER][{self.label}] stage={name} "
            f"dt={_fmt_sec(dt)} elapsed={_fmt_sec(total)}{extra_txt}",
            flush=True,
        )

    def total_sec(self) -> float:
        return time.monotonic() - self.t0


class ProgressETA:
    def __init__(self, total: int, label: str = "TRELLIS"):
        self.total = max(0, int(total))
        self.label = label
        self.t0 = time.monotonic()
        self.done = 0
        self.success = 0
        self.failed = 0
        self.skipped = 0

    def update(
        self,
        done: int | None = None,
        *,
        success_delta: int = 0,
        failed_delta: int = 0,
        skipped_delta: int = 0,
        target_id: str | None = None,
        status: str | None = None,
        candidate_index: int | None = None,
        candidate_total: int | None = None,
        unique_key: str | None = None,
    ) -> None:
        if done is None:
            self.done += 1
        else:
            self.done = int(done)  # pragma: no cover
        self.success += int(success_delta)
        self.failed += int(failed_delta)
        self.skipped += int(skipped_delta)

        elapsed = time.monotonic() - self.t0
        avg = elapsed / max(1, self.done)
        left = max(0, self.total - self.done)
        eta = avg * left

        cand = ""
        if candidate_index is not None and candidate_total is not None:
            cand = f" candidate={candidate_index}/{candidate_total}"
        elif candidate_index is not None:  # pragma: no cover
            cand = f" candidate={candidate_index}"  # pragma: no cover

        uk = ""
        if unique_key:
            short = str(unique_key)
            if len(short) > 96:
                short = short[:93] + "..."
            uk = f" unique_key={short}"

        tid = f" target={target_id}" if target_id else ""
        st = f" status={status}" if status else ""

        print(
            f"[{self.label}][{self.done:03d}/{self.total:03d}]"
            f"{tid}{cand}{st}{uk} "
            f"elapsed={_fmt_sec(elapsed)} avg/item={_fmt_sec(avg)} eta={_fmt_sec(eta)} "
            f"ok={self.success} failed={self.failed} skipped={self.skipped}",
            flush=True,
        )


class TrellisCandidateBlacklist:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {"schema": "trellis_candidate_blacklist/v1", "targets": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
                    self.data.setdefault("targets", {})
            except Exception:
                pass

    def is_blocked(self, target_id: str, unique_key: str) -> bool:
        target = self.data.setdefault("targets", {}).get(str(target_id), {})
        bad = target.get("bad_unique_keys", {})
        return str(unique_key) in bad

    def add_failure(
        self,
        target_id: str,
        unique_key: str,
        *,
        error: str = "",
        max_failures: int = 2,
    ) -> int:
        targets = self.data.setdefault("targets", {})
        target = targets.setdefault(str(target_id), {})
        bad = target.setdefault("bad_unique_keys", {})
        row = bad.setdefault(str(unique_key), {"failures": 0, "errors": []})
        row["failures"] = int(row.get("failures", 0)) + 1
        if error:
            errors = row.setdefault("errors", [])
            errors.append(str(error)[-4000:])
            del errors[:-5]
        row["blocked"] = row["failures"] >= int(max_failures)
        self.save()
        return int(row["failures"])

    def failures(self, target_id: str, unique_key: str) -> int:
        target = self.data.setdefault("targets", {}).get(str(target_id), {})
        bad = target.get("bad_unique_keys", {})
        row = bad.get(str(unique_key), {})
        return int(row.get("failures", 0))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def extract_candidate_pool(binding: dict[str, Any]) -> list[dict[str, Any]]:
    pools: list[Any] = []
    for key in (
        "supplier_candidate_pool",
        "candidate_pool",
        "candidates",
        "top_candidates",
        "supplier_candidates",
        "ranked_candidates",
    ):
        if isinstance(binding.get(key), list):
            pools.append(binding[key])

    meta = binding.get("meta")
    if isinstance(meta, dict):
        for key in (
            "supplier_candidate_pool",
            "candidate_pool",
            "candidates",
            "top_candidates",
            "supplier_candidates",
            "ranked_candidates",
        ):
            if isinstance(meta.get(key), list):
                pools.append(meta[key])

    source = binding.get("source")
    if isinstance(source, dict):
        for key in (
            "supplier_candidate_pool",
            "candidate_pool",
            "candidates",
            "top_candidates",
            "supplier_candidates",
            "ranked_candidates",
        ):
            if isinstance(source.get(key), list):
                pools.append(source[key])

    current = None
    for key in ("candidate", "supplier_candidate", "selected_candidate", "best_candidate"):
        if isinstance(binding.get(key), dict):
            current = binding[key]
            break
    if current is None and isinstance(meta, dict):
        for key in ("candidate", "supplier_candidate", "selected_candidate", "best_candidate"):  # pragma: no cover
            if isinstance(meta.get(key), dict):  # pragma: no cover
                current = meta[key]  # pragma: no cover
                break  # pragma: no cover

    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(c: Any) -> None:
        if not isinstance(c, dict):
            return
        uk = str(c.get("unique_key") or c.get("id") or c.get("product_url") or c.get("model_url") or "")
        if not uk:
            uk = json.dumps(c, ensure_ascii=False, sort_keys=True)[:300]
        if uk in seen:
            return
        seen.add(uk)
        result.append(c)

    add(current)
    for pool in pools:
        for c in pool:
            add(c)
    return result


def candidate_unique_key(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("unique_key")
        or candidate.get("id")
        or candidate.get("product_url")
        or candidate.get("model_page_url")
        or candidate.get("model_download_url")
        or candidate.get("title")
        or ""
    )


def apply_candidate_to_binding(binding: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    # Preserve compatibility with multiple binding schemas.
    for key in ("candidate", "supplier_candidate", "selected_candidate", "best_candidate"):
        if key in binding:
            binding[key] = candidate

    meta = binding.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["supplier_candidate"] = candidate
        meta["supplier_candidate_unique_key"] = candidate_unique_key(candidate)
        meta["candidate_fallback_applied"] = True

    source = binding.setdefault("source", {})
    if isinstance(source, dict):
        source["supplier_unique_key"] = candidate_unique_key(candidate)
        source["supplier_source_site"] = candidate.get("source_site")
        source["supplier_product_url"] = candidate.get("product_url")
        source["supplier_model_url"] = candidate.get("model_download_url") or candidate.get("model_page_url")

    # The orchestrator often reads fields directly from the binding.
    for k, v in candidate.items():
        binding.setdefault(k, v)

    binding["unique_key"] = candidate_unique_key(candidate)
    return binding
