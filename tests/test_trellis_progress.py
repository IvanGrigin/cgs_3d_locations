import json

import pytest

from src import trellis_progress as tp


def test_stage_timer_and_progress_eta_print_deterministic(monkeypatch, capsys):
    times = iter([10.0, 11.2, 15.8, 20.0, 23.0])
    monkeypatch.setattr(tp.time, "monotonic", lambda: next(times))

    timer = tp.StageTimer("unit")
    timer.mark("load", count=2)
    assert timer.stages == [{"stage": "load", "dt_sec": 1.2, "elapsed_sec": 1.2, "count": 2}]
    assert timer.total_sec() == pytest.approx(5.8)

    progress = tp.ProgressETA(total=3, label="TEST")
    progress.update(success_delta=1, target_id="t1", status="done", candidate_index=1, candidate_total=2, unique_key="x" * 120)
    out = capsys.readouterr().out
    assert "[TIMER][unit] stage=load" in out
    assert "[TEST][001/003]" in out
    assert "candidate=1/2" in out
    assert "ok=1 failed=0 skipped=0" in out
    assert "..." in out


def test_candidate_blacklist_counts_blocks_and_recovers_from_bad_json(tmp_path):
    bad_path = tmp_path / "blacklist.json"
    bad_path.write_text("{not-json", encoding="utf-8")

    blacklist = tp.TrellisCandidateBlacklist(bad_path)
    assert not blacklist.is_blocked("target", "key")
    assert blacklist.failures("target", "key") == 0

    assert blacklist.add_failure("target", "key", error="first", max_failures=2) == 1
    assert blacklist.is_blocked("target", "key")
    assert blacklist.add_failure("target", "key", error="second", max_failures=2) == 2
    assert blacklist.is_blocked("target", "key")
    assert blacklist.failures("target", "key") == 2

    persisted = json.loads(bad_path.read_text(encoding="utf-8"))
    row = persisted["targets"]["target"]["bad_unique_keys"]["key"]
    assert row["blocked"] is True
    assert row["errors"] == ["first", "second"]


def test_candidate_pool_unique_keys_and_binding_application():
    current = {"id": "current", "source_site": "site-a", "product_url": "https://item/current"}
    duplicate = {"id": "current", "title": "dupe"}
    extra = {"model_download_url": "https://model.glb", "source_site": "site-b"}
    source_extra = {"title": "source-title"}
    binding = {
        "candidate": current,
        "candidate_pool": [duplicate, extra],
        "meta": {"supplier_candidate_pool": [{"unique_key": "meta-key", "price": 5}]},
        "source": {"ranked_candidates": [source_extra]},
    }

    pool = tp.extract_candidate_pool(binding)

    assert [tp.candidate_unique_key(c) for c in pool] == [
        "current",
        "https://model.glb",
        "meta-key",
        "source-title",
    ]
    assert tp.extract_candidate_pool({"candidate_pool": ["bad", {"id": "ok"}]}) == [{"id": "ok"}]

    applied = tp.apply_candidate_to_binding(
        {
            "candidate": {},
            "supplier_candidate": {},
            "selected_candidate": {},
            "best_candidate": {},
        },
        {
            "unique_key": "u1",
            "source_site": "catalog",
            "product_url": "https://product",
            "model_page_url": "https://model-page",
            "extra_field": 42,
        },
    )

    for key in ("candidate", "supplier_candidate", "selected_candidate", "best_candidate"):
        assert applied[key]["unique_key"] == "u1"
    assert applied["meta"]["supplier_candidate_unique_key"] == "u1"
    assert applied["meta"]["candidate_fallback_applied"] is True
    assert applied["source"]["supplier_source_site"] == "catalog"
    assert applied["source"]["supplier_product_url"] == "https://product"
    assert applied["source"]["supplier_model_url"] == "https://model-page"
    assert applied["extra_field"] == 42
    assert applied["unique_key"] == "u1"
