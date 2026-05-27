from __future__ import annotations

from typing import Any


def patch_apply_postprocess(monkeypatch: Any, module: Any, *enabled: str) -> None:
    identity3 = lambda _data, items, _by_target: (items, {})
    identity2 = lambda items, _by_target: (items, {})
    if "ceiling" not in enabled:
        monkeypatch.setattr(module, "_collapse_ceiling_lights", identity3)
    if "lights" not in enabled:
        monkeypatch.setattr(module, "_normalize_supported_light_placements", identity3)
    if "computer" not in enabled:
        monkeypatch.setattr(module, "_ensure_computer_replacements", identity2)
    if "table_chair" not in enabled:
        monkeypatch.setattr(module, "_ensure_table_chair_affordances", identity3)
    if "tv" not in enabled:
        monkeypatch.setattr(module, "_ensure_tv_affordance", identity3)
