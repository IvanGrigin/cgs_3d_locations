from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_as_script(module_path: Path, module_name: str):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_script_style_import_fallbacks(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    monkeypatch.syspath_prepend(str(src))

    modules = {
        "apply_supplier_bindings_script_fallback_test": src / "apply_supplier_bindings.py",
        "run_pipeline_script_fallback_test": src / "run_pipeline.py",
        "run_procedural_room_supplier_script_fallback_test": src / "tools" / "run_procedural_room_supplier.py",
        "trellis_supplier_asset_orchestrator_script_fallback_test": src / "trellis_supplier_asset_orchestrator.py",
    }

    imported = {name: _import_as_script(path, name) for name, path in modules.items()}

    assert imported["apply_supplier_bindings_script_fallback_test"].ASSET_FALLBACK_MODE_NONE == "none"
    assert hasattr(imported["run_pipeline_script_fallback_test"], "run_pipeline_for_mode")
    assert hasattr(imported["run_procedural_room_supplier_script_fallback_test"], "build_cli")
    assert hasattr(imported["trellis_supplier_asset_orchestrator_script_fallback_test"], "run_orchestration")
