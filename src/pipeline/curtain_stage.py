from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_curtain_catalog(catalog_path: str | Path) -> tuple[list[dict[str, Any]], Path]:
    root = Path(catalog_path).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if root.is_dir():
        candidates = [
            root / "shtorystore_curtains.jsonl",
            root / "products.jsonl",
        ]
        jsonl_path = next((p for p in candidates if p.is_file()), candidates[0])
        base_dir = root
    else:
        jsonl_path = root
        base_dir = root.parent

    rows = []
    for row in _load_jsonl(jsonl_path):
        if str(row.get("source") or "").lower() != "shtorystore":
            continue
        image_paths = row.get("local_image_paths")
        if not isinstance(image_paths, list) or not image_paths:
            continue
        rows.append(row)
    return rows, base_dir


def discover_curtain_models(models_dir: str | Path | None) -> list[str]:
    if not models_dir:
        return []
    root = Path(models_dir).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.is_dir():
        return []
    exts = {".fbx", ".glb", ".gltf", ".obj"}
    model_paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            model_paths.append(path)
    return [str(path.resolve()) for path in sorted(model_paths, key=lambda p: (p.suffix.lower(), str(p).lower()))]


def _curtain_model_rank_key(item: dict[str, Any]) -> tuple[float, str]:
    path = str(item.get("asset_local_path") or item.get("mesh_path") or "")
    title = str(item.get("title") or "")
    title_l = title.strip().lower()
    text = f"{title} {path}".lower()
    score = 0.0

    # Shtorystore products are mostly ordinary straight curtains, so prefer
    # rectangular curtain meshes over decorative swags/french curtains.
    if title_l == "штора" or path.lower().endswith("/shtora.fbx"):
        score += 10.0
    if "shtora" in text or "штора" in text:
        score += 5.0
    if "шторы" in text or "curtains" in text:
        score += 3.0
    # The local grommet model imports with many small auxiliary parts and
    # tends to leave vertical rail/eyelet artifacts after bounding-box fitting.
    if "люверс" in text or "grommet" in text or "curtain 2" in text:
        score -= 4.0
    if "француз" in text or "french" in text:
        score -= 9.0
    if "кружев" in text or "lace" in text:
        score -= 5.0

    suffix_bonus = {
        ".fbx": 0.4,
        ".glb": 0.3,
        ".gltf": 0.2,
        ".obj": 0.0,
    }.get(Path(path).suffix.lower(), 0.0)
    return (-score - suffix_bonus, text)


def _is_primary_plain_curtain_model(item: dict[str, Any] | str | Path) -> bool:
    if isinstance(item, dict):
        path = str(item.get("asset_local_path") or item.get("mesh_path") or "")
        title = str(item.get("title") or "")
    else:
        path = str(item)
        title = ""
    p = Path(path)
    text = f"{title} {path}".lower()
    return (
        p.suffix.lower() == ".fbx"
        and p.name.lower() == "shtora.fbx"
        and not any(token in text for token in ("люверс", "grommet", "curtain 2", "француз", "french", "кружев", "lace"))
    )


def discover_supplier_curtain_models(
    supplier_catalog_path: str | Path | None = "data/sourse/suppliers/supplier_catalog_canonical.json",
    manual_assets_root: str | Path | None = "data/sourse/suppliers/manual_assets/3ddd",
) -> list[dict[str, Any]]:
    exts = {".fbx", ".glb", ".gltf", ".obj"}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(path: str | Path, source: dict[str, Any]) -> None:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.is_file() or p.suffix.lower() not in exts:
            return
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        item = dict(source)
        item["asset_local_path"] = key
        item["asset_format"] = p.suffix.lstrip(".").lower()
        out.append(item)

    catalog_path = Path(str(supplier_catalog_path or "")).expanduser()
    if catalog_path and not catalog_path.is_absolute():
        catalog_path = (Path.cwd() / catalog_path).resolve()
    if catalog_path and catalog_path.is_file():
        try:
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            rows = data.get("items") if isinstance(data, dict) else data
        except Exception:
            rows = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("category_norm") or "").strip() != "curtain_blinds":
                    continue
                path = str(row.get("asset_local_path") or "").strip()
                if path:
                    add(
                        path,
                        {
                            "source": "supplier_catalog",
                            "title": row.get("title"),
                            "unique_key": row.get("unique_key"),
                            "asset_status": row.get("asset_status"),
                            "dimensions_cm": row.get("dimensions_cm"),
                            "preview_local_path": row.get("preview_local_path"),
                        },
                    )

    manual_root = Path(str(manual_assets_root or "")).expanduser()
    if manual_root and not manual_root.is_absolute():
        manual_root = (Path.cwd() / manual_root).resolve()
    if manual_root and manual_root.is_dir():
        for path in sorted(manual_root.rglob("*"), key=lambda p: str(p).lower()):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            text = str(path).lower()
            if not any(token in text for token in ("штор", "shtor", "curtain", "tulle", "blind")):
                continue
            add(
                path,
                {
                    "source": "supplier_manual_assets",
                    "title": path.parent.name,
                    "unique_key": f"manual_asset::{path}",
                    "asset_status": "local_manual_asset",
                    "dimensions_cm": None,
                    "preview_local_path": None,
                },
            )

    primary = [item for item in out if _is_primary_plain_curtain_model(item)]
    if primary:
        return sorted(primary, key=_curtain_model_rank_key)
    return sorted(out, key=_curtain_model_rank_key)
