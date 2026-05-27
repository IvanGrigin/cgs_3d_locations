#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import html
from copy import deepcopy
from pathlib import Path
from typing import Any


SELECTED_BINDING_STATUSES = {
    "heuristic_top1_selected",
    "heuristic_first_acceptable_selected",
    "llm_reranked_top1_selected",
    "llm_reranked_first_acceptable_selected",
}


def _read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _md_escape(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text.replace("|", "\\|").replace("\n", " ")


def _h(value: Any) -> str:
    return html.escape(str(value if value is not None else "").strip(), quote=True)


def _parse_images(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            return [raw]
    return []


def _format_price(value: Any, currency: Any = None) -> str:
    if value is None or value == "":
        return "цена не указана"
    currency_text = str(currency or "RUB").strip() or "RUB"
    try:
        number = float(value)
        if number.is_integer():
            formatted = f"{int(number):,}".replace(",", " ")
        else:
            formatted = f"{number:,.2f}".replace(",", " ")
        if currency_text.upper() == "RUB":
            return f"{formatted} руб."
        return f"{formatted} {currency_text}"
    except Exception:
        return f"{value} {currency_text}".strip()


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "не указано"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _fmt_score(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if _float_or_none(value) is not None else None
    return value


def _candidate_id(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("unique_key", "supplier_id", "sku", "external_id", "id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return str(candidate.get("title") or candidate.get("name") or "").strip()


def _candidate_title(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("title") or candidate.get("name") or "Без названия").strip()


def _candidate_category(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("category_norm", "category_raw", "semantic_group", "category"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


def _score_breakdown(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}  # pragma: no cover
    value = candidate.get("score_breakdown")
    return value if isinstance(value, dict) else {}


def _candidate_final_score(candidate: dict[str, Any] | None) -> float | None:
    if not isinstance(candidate, dict):
        return None  # pragma: no cover
    return _float_or_none(candidate.get("final_score") or _score_breakdown(candidate).get("final_score") or candidate.get("score"))


def _candidate_has_local_asset(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    return bool(str(candidate.get("asset_local_path") or "").strip())


def _candidate_has_downloadable_asset(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False  # pragma: no cover
    if str(candidate.get("model_download_url") or "").strip():
        return True
    if str(candidate.get("model_download_landing_url") or "").strip():
        return True
    breakdown = _score_breakdown(candidate)
    return bool(breakdown.get("has_downloadable_asset") or breakdown.get("has_model_url"))


def _candidate_model_url(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""  # pragma: no cover
    for key in ("model_download_url", "download_url", "model_download_landing_url", "model_page_url", "model_vendor_url"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


def _selection_mode_for_row(binding: dict[str, Any], candidate: dict[str, Any] | None, bindings_meta: dict[str, Any]) -> str:
    for value in (
        binding.get("supplier_selection_mode"),
        binding.get("selection_mode"),
        candidate.get("selection_mode") if isinstance(candidate, dict) else None,
        _score_breakdown(candidate).get("selection_mode"),
        bindings_meta.get("supplier_selection_mode"),
        bindings_meta.get("selection_mode"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _status_badge_class(status: Any) -> str:
    text = str(status or "").lower()
    if text in SELECTED_BINDING_STATUSES or "selected" in text:
        return "ok"
    if "no_real_asset" in text or "failed" in text or "error" in text:
        return "bad"
    if "no_candidates" in text or "no_acceptable" in text:
        return "warn"
    if "kept" in text or "generated" in text:
        return "muted"
    return "neutral"


def _status_badge(status: Any) -> str:
    text = str(status or "unknown").strip() or "unknown"
    return f'<span class="badge {_status_badge_class(text)}">{_h(text)}</span>'


def _price_value_sum(rows: list[dict[str, Any]], key: str) -> float | None:
    total = 0.0
    found = False
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            total += float(value)
            found = True
        except Exception:
            continue
    return total if found else None


def _product_link(candidate: dict[str, Any]) -> str:
    for key in ("product_url", "model_page_url", "model_vendor_url", "source_url", "model_download_landing_url"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


def _selected_candidate(binding: dict[str, Any]) -> dict[str, Any] | None:
    status = str(binding.get("selection_status") or "").strip()
    if status not in SELECTED_BINDING_STATUSES:
        return None
    chosen = binding.get("chosen_candidate")
    if not isinstance(chosen, dict):
        return None  # pragma: no cover
    final_source = ((binding.get("provenance") or {}).get("final_asset_source") or "")
    if final_source not in {"supplier_catalog", "supplier_catalog_pending"}:
        return None
    return chosen


def _applied_scene_items(scene_json_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not scene_json_path:
        return {}
    path = Path(scene_json_path).expanduser().resolve()
    if not path.is_file():
        return {}  # pragma: no cover
    data = _read_json(path)
    placements = data.get("placements") or data.get("items") or []
    if not isinstance(placements, list):
        return {}  # pragma: no cover

    out: dict[str, dict[str, Any]] = {}
    for item in placements:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") or {}
        if not isinstance(meta, dict) or not meta.get("supplier_binding_applied"):
            continue
        target_id = str(meta.get("supplier_binding_target_id") or item.get("id") or "").strip()
        if target_id:
            out[target_id] = item
    return out


def _scene_supplier_summary(scene_json_path: str | Path | None) -> dict[str, Any]:
    if not scene_json_path:
        return {}
    path = Path(scene_json_path).expanduser().resolve()
    if not path.is_file():
        return {}
    try:
        data = _read_json(path)
    except Exception:  # pragma: no cover
        return {}  # pragma: no cover
    meta = data.get("meta") or {}
    if not isinstance(meta, dict):
        return {}  # pragma: no cover
    summary = meta.get("supplier_binding_summary") or {}
    return summary if isinstance(summary, dict) else {}


def _build_issues_by_target(blender_build_report_path: str | Path | None) -> dict[str, list[str]]:
    if not blender_build_report_path:
        return {}
    path = Path(blender_build_report_path).expanduser().resolve()
    if not path.is_file():
        return {}
    try:
        data = _read_json(path)
    except Exception:  # pragma: no cover
        return {}  # pragma: no cover
    raw = data.get("item_issues") if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}  # pragma: no cover
    out: dict[str, list[str]] = {}
    for target_id, issues in raw.items():
        if isinstance(issues, list):
            out[str(target_id)] = [str(x) for x in issues if str(x).strip()]
    return out


def _binding_consistency_info(binding: dict[str, Any], bindings_meta: dict[str, Any]) -> dict[str, Any]:
    notes = [str(x) for x in (binding.get("selection_notes") or [])]
    shared_candidate = ""
    for note in notes:
        if note.startswith("scene_consistency_shared_candidate:"):
            shared_candidate = note.split(":", 1)[1].strip()
            break
    group_id = str(binding.get("consistency_group_id") or "").strip()
    applied = bool(shared_candidate or binding.get("consistency_applied"))
    scene_consistency = bindings_meta.get("scene_consistency")
    if not group_id and isinstance(scene_consistency, dict):
        for group in scene_consistency.get("applied_groups") or []:
            if not isinstance(group, dict):
                continue  # pragma: no cover
            target_ids = {str(x) for x in group.get("target_ids") or []}
            if str(binding.get("target_id") or "") in target_ids:
                group_id = str(group.get("group_id") or group.get("semantic_group") or "").strip()
                applied = True
                if not shared_candidate:
                    shared_candidate = str(group.get("chosen_candidate_id") or group.get("shared_candidate") or "").strip()
                break
    return {
        "consistency_group_id": group_id or None,
        "consistency_applied": applied,
        "shared_candidate": shared_candidate or None,
    }


def _apply_status_for_row(binding: dict[str, Any], applied_item: dict[str, Any] | None) -> str:
    if applied_item is not None:
        return "applied"
    if str(binding.get("selection_status") or "") in SELECTED_BINDING_STATUSES:
        return "not_applied"
    return "not_selected"


def _replacement_rows(
    *,
    bindings_json_path: str | Path,
    supplier_scene_json_path: str | Path | None,
    blender_build_report_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    bindings_data = _read_json(bindings_json_path)
    bindings_meta = bindings_data.get("meta") or {}
    if not isinstance(bindings_meta, dict):
        bindings_meta = {}  # pragma: no cover
    bindings = bindings_data.get("bindings") or []
    if not isinstance(bindings, list):
        raise RuntimeError(f"Некорректный supplier bindings JSON: {bindings_json_path}")

    applied_items = _applied_scene_items(supplier_scene_json_path)
    build_issues = _build_issues_by_target(blender_build_report_path)
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        candidate = _selected_candidate(binding)
        raw_chosen = binding.get("chosen_candidate")
        if candidate is None and isinstance(raw_chosen, dict):
            candidate = raw_chosen

        target_id = str(binding.get("target_id") or "").strip()
        applied_item = applied_items.get(target_id)

        images = _parse_images(candidate.get("images_json") or candidate.get("images")) if isinstance(candidate, dict) else []
        original = {}
        if isinstance(applied_item, dict):
            original = deepcopy(((applied_item.get("meta") or {}).get("original_generated_item") or {}))
        consistency = _binding_consistency_info(binding, bindings_meta)
        score_breakdown = _score_breakdown(candidate)
        top_candidates = [deepcopy(x) for x in (binding.get("top_candidates") or []) if isinstance(x, dict)]
        notes = deepcopy(binding.get("selection_notes") or [])
        target_build_issues = build_issues.get(target_id) or []
        used_alternative = any("asset_acquisition_selected_real_candidate_rank" in str(x) and not str(x).endswith(":1") for x in notes)
        used_alternative = used_alternative or any(str(x).startswith("used_alternative_candidate:") for x in target_build_issues)
        status = str(binding.get("selection_status") or "").strip()
        rejection_reason = ""
        if status in {"no_candidates_found", "no_acceptable_candidates_found", "no_real_asset_after_acquisition"}:
            rejection_reason = ", ".join(str(x) for x in notes) or status

        rows.append(
            {
                "target_id": target_id,
                "old_name": original.get("name") or binding.get("category") or "",
                "old_category": original.get("category") or binding.get("category") or "",
                "semantic_group": binding.get("semantic_group"),
                "target_category": binding.get("category"),
                "replacement_policy": binding.get("replacement_policy"),
                "new_title": _candidate_title(candidate) if candidate else "n/a",
                "candidate_id": _candidate_id(candidate),
                "candidate_category": _candidate_category(candidate),
                "brand": candidate.get("brand") if candidate else None,
                "collection": candidate.get("collection") if candidate else None,
                "source_site": candidate.get("source_site") if candidate else None,
                "product_url": _product_link(candidate) if candidate else "",
                "model_url": _candidate_model_url(candidate),
                "price": _format_price(candidate.get("price_value"), candidate.get("price_currency")) if candidate else "n/a",
                "price_value": candidate.get("price_value") if candidate else None,
                "price_currency": (candidate.get("price_currency") if candidate else None) or "RUB",
                "image_url": images[0] if images else "",
                "images": images,
                "asset_status": candidate.get("asset_status") if candidate else None,
                "asset_format": candidate.get("asset_format") if candidate else None,
                "asset_local_path": candidate.get("asset_local_path") if candidate else None,
                "has_local_asset": _candidate_has_local_asset(candidate),
                "has_downloadable_asset": _candidate_has_downloadable_asset(candidate),
                "style": candidate.get("style") if candidate else None,
                "color": candidate.get("color") if candidate else None,
                "materials": candidate.get("materials") if candidate else None,
                "description": candidate.get("description") if candidate else None,
                "dimensions_cm": [
                    candidate.get("width_cm") if candidate else None,
                    candidate.get("depth_cm") if candidate else None,
                    candidate.get("height_cm") if candidate else None,
                ],
                "status": status,
                "selection_status": status,
                "selection_mode": _selection_mode_for_row(binding, candidate, bindings_meta),
                "replacement_reason": binding.get("replacement_reason"),
                "selection_notes": notes,
                "selection_reason": binding.get("selection_reason") or (candidate.get("selection_reason") if candidate else None),
                "rejection_reason": binding.get("rejection_reason") or rejection_reason,
                "acquisition_status": candidate.get("asset_status") if candidate else status,
                "apply_status": _apply_status_for_row(binding, applied_item),
                "used_alternative_candidate": used_alternative,
                "build_issues": target_build_issues,
                "final_score": _candidate_final_score(candidate),
                "score_breakdown": score_breakdown,
                "top_candidates": top_candidates,
                "chosen_candidate": deepcopy(candidate) if candidate else None,
                "candidate_count": binding.get("candidate_count"),
                "is_selected": status in SELECTED_BINDING_STATUSES and candidate is not None,
                **consistency,
            }
        )
    return rows


def _total_price(rows: list[dict[str, Any]]) -> str:
    currency = "RUB"
    for row in rows:
        if row.get("price_value") is not None and row.get("price_value") != "":
            currency = str(row.get("price_currency") or currency or "RUB")
    total = _price_value_sum(rows, "price_value")
    if total is None:
        return "цены не указаны"  # pragma: no cover
    return _format_price(total, currency)


def _surface_total_price(surface_rows: list[dict[str, Any]]) -> str:
    total = _price_value_sum(surface_rows, "final_price_value")
    if total is None:
        return "цены не указаны"
    return _format_price(total, "RUB")


def _estimate_total_price(rows: list[dict[str, Any]], surface_rows: list[dict[str, Any]]) -> str:
    furniture_total = _price_value_sum(rows, "price_value") or 0.0
    surface_total = _price_value_sum(surface_rows, "final_price_value") or 0.0
    if furniture_total <= 0.0 and surface_total <= 0.0:
        return "цены не указаны"  # pragma: no cover
    return _format_price(furniture_total + surface_total, "RUB")


def _surface_quantity_text(row: dict[str, Any]) -> str:
    quantity = row.get("quantity")
    unit = str(row.get("quantity_unit") or "").strip()
    if quantity is None or quantity == "":
        return "не рассчитано"
    unit_name = {"package": "уп.", "roll": "рул."}.get(unit, unit or "шт.")
    return f"{_format_number(quantity, 0)} {unit_name}"


def _short_report_markdown(rows: list[dict[str, Any]], surface_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Краткая смета по заменам",
        "",
        f"- Заменено товаров: {len(rows)}",
        f"- Мебель, сумма по известным ценам: {_total_price(rows)}",
        f"- Материалы поверхностей, расчетная сумма: {_surface_total_price(surface_rows)}",
        f"- Итого по известным позициям: {_estimate_total_price(rows, surface_rows)}",
        "",
        "| Категория заменяемого товара | Новый товар | Где купить | Цена | Фото |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        product_url = str(row.get("product_url") or "").strip()
        product_cell = f"[{_md_escape(row['new_title'])}]({product_url})" if product_url else _md_escape(row["new_title"])
        buy_cell = f"[ссылка]({product_url})" if product_url else "нет ссылки"
        image_url = str(row.get("image_url") or "").strip()
        image_cell = f"![фото]({image_url})" if image_url else "нет фото"
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(row.get("old_category") or row.get("semantic_group")),
                    product_cell,
                    buy_cell,
                    _md_escape(row.get("price")),
                    image_cell,
                ]
            )
            + " |"
        )
    if surface_rows:
        lines.extend(
            [
                "",
                "## Материалы поверхностей",
                "",
                "| Поверхность | Материал | Площадь, м² | Площадь упаковки/рулона, м² | Купить | Цена за ед. | Итого |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in surface_rows:
            product_url = str(row.get("product_url") or "").strip()
            title = str(row.get("title") or "Без названия").strip()
            title_cell = f"[{_md_escape(title)}]({product_url})" if product_url else _md_escape(title)
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(row.get("label")),
                        title_cell,
                        _md_escape(_format_number(row.get("coverage_area_m2"))),
                        _md_escape(_format_number(row.get("package_area_m2"))),
                        _md_escape(_surface_quantity_text(row)),
                        _md_escape(_format_price(row.get("unit_price_value"), row.get("currency"))),
                        _md_escape(_format_price(row.get("final_price_value"), row.get("currency"))),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _extended_report_markdown(
    rows: list[dict[str, Any]],
    *,
    bindings_json_path: str | Path,
    scene_json_path: str | Path | None,
    surface_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Расширенная смета по заменам",
        "",
        "## Сводка",
        "",
        f"- Заменено товаров: {len(rows)}",
        f"- Мебель, сумма по известным ценам: {_total_price(rows)}",
        f"- Материалы поверхностей, расчетная сумма: {_surface_total_price(surface_rows)}",
        f"- Итого по известным позициям: {_estimate_total_price(rows, surface_rows)}",
        f"- Supplier bindings: `{Path(bindings_json_path).expanduser().resolve()}`",
    ]
    if scene_json_path:
        lines.append(f"- Supplier scene: `{Path(scene_json_path).expanduser().resolve()}`")
    if surface_rows:
        lines.extend(["", "## Материалы поверхностей", ""])
        for row in surface_rows:
            lines.extend(
                [
                    f"### {row.get('label')}: {row.get('title') or 'Без названия'}",
                    "",
                    f"- SKU: `{row.get('sku') or 'не указан'}`",
                    f"- Бренд: {row.get('brand') or 'не указан'}",
                    f"- Площадь покрытия: {_format_number(row.get('coverage_area_m2'))} м²",
                    f"- Площадь упаковки/рулона: {_format_number(row.get('package_area_m2'))} м²",
                    f"- Купить: {_surface_quantity_text(row)}",
                    f"- Цена за единицу: {_format_price(row.get('unit_price_value'), row.get('currency'))}",
                    f"- Итого: **{_format_price(row.get('final_price_value'), row.get('currency'))}**",
                    f"- Статус цены: `{row.get('price_status') or 'unknown'}`",
                    f"- Ссылка на товар: {f'[{row.get("product_url")}]({row.get("product_url")})' if row.get('product_url') else 'нет ссылки'}",
                    f"- Pricing JSON: `{row.get('pricing_json') or 'нет'}`",
                    f"- Selection JSON: `{row.get('selection_json')}`",
                    "",
                ]
            )
    lines.extend(["", "## Замененные товары", ""])

    for idx, row in enumerate(rows, start=1):
        product_url = str(row.get("product_url") or "").strip()
        model_url = str(row.get("model_url") or "").strip()
        title = str(row.get("new_title") or "Без названия").strip()
        title_text = f"[{title}]({product_url})" if product_url else title
        image_url = str(row.get("image_url") or "").strip()
        width, depth, height = row.get("dimensions_cm") or [None, None, None]
        lines.extend(
            [
                f"### {idx}. {title_text}",
                "",
                f"- Target ID: `{row.get('target_id')}`",
                f"- Заменяемая категория: `{row.get('old_category') or row.get('semantic_group')}`",
                f"- Семантическая группа: `{row.get('semantic_group')}`",
                f"- Поставщик/источник: `{row.get('source_site') or 'unknown'}`",
                f"- Бренд: {row.get('brand') or 'не указан'}",
                f"- Коллекция: {row.get('collection') or 'не указана'}",
                f"- Цена: **{row.get('price')}**",
                f"- Стиль: {row.get('style') or 'не указан'}",
                f"- Цвет: {row.get('color') or 'не указан'}",
                f"- Материалы: {row.get('materials') or 'не указаны'}",
                f"- Описание: {row.get('description') or 'не указано'}",
                f"- Размеры, см: {width or '?'} x {depth or '?'} x {height or '?'}",
                f"- Статус ассета: `{row.get('asset_status') or 'unknown'}`",
                f"- Формат ассета: `{row.get('asset_format') or 'unknown'}`",
                f"- Локальный ассет: `{row.get('asset_local_path') or 'нет'}`",
                f"- Ссылка на товар: {f'[{product_url}]({product_url})' if product_url else 'нет ссылки'}",
                f"- Ссылка на модель: {f'[{model_url}]({model_url})' if model_url else 'нет ссылки'}",
            ]
        )
        if image_url:
            lines.extend(["", f"![{_md_escape(title)}]({image_url})"])
        if row.get("selection_notes"):
            notes = ", ".join(str(x) for x in row["selection_notes"])
            lines.extend(["", f"- Примечания выбора: {notes}"])
        lines.append("")

    return "\n".join(lines)


def _image_links_html(row: dict[str, Any]) -> str:
    images = row.get("images") or []
    if not images and row.get("image_url"):
        images = [row.get("image_url")]  # pragma: no cover
    links = []
    for idx, url in enumerate(images, start=1):
        url_text = str(url or "").strip()
        if not url_text:
            continue  # pragma: no cover
        links.append(
            f'<a href="{_h(url_text)}" target="_blank" rel="noopener">фото {idx}</a>'
            f' <a href="{_h(url_text)}" download>скачать</a>'
        )
    return "<br>".join(links) if links else "нет фото"


DESIGN_SCORE_KEYS = [
    "identity_gate_checked",
    "identity_gate_passed",
    "identity_target_group",
    "identity_required_hits",
    "identity_forbidden_hits",
    "identity_reject_reason",
    "category_score",
    "size_score",
    "asset_availability_score",
    "color_score",
    "material_score",
    "style_score",
    "epoch_score",
    "description_score",
    "price_score",
    "design_similarity_score",
    "negative_penalty",
]


def _score_table_html(score_breakdown: dict[str, Any]) -> str:
    if not isinstance(score_breakdown, dict) or not score_breakdown:
        return '<div class="na">n/a</div>'
    cells = []
    for key in DESIGN_SCORE_KEYS:
        value = score_breakdown.get(key)
        if isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            text = _fmt_score(value)
        elif value is None:
            text = "n/a"
        else:
            text = str(value)  # pragma: no cover
        cells.append(f"<tr><th>{_h(key)}</th><td>{_h(text)}</td></tr>")
    return f'<table class="score-table"><tbody>{"".join(cells)}</tbody></table>'


def _candidate_bool(value: bool) -> str:
    return "yes" if value else "no"


def _top_candidates_html(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return '<div class="na">n/a</div>'
    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        breakdown = _score_breakdown(candidate)
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{_h(_candidate_id(candidate))}</code></td>"
            f"<td>{_h(_candidate_title(candidate))}</td>"
            f"<td>{_h(_candidate_category(candidate))}</td>"
            f"<td>{_h(_format_price(candidate.get('price_value'), candidate.get('price_currency')))}</td>"
            f"<td>{_h(_fmt_score(_candidate_final_score(candidate)))}</td>"
            f"<td>{_h(_fmt_score(breakdown.get('category_score')))}</td>"
            f"<td>{_h(_fmt_score(breakdown.get('size_score')))}</td>"
            f"<td>{_h(_fmt_score(breakdown.get('style_score')))}</td>"
            f"<td>{_h(_fmt_score(breakdown.get('color_score')))}</td>"
            f"<td>{_h(_fmt_score(breakdown.get('material_score')))}</td>"
            f"<td>{_h(_fmt_score(breakdown.get('price_score')))}</td>"
            f"<td>{_h(_fmt_score(breakdown.get('asset_availability_score')))}</td>"
            f"<td>{_h(_candidate_bool(_candidate_has_local_asset(candidate)))}</td>"
            f"<td>{_h(_candidate_bool(_candidate_has_downloadable_asset(candidate)))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table class="compact">'
        "<thead><tr><th>#</th><th>candidate_id</th><th>title</th><th>category</th><th>price</th>"
        "<th>final</th><th>cat</th><th>size</th><th>style</th><th>color</th><th>mat</th><th>price</th>"
        "<th>asset</th><th>local</th><th>dl</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _diagnostics_html(row: dict[str, Any]) -> str:
    items = [
        ("selection_reason", row.get("selection_reason")),
        ("rejection_reason", row.get("rejection_reason")),
        ("acquisition_status", row.get("acquisition_status")),
        ("apply_status", row.get("apply_status")),
        ("used_alternative_candidate", "yes" if row.get("used_alternative_candidate") else None),
        ("build_issues", ", ".join(str(x) for x in (row.get("build_issues") or [])) or None),
        ("selection_notes", ", ".join(str(x) for x in (row.get("selection_notes") or [])) or None),
    ]
    lines = [
        f"<dt>{_h(label)}</dt><dd>{_h(value)}</dd>"
        for label, value in items
        if value is not None and str(value).strip()
    ]
    return f"<dl>{''.join(lines)}</dl>" if lines else '<div class="na">n/a</div>'


def _load_surface_pricing_items(run_dir_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir_path.glob("surface_materials.pricing*.json")):
        try:
            data = _read_json(path)
        except Exception:
            continue
        items = data.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            row = deepcopy(item)
            row["pricing_json"] = str(path.resolve())
            rows.append(row)
    return rows


def _match_surface_pricing_item(
    *,
    label: str,
    material: dict[str, Any],
    pricing_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    product_url = str(material.get("product_url") or "").strip()
    sku = str(material.get("sku") or "").strip()
    expected_target = "surface_floor" if label == "Пол" else "surface_walls" if label == "Обои" else ""
    fallback: dict[str, Any] | None = None

    for item in pricing_items:
        if expected_target and str(item.get("target_id") or "") != expected_target:
            continue
        if fallback is None:
            fallback = item
        item_url = str(item.get("product_url") or "").strip()
        item_sku = str(item.get("sku") or "").strip()
        if product_url and item_url == product_url:
            return item
        if sku and item_sku == sku:
            return item
    return fallback  # pragma: no cover


def _surface_material_rows(run_dir_path: Path) -> list[dict[str, Any]]:
    specs = [
        ("Пол", "flooring.selection.supplier.v1.json", "data/sourse/obi_floor_coverings_cards"),
        ("Обои", "wall_material.selection.supplier.v1.json", "data/sourse/domlenta_wallpapers"),
        ("Пол", "flooring.selection.base.v1.json", "data/sourse/obi_floor_coverings_cards"),
        ("Обои", "wall_material.selection.base.v1.json", "data/sourse/domlenta_wallpapers"),
    ]
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    pricing_items = _load_surface_pricing_items(run_dir_path)
    project_root = run_dir_path
    for parent in [run_dir_path, *run_dir_path.parents]:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            project_root = parent  # pragma: no cover
            break  # pragma: no cover
    for label, filename, base_dir in specs:
        path = run_dir_path / filename
        if not path.is_file():
            continue
        try:
            selection = _read_json(path)
        except Exception:  # pragma: no cover
            continue  # pragma: no cover
        material = selection.get("selected_material") or {}
        if not isinstance(material, dict):
            continue
        key = (label, str(material.get("sku") or ""))
        if key in seen:
            continue  # pragma: no cover
        seen.add(key)
        local_images = _parse_images(material.get("local_image_paths"))
        image_urls = _parse_images(material.get("image_urls"))
        pricing_item = _match_surface_pricing_item(label=label, material=material, pricing_items=pricing_items)
        resolved_local = []
        for image in local_images:
            image_path = Path(image).expanduser()
            if not image_path.is_absolute():
                image_path = project_root / base_dir / image_path
            if image_path.is_file():
                resolved_local.append(str(image_path.resolve()))
        row = {
            "label": label,
            "title": material.get("name") or "Без названия",
            "brand": material.get("brand"),
            "sku": material.get("sku"),
            "product_url": material.get("product_url"),
            "price": _format_price(material.get("price"), material.get("price_currency")),
            "currency": material.get("price_currency") or "RUB",
            "unit_price_value": material.get("price"),
            "image_url": image_urls[0] if image_urls else (resolved_local[0] if resolved_local else ""),
            "images": image_urls + resolved_local,
            "selection_json": str(path.resolve()),
        }
        if pricing_item:
            row.update(
                {
                    "coverage_area_m2": pricing_item.get("coverage_area_m2"),
                    "package_area_m2": pricing_item.get("package_area_m2"),
                    "quantity": pricing_item.get("quantity"),
                    "quantity_unit": pricing_item.get("quantity_unit"),
                    "unit_price_value": pricing_item.get("unit_price_value"),
                    "final_price_value": pricing_item.get("final_price_value"),
                    "currency": pricing_item.get("currency") or material.get("price_currency") or "RUB",
                    "price_status": pricing_item.get("price_status"),
                    "pricing_json": pricing_item.get("pricing_json"),
                }
            )
        rows.append(row)
    return rows


def _surface_materials_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cards = []
    for row in rows:
        product_url = str(row.get("product_url") or "").strip()
        image_url = str(row.get("image_url") or "").strip()
        image_html = (
            f'<a href="{_h(image_url)}" target="_blank" rel="noopener"><img src="{_h(image_url)}" alt="{_h(row.get("title"))}"></a>'
            if image_url
            else '<div class="no-image">нет фото</div>'
        )
        title = (
            f'<a href="{_h(product_url)}" target="_blank" rel="noopener">{_h(row.get("title"))}</a>'
            if product_url
            else _h(row.get("title"))
        )
        cards.append(
            f"""
            <article class="card">
              <div class="thumb">{image_html}</div>
              <div class="body">
                <h2>{_h(row.get('label'))}: {title}</h2>
                <dl>
                  <dt>SKU</dt><dd>{_h(row.get('sku'))}</dd>
                  <dt>Бренд</dt><dd>{_h(row.get('brand') or 'не указан')}</dd>
                  <dt>Площадь</dt><dd>{_h(_format_number(row.get('coverage_area_m2')))} м²</dd>
                  <dt>Упаковка/рулон</dt><dd>{_h(_format_number(row.get('package_area_m2')))} м²</dd>
                  <dt>Купить</dt><dd>{_h(_surface_quantity_text(row))}</dd>
                  <dt>Цена за ед.</dt><dd>{_h(_format_price(row.get('unit_price_value'), row.get('currency')))}</dd>
                  <dt>Итого</dt><dd><strong>{_h(_format_price(row.get('final_price_value'), row.get('currency')))}</strong></dd>
                  <dt>Товар</dt><dd>{f'<a href="{_h(product_url)}" target="_blank" rel="noopener">{_h(product_url)}</a>' if product_url else 'нет ссылки'}</dd>
                  <dt>Фото</dt><dd>{_image_links_html(row)}</dd>
                  <dt>Pricing</dt><dd><code>{_h(row.get('pricing_json') or 'нет')}</code></dd>
                  <dt>Selection</dt><dd><code>{_h(row.get('selection_json'))}</code></dd>
                </dl>
              </div>
            </article>
            """
        )
    return "<h1>Материалы поверхностей</h1>" + "".join(cards)


def _replacement_report_html(
    rows: list[dict[str, Any]],
    *,
    bindings_json_path: str | Path,
    scene_json_path: str | Path | None,
    surface_rows: list[dict[str, Any]] | None = None,
) -> str:
    selected_rows = [row for row in rows if row.get("is_selected")]
    applied_rows = [row for row in rows if row.get("apply_status") == "applied"]
    cards = []
    for idx, row in enumerate(rows, start=1):
        product_url = str(row.get("product_url") or "").strip()
        model_url = str(row.get("model_url") or "").strip()
        image_url = str(row.get("image_url") or "").strip()
        image_html = (
            f'<a href="{_h(image_url)}" target="_blank" rel="noopener">'
            f'<img src="{_h(image_url)}" alt="{_h(row.get("new_title"))}"></a>'
            if image_url
            else '<div class="no-image">нет фото</div>'
        )
        product_link = (
            f'<a href="{_h(product_url)}" target="_blank" rel="noopener">{_h(row.get("new_title"))}</a>'
            if product_url
            else _h(row.get("new_title"))
        )
        model_link = (
            f'<a href="{_h(model_url)}" target="_blank" rel="noopener">{_h(model_url)}</a>'
            if model_url
            else "нет ссылки"
        )
        width, depth, height = row.get("dimensions_cm") or [None, None, None]
        score_table = _score_table_html(row.get("score_breakdown") or {})
        top_candidates_table = _top_candidates_html(row.get("top_candidates") or [])
        diagnostics = _diagnostics_html(row)
        cards.append(
            f"""
            <article class="card">
              <div class="thumb">{image_html}</div>
              <div class="body">
                <h2>{idx}. {product_link} {_status_badge(row.get('status'))}</h2>
                <dl>
                  <dt>Target</dt><dd><code>{_h(row.get('target_id'))}</code></dd>
                  <dt>Target category</dt><dd>{_h(row.get('target_category') or row.get('old_category') or 'n/a')}</dd>
                  <dt>Semantic group</dt><dd>{_h(row.get('semantic_group') or 'n/a')}</dd>
                  <dt>Policy</dt><dd>{_h(row.get('replacement_policy') or 'n/a')}</dd>
                  <dt>Mode</dt><dd>{_h(row.get('selection_mode') or 'n/a')}</dd>
                  <dt>Consistency</dt><dd>{_h(row.get('consistency_group_id') or 'n/a')} / applied={_h(row.get('consistency_applied'))} / shared={_h(row.get('shared_candidate') or 'n/a')}</dd>
                  <dt>Candidate ID</dt><dd><code>{_h(row.get('candidate_id') or 'n/a')}</code></dd>
                  <dt>Candidate category</dt><dd>{_h(row.get('candidate_category') or 'n/a')}</dd>
                  <dt>Источник</dt><dd>{_h(row.get('source_site') or 'unknown')}</dd>
                  <dt>Бренд</dt><dd>{_h(row.get('brand') or 'не указан')}</dd>
                  <dt>Цена</dt><dd>{_h(row.get('price'))}</dd>
                  <dt>Final score</dt><dd>{_h(_fmt_score(row.get('final_score')))}</dd>
                  <dt>Размеры, см</dt><dd>{_h(width or '?')} x {_h(depth or '?')} x {_h(height or '?')}</dd>
                  <dt>Материалы</dt><dd>{_h(row.get('materials') or 'не указаны')}</dd>
                  <dt>Товар</dt><dd>{f'<a href="{_h(product_url)}" target="_blank" rel="noopener">{_h(product_url)}</a>' if product_url else 'нет ссылки'}</dd>
                  <dt>3D-модель</dt><dd>{model_link}</dd>
                  <dt>Фото</dt><dd>{_image_links_html(row)}</dd>
                  <dt>Local asset?</dt><dd>{_h(_candidate_bool(bool(row.get('has_local_asset'))))}</dd>
                  <dt>Downloadable?</dt><dd>{_h(_candidate_bool(bool(row.get('has_downloadable_asset'))))}</dd>
                  <dt>Локальный ассет</dt><dd><code>{_h(row.get('asset_local_path') or 'нет')}</code></dd>
                </dl>
                <details>
                  <summary>Score breakdown</summary>
                  {score_table}
                </details>
                <details>
                  <summary>Top candidates ({_h(len(row.get('top_candidates') or []))})</summary>
                  {top_candidates_table}
                </details>
                <details>
                  <summary>Reasons / diagnostics</summary>
                  {diagnostics}
                </details>
              </div>
            </article>
            """
        )

    surface_html = _surface_materials_html(surface_rows or [])

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Отчет по заменам supplier</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f8; color: #1f2933; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .summary {{ margin: 0 0 24px; color: #4b5563; line-height: 1.5; }}
    .card {{ display: grid; grid-template-columns: 240px 1fr; gap: 20px; background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .thumb img {{ width: 240px; height: 240px; object-fit: contain; background: #f1f5f9; border-radius: 6px; }}
    .no-image {{ width: 240px; height: 240px; display: grid; place-items: center; background: #f1f5f9; border-radius: 6px; color: #64748b; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    dl {{ display: grid; grid-template-columns: 130px 1fr; gap: 8px 14px; margin: 0; }}
    dt {{ color: #64748b; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    a {{ color: #075985; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    details {{ margin-top: 14px; border-top: 1px solid #e5e7eb; padding-top: 10px; }}
    summary {{ cursor: pointer; color: #334155; font-weight: 600; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; color: #475569; font-weight: 600; }}
    .compact {{ font-size: 12px; min-width: 1080px; }}
    .score-table {{ max-width: 520px; }}
    .table-scroll {{ overflow-x: auto; }}
    .badge {{ display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; vertical-align: middle; }}
    .badge.ok {{ background: #dcfce7; color: #166534; }}
    .badge.warn {{ background: #fef3c7; color: #92400e; }}
    .badge.bad {{ background: #fee2e2; color: #991b1b; }}
    .badge.muted {{ background: #e5e7eb; color: #4b5563; }}
    .badge.neutral {{ background: #dbeafe; color: #1e40af; }}
    .na {{ color: #64748b; margin-top: 8px; }}
    @media (max-width: 760px) {{ main {{ padding: 16px; }} .card {{ grid-template-columns: 1fr; }} .thumb img, .no-image {{ width: 100%; }} dl {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Отчет по заменам supplier</h1>
    <p class="summary">
      Targets в bindings: <strong>{len(rows)}</strong><br>
      Selected candidates: <strong>{len(selected_rows)}</strong><br>
      Applied replacements: <strong>{len(applied_rows)}</strong><br>
      Мебель, сумма по известным ценам: <strong>{_h(_total_price(selected_rows))}</strong><br>
      Материалы поверхностей, расчетная сумма: <strong>{_h(_surface_total_price(surface_rows or []))}</strong><br>
      Итого по известным позициям: <strong>{_h(_estimate_total_price(selected_rows, surface_rows or []))}</strong><br>
      Supplier bindings: <code>{_h(Path(bindings_json_path).expanduser().resolve())}</code><br>
      Supplier scene: <code>{_h(Path(scene_json_path).expanduser().resolve()) if scene_json_path else 'нет'}</code>
    </p>
    {surface_html}
    <h1>Замены мебели supplier</h1>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def _average_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "final_score",
        "category_score",
        "size_score",
        "style_score",
        "color_score",
        "material_score",
        "price_score",
        "asset_availability_score",
    ]
    out: dict[str, float] = {}
    for key in keys:
        values: list[float] = []
        for row in rows:
            value = row.get("final_score") if key == "final_score" else (row.get("score_breakdown") or {}).get(key)
            number = _float_or_none(value)
            if number is not None:
                values.append(number)
        out[key] = round(sum(values) / len(values), 6) if values else 0.0
    return out


def _summary_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = []
    for row in rows:
        targets.append(
            {
                "target_id": row.get("target_id"),
                "category": row.get("target_category") or row.get("old_category"),
                "status": row.get("status"),
                "chosen_candidate_id": row.get("candidate_id") or None,
                "chosen_candidate_title": row.get("new_title") if row.get("chosen_candidate") else None,
                "price": _float_or_none(row.get("price_value")),
                "final_score": _float_or_none(row.get("final_score")),
                "score_breakdown": _json_safe(row.get("score_breakdown")) if isinstance(row.get("score_breakdown"), dict) else {},
                "asset_local_path": row.get("asset_local_path") or None,
                "acquisition_status": row.get("acquisition_status") or None,
                "build_issues": row.get("build_issues") or [],
                "consistency": {
                    "consistency_group_id": row.get("consistency_group_id"),
                    "consistency_applied": bool(row.get("consistency_applied")),
                    "shared_candidate": row.get("shared_candidate"),
                },
            }
        )
    return targets


def _replacement_summary_json(
    rows: list[dict[str, Any]],
    *,
    bindings_json_path: str | Path,
    scene_json_path: str | Path | None,
    mode: str | None,
    blender_build_report_path: str | Path | None = None,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if row.get("is_selected")]
    scene_summary = _scene_supplier_summary(scene_json_path)
    warnings: list[str] = []
    if any(row.get("selection_mode") and row.get("is_selected") and not row.get("score_breakdown") for row in rows):
        warnings.append("Some selected design-aware rows do not contain score_breakdown.")
    if any(row.get("status") in SELECTED_BINDING_STATUSES and not row.get("candidate_id") for row in rows):
        warnings.append("Some selected rows do not contain chosen_candidate_id.")
    if scene_json_path and not scene_summary:
        warnings.append("Scene supplier_binding_summary was not found or empty.")

    return {
        "mode": mode or "",
        "bindings_path": str(Path(bindings_json_path).expanduser().resolve()),
        "scene_path": str(Path(scene_json_path).expanduser().resolve()) if scene_json_path else None,
        "blender_build_report_path": str(Path(blender_build_report_path).expanduser().resolve()) if blender_build_report_path else None,
        "counts": {
            "total_targets": len(rows),
            "replace_with_supplier_targets": sum(1 for row in rows if row.get("replacement_policy") == "replace_with_supplier"),
            "selected_count": len(selected_rows),
            "no_candidates_count": sum(1 for row in rows if row.get("status") in {"no_candidates_found", "no_acceptable_candidates_found"}),
            "no_real_asset_after_acquisition_count": sum(1 for row in rows if row.get("status") == "no_real_asset_after_acquisition"),
            "applied_replacement_count": int(scene_summary.get("replaced_count", 0) or sum(1 for row in rows if row.get("apply_status") == "applied")),
            "local_asset_replaced_count": int(scene_summary.get("local_asset_replaced_count", 0) or sum(1 for row in rows if row.get("apply_status") == "applied" and row.get("has_local_asset"))),
            "used_alternative_candidate_count": sum(1 for row in rows if row.get("used_alternative_candidate")),
        },
        "score_averages": _average_scores(selected_rows),
        "targets": _summary_targets(rows),
        "warnings": warnings,
    }


def write_supplier_replacement_reports(
    *,
    bindings_json_path: str | Path,
    run_dir: str | Path,
    supplier_scene_json_path: str | Path | None = None,
    blender_build_report_path: str | Path | None = None,
    short_filename: str = "supplier_replacements.short.md",
    extended_filename: str = "supplier_replacements.full.md",
    html_filename: str = "supplier_replacements.html",
    summary_filename: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    run_dir_path = Path(run_dir).expanduser().resolve()
    surface_rows = _surface_material_rows(run_dir_path)
    rows = _replacement_rows(
        bindings_json_path=bindings_json_path,
        supplier_scene_json_path=supplier_scene_json_path,
        blender_build_report_path=blender_build_report_path,
    )
    selected_rows = [row for row in rows if row.get("is_selected")]
    short_path = run_dir_path / short_filename
    extended_path = run_dir_path / extended_filename
    html_path = run_dir_path / html_filename
    _write_text(short_path, _short_report_markdown(selected_rows, surface_rows))
    _write_text(
        extended_path,
        _extended_report_markdown(
            selected_rows,
            bindings_json_path=bindings_json_path,
            scene_json_path=supplier_scene_json_path,
            surface_rows=surface_rows,
        ),
    )
    _write_text(
        html_path,
        _replacement_report_html(
            rows,
            bindings_json_path=bindings_json_path,
            scene_json_path=supplier_scene_json_path,
            surface_rows=surface_rows,
        ),
    )
    summary_path: Path | None = None
    summary: dict[str, Any] | None = None
    if summary_filename:
        summary_path = run_dir_path / summary_filename
        summary = _replacement_summary_json(
            rows,
            bindings_json_path=bindings_json_path,
            scene_json_path=supplier_scene_json_path,
            blender_build_report_path=blender_build_report_path,
            mode=mode,
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    result = {
        "short_md": str(short_path),
        "extended_md": str(extended_path),
        "html": str(html_path),
        "summary_json": str(summary_path) if summary_path else None,
        "replacement_count": len(selected_rows),
        "surface_material_count": len(surface_rows),
        "known_price_total": _total_price(selected_rows),
        "surface_material_total": _surface_total_price(surface_rows),
        "estimate_total": _estimate_total_price(selected_rows, surface_rows),
    }
    if summary is not None:
        result["counts"] = summary.get("counts")
        result["score_averages"] = summary.get("score_averages")
    return result
