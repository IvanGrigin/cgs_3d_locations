#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import html
from copy import deepcopy
from pathlib import Path
from typing import Any


SELECTED_BINDING_STATUSES = {"heuristic_top1_selected", "llm_reranked_top1_selected"}


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
        return None
    final_source = ((binding.get("provenance") or {}).get("final_asset_source") or "")
    if final_source not in {"supplier_catalog", "supplier_catalog_pending"}:
        return None
    return chosen


def _applied_scene_items(scene_json_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not scene_json_path:
        return {}
    path = Path(scene_json_path).expanduser().resolve()
    if not path.is_file():
        return {}
    data = _read_json(path)
    placements = data.get("placements") or data.get("items") or []
    if not isinstance(placements, list):
        return {}

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


def _replacement_rows(
    *,
    bindings_json_path: str | Path,
    supplier_scene_json_path: str | Path | None,
) -> list[dict[str, Any]]:
    bindings_data = _read_json(bindings_json_path)
    bindings = bindings_data.get("bindings") or []
    if not isinstance(bindings, list):
        raise RuntimeError(f"Некорректный supplier bindings JSON: {bindings_json_path}")

    applied_items = _applied_scene_items(supplier_scene_json_path)
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        candidate = _selected_candidate(binding)
        if candidate is None:
            continue

        target_id = str(binding.get("target_id") or "").strip()
        applied_item = applied_items.get(target_id)
        if applied_items and applied_item is None:
            continue

        images = _parse_images(candidate.get("images_json") or candidate.get("images"))
        original = {}
        if isinstance(applied_item, dict):
            original = deepcopy(((applied_item.get("meta") or {}).get("original_generated_item") or {}))

        rows.append(
            {
                "target_id": target_id,
                "old_name": original.get("name") or binding.get("category") or "",
                "old_category": original.get("category") or binding.get("category") or "",
                "semantic_group": binding.get("semantic_group"),
                "new_title": candidate.get("title") or "Без названия",
                "brand": candidate.get("brand"),
                "collection": candidate.get("collection"),
                "source_site": candidate.get("source_site"),
                "product_url": _product_link(candidate),
                "model_url": candidate.get("model_download_url") or candidate.get("model_page_url"),
                "price": _format_price(candidate.get("price_value"), candidate.get("price_currency")),
                "price_value": candidate.get("price_value"),
                "price_currency": candidate.get("price_currency") or "RUB",
                "image_url": images[0] if images else "",
                "images": images,
                "asset_status": candidate.get("asset_status"),
                "asset_format": candidate.get("asset_format"),
                "asset_local_path": candidate.get("asset_local_path"),
                "style": candidate.get("style"),
                "color": candidate.get("color"),
                "materials": candidate.get("materials"),
                "description": candidate.get("description"),
                "dimensions_cm": [
                    candidate.get("width_cm"),
                    candidate.get("depth_cm"),
                    candidate.get("height_cm"),
                ],
                "selection_status": binding.get("selection_status"),
                "replacement_reason": binding.get("replacement_reason"),
                "selection_notes": deepcopy(binding.get("selection_notes") or []),
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
        return "цены не указаны"
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
        return "цены не указаны"
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
        images = [row.get("image_url")]
    links = []
    for idx, url in enumerate(images, start=1):
        url_text = str(url or "").strip()
        if not url_text:
            continue
        links.append(
            f'<a href="{_h(url_text)}" target="_blank" rel="noopener">фото {idx}</a>'
            f' <a href="{_h(url_text)}" download>скачать</a>'
        )
    return "<br>".join(links) if links else "нет фото"


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
    return fallback


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
            project_root = parent
            break
    for label, filename, base_dir in specs:
        path = run_dir_path / filename
        if not path.is_file():
            continue
        try:
            selection = _read_json(path)
        except Exception:
            continue
        material = selection.get("selected_material") or {}
        if not isinstance(material, dict):
            continue
        key = (label, str(material.get("sku") or ""))
        if key in seen:
            continue
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
            f'<a href="{_h(model_url)}" target="_blank" rel="noopener">модель</a>'
            if model_url
            else "нет ссылки"
        )
        width, depth, height = row.get("dimensions_cm") or [None, None, None]
        cards.append(
            f"""
            <article class="card">
              <div class="thumb">{image_html}</div>
              <div class="body">
                <h2>{idx}. {product_link}</h2>
                <dl>
                  <dt>Target</dt><dd><code>{_h(row.get('target_id'))}</code></dd>
                  <dt>Заменяет</dt><dd>{_h(row.get('old_category') or row.get('semantic_group'))}</dd>
                  <dt>Источник</dt><dd>{_h(row.get('source_site') or 'unknown')}</dd>
                  <dt>Бренд</dt><dd>{_h(row.get('brand') or 'не указан')}</dd>
                  <dt>Цена</dt><dd>{_h(row.get('price'))}</dd>
                  <dt>Размеры, см</dt><dd>{_h(width or '?')} x {_h(depth or '?')} x {_h(height or '?')}</dd>
                  <dt>Материалы</dt><dd>{_h(row.get('materials') or 'не указаны')}</dd>
                  <dt>Товар</dt><dd>{f'<a href="{_h(product_url)}" target="_blank" rel="noopener">{_h(product_url)}</a>' if product_url else 'нет ссылки'}</dd>
                  <dt>3D-модель</dt><dd>{model_link}</dd>
                  <dt>Фото</dt><dd>{_image_links_html(row)}</dd>
                  <dt>Локальный ассет</dt><dd><code>{_h(row.get('asset_local_path') or 'нет')}</code></dd>
                </dl>
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
    @media (max-width: 760px) {{ main {{ padding: 16px; }} .card {{ grid-template-columns: 1fr; }} .thumb img, .no-image {{ width: 100%; }} dl {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Отчет по заменам supplier</h1>
    <p class="summary">
      Заменено товаров: <strong>{len(rows)}</strong><br>
      Мебель, сумма по известным ценам: <strong>{_h(_total_price(rows))}</strong><br>
      Материалы поверхностей, расчетная сумма: <strong>{_h(_surface_total_price(surface_rows or []))}</strong><br>
      Итого по известным позициям: <strong>{_h(_estimate_total_price(rows, surface_rows or []))}</strong><br>
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


def write_supplier_replacement_reports(
    *,
    bindings_json_path: str | Path,
    run_dir: str | Path,
    supplier_scene_json_path: str | Path | None = None,
    short_filename: str = "supplier_replacements.short.md",
    extended_filename: str = "supplier_replacements.full.md",
    html_filename: str = "supplier_replacements.html",
) -> dict[str, Any]:
    run_dir_path = Path(run_dir).expanduser().resolve()
    surface_rows = _surface_material_rows(run_dir_path)
    rows = _replacement_rows(
        bindings_json_path=bindings_json_path,
        supplier_scene_json_path=supplier_scene_json_path,
    )
    short_path = run_dir_path / short_filename
    extended_path = run_dir_path / extended_filename
    html_path = run_dir_path / html_filename
    _write_text(short_path, _short_report_markdown(rows, surface_rows))
    _write_text(
        extended_path,
        _extended_report_markdown(
            rows,
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
    return {
        "short_md": str(short_path),
        "extended_md": str(extended_path),
        "html": str(html_path),
        "replacement_count": len(rows),
        "surface_material_count": len(surface_rows),
        "known_price_total": _total_price(rows),
        "surface_material_total": _surface_total_price(surface_rows),
        "estimate_total": _estimate_total_price(rows, surface_rows),
    }
