#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

# Требуется: pip install beautifulsoup4
from bs4 import BeautifulSoup


ARCHIVE_EXTS = {".rar", ".zip", ".7z"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm_href(href: str) -> str:
    return href.strip()


def _ext_from_url(url: str) -> str:
    p = urlparse(url)
    return Path(p.path).suffix.lower()


def _is_archive(url: str) -> bool:
    return _ext_from_url(url) in ARCHIVE_EXTS


def _is_image(url: str) -> bool:
    return _ext_from_url(url) in IMAGE_EXTS


def _url_filename(url: str) -> str:
    p = urlparse(url)
    return Path(p.path).name


def _pick_preview_img(tag) -> Optional[str]:
    """
    У imodern часто есть img src и data-src. Обычно data-src — реальный путь.
    """
    if not tag:
        return None
    if tag.has_attr("data-src") and tag["data-src"]:
        return tag["data-src"].strip()
    if tag.has_attr("src") and tag["src"]:
        return tag["src"].strip()
    return None


@dataclass
class ItemRow:
    bx_id: Optional[str]
    name: Optional[str]
    description: Optional[str]
    archive_url: Optional[str]
    archive_url_abs: Optional[str]
    archive_filename: Optional[str]
    archive_ext: Optional[str]
    preview_img_url: Optional[str]
    preview_img_url_abs: Optional[str]
    preview_img_alt: Optional[str]
    preview_img_title: Optional[str]
    source_file: str
    parsed_at: str
    raw_item_html: str
    extra_json: str


@dataclass
class LinkRow:
    bx_id: Optional[str]
    url: str
    url_abs: str
    kind: str
    pos: int


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS imodern_item (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bx_id TEXT UNIQUE,
                name TEXT,
                description TEXT,

                archive_url TEXT,
                archive_url_abs TEXT,
                archive_filename TEXT,
                archive_ext TEXT,

                preview_img_url TEXT,
                preview_img_url_abs TEXT,
                preview_img_alt TEXT,
                preview_img_title TEXT,

                source_file TEXT NOT NULL,
                parsed_at TEXT NOT NULL,
                raw_item_html TEXT NOT NULL,
                extra_json TEXT NOT NULL
            );
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS imodern_link (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bx_id TEXT,
                url TEXT NOT NULL,
                url_abs TEXT NOT NULL,
                kind TEXT NOT NULL,
                pos INTEGER NOT NULL,
                UNIQUE(bx_id, url, kind)
            );
            """
        )

        con.execute("CREATE INDEX IF NOT EXISTS idx_imodern_link_bx ON imodern_link(bx_id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_imodern_item_name ON imodern_item(name);")


def parse_html_file(html_path: Path, base_url: str) -> tuple[List[ItemRow], List[LinkRow]]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    items: List[ItemRow] = []
    links: List[LinkRow] = []

    parsed_at = _now_utc_iso()
    source_file = str(html_path)

    # Карточки
    cards = soup.select("div.model-item.catalog-filter-item")
    for card in cards:
        bx_id = card.get("id")

        name_tag = card.select_one(".model-info .model-name")
        name = name_tag.get_text(" ", strip=True) if name_tag else None

        # Потенциальное описание: если на сайте появится блок — подхватим.
        desc_tag = card.select_one(".model-info .model-desc, .model-info .description, .model-info .model-description")
        description = desc_tag.get_text(" ", strip=True) if desc_tag else None

        # Все ссылки в карточке
        hrefs_raw = []
        for a in card.select("a[href]"):
            href = _norm_href(a.get("href", ""))
            if href:
                hrefs_raw.append(href)

        # Дедуп с сохранением порядка
        seen = set()
        hrefs = []
        for h in hrefs_raw:
            if h not in seen:
                seen.add(h)
                hrefs.append(h)

        # Основная ссылка на архив (первая archive-ссылка в карточке)
        archive_url = next((h for h in hrefs if _is_archive(h)), None)
        archive_url_abs = urljoin(base_url, archive_url) if archive_url else None
        archive_filename = _url_filename(archive_url) if archive_url else None
        archive_ext = _ext_from_url(archive_url) if archive_url else None

        # Превью-картинка
        img_tag = card.select_one(".model-image img")
        preview_img_url = _pick_preview_img(img_tag)
        preview_img_url_abs = urljoin(base_url, preview_img_url) if preview_img_url else None
        preview_img_alt = img_tag.get("alt") if img_tag else None
        preview_img_title = img_tag.get("title") if img_tag else None

        # Доп. данные (на случай будущих полей)
        extra = {
            "card_classes": card.get("class", []),
            "card_id": bx_id,
            "all_hrefs": hrefs,
        }
        extra_json = json.dumps(extra, ensure_ascii=False)

        raw_item_html = str(card)

        items.append(
            ItemRow(
                bx_id=bx_id,
                name=name,
                description=description,
                archive_url=archive_url,
                archive_url_abs=archive_url_abs,
                archive_filename=archive_filename,
                archive_ext=archive_ext,
                preview_img_url=preview_img_url,
                preview_img_url_abs=preview_img_url_abs,
                preview_img_alt=preview_img_alt,
                preview_img_title=preview_img_title,
                source_file=source_file,
                parsed_at=parsed_at,
                raw_item_html=raw_item_html,
                extra_json=extra_json,
            )
        )

        # Вторая таблица — ВСЕ ссылки из карточки
        for i, h in enumerate(hrefs):
            h_abs = urljoin(base_url, h)
            if _is_archive(h):
                kind = "archive"
            elif _is_image(h):
                kind = "image"
            else:
                kind = "other"
            links.append(LinkRow(bx_id=bx_id, url=h, url_abs=h_abs, kind=kind, pos=i))

        # Также добавим ссылки на изображения из img src/data-src, если их нет в href
        # (чтобы не потерять превью при каких-то изменениях шаблона)
        if preview_img_url and preview_img_url not in seen:
            links.append(
                LinkRow(
                    bx_id=bx_id,
                    url=preview_img_url,
                    url_abs=urljoin(base_url, preview_img_url),
                    kind="image",
                    pos=len(hrefs),
                )
            )

    return items, links


def upsert_to_db(db_path: Path, items: List[ItemRow], links: List[LinkRow]) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys=OFF;")

        con.executemany(
            """
            INSERT INTO imodern_item (
                bx_id, name, description,
                archive_url, archive_url_abs, archive_filename, archive_ext,
                preview_img_url, preview_img_url_abs, preview_img_alt, preview_img_title,
                source_file, parsed_at, raw_item_html, extra_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bx_id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                archive_url=excluded.archive_url,
                archive_url_abs=excluded.archive_url_abs,
                archive_filename=excluded.archive_filename,
                archive_ext=excluded.archive_ext,
                preview_img_url=excluded.preview_img_url,
                preview_img_url_abs=excluded.preview_img_url_abs,
                preview_img_alt=excluded.preview_img_alt,
                preview_img_title=excluded.preview_img_title,
                source_file=excluded.source_file,
                parsed_at=excluded.parsed_at,
                raw_item_html=excluded.raw_item_html,
                extra_json=excluded.extra_json
            ;
            """,
            [
                (
                    it.bx_id,
                    it.name,
                    it.description,
                    it.archive_url,
                    it.archive_url_abs,
                    it.archive_filename,
                    it.archive_ext,
                    it.preview_img_url,
                    it.preview_img_url_abs,
                    it.preview_img_alt,
                    it.preview_img_title,
                    it.source_file,
                    it.parsed_at,
                    it.raw_item_html,
                    it.extra_json,
                )
                for it in items
            ],
        )

        con.executemany(
            """
            INSERT OR IGNORE INTO imodern_link (bx_id, url, url_abs, kind, pos)
            VALUES (?, ?, ?, ?, ?);
            """,
            [(ln.bx_id, ln.url, ln.url_abs, ln.kind, ln.pos) for ln in links],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("html", type=str, help="Путь к links.html")
    ap.add_argument(
        "--db",
        type=str,
        default="src/data/sourse/imodern.db",
        help="Путь к SQLite БД (по умолчанию src/data/sourse/imodern.db)",
    )
    ap.add_argument(
        "--base",
        type=str,
        default="https://imodern.ru",
        help="Base URL для сборки абсолютных ссылок",
    )
    args = ap.parse_args()

    html_path = Path(args.html)
    db_path = Path(args.db)
    base_url = args.base.rstrip("/") + "/"

    if not html_path.exists():
        raise FileNotFoundError(f"Не найден файл: {html_path}")

    init_db(db_path)
    items, links = parse_html_file(html_path, base_url)
    upsert_to_db(db_path, items, links)

    archives = sum(1 for it in items if it.archive_url)
    print(f"Карточек найдено: {len(items)}")
    print(f"Карточек с архивом: {archives}")
    print(f"Ссылок (всех типов) добавлено/учтено: {len(links)}")
    print(f"База данных: {db_path.resolve()}")


if __name__ == "__main__":
    main()