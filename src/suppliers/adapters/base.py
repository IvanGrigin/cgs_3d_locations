# -*- coding: utf-8 -*-
"""
This module defines the shared adapter base class for supplier parsers.
It owns HTML fetching, partial-response fallback logic, and common extractors.
Concrete site adapters inherit these helpers and only implement site rules.
The goal is to keep network behavior and normalization consistent.
Changes here affect every supplier adapter.
"""
from __future__ import annotations

import abc
import json
import re
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs, unquote, urljoin

import requests
from bs4 import BeautifulSoup

from src.suppliers.models import ProductRecord
from src.suppliers.utils import DEFAULT_HEADERS, now_utc_iso


class SupplierAdapter(abc.ABC):
    site_name: str = "base"
    max_html_bytes: int = 5 * 1024 * 1024
    empty_parse_is_skip: bool = False

    def __init__(self, timeout: int = 45) -> None:
        self.timeout = timeout

    @abc.abstractmethod
    def can_handle(self, url: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        raise NotImplementedError

    def fetch_html(self, url: str) -> tuple[str, str]:
        last_error: Exception | None = None

        try:
            return self._fetch_html_via_requests(url)
        except Exception as e:
            last_error = e

        try:
            return self._fetch_html_via_curl(url)
        except Exception as e:
            if last_error is not None:
                raise RuntimeError(
                    f"Не удалось получить HTML ни через requests, ни через curl. "
                    f"requests: {last_error}; curl: {e}"
                ) from e
            raise

    def _fetch_html_via_requests(self, url: str) -> tuple[str, str]:
        with requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=(10, self.timeout),
            allow_redirects=True,
            stream=True,
        ) as r:
            r.raise_for_status()

            chunks: list[bytes] = []
            tail = b""
            total = 0

            try:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue

                    chunks.append(chunk)
                    total += len(chunk)
                    tail = (tail + chunk)[-16384:]

                    if self._has_enough_html_signals(tail):
                        break

                    if total >= self.max_html_bytes:
                        break

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ):
                raw = b"".join(chunks)
                if self._is_usable_partial_html(raw):
                    return self._decode_html(raw, r.encoding), r.url
                raise

            raw = b"".join(chunks)
            if not raw:
                raise RuntimeError("Пустой ответ при загрузке HTML через requests")

            return self._decode_html(raw, r.encoding), r.url

    def _fetch_html_via_curl(self, url: str) -> tuple[str, str]:
        marker = "__CODEX_EFFECTIVE_URL__:"
        command = [
            "curl",
            "-L",
            "--compressed",
            "--connect-timeout",
            "10",
            "--max-time",
            str(self.timeout),
            "--retry",
            "2",
            "--retry-delay",
            "1",
            "--retry-all-errors",
            "-A",
            DEFAULT_HEADERS["User-Agent"],
            "-H",
            f"Accept-Language: {DEFAULT_HEADERS['Accept-Language']}",
            "-o",
            "-",
            "-w",
            f"\n{marker}%{{url_effective}}",
            url,
        ]

        result = subprocess.run(command, capture_output=True, check=False)

        stdout = result.stdout or b""
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()

        marker_bytes = f"\n{marker}".encode("utf-8")
        idx = stdout.rfind(marker_bytes)

        if idx != -1:
            raw = stdout[:idx]
            final_url = stdout[idx + len(marker_bytes):].decode("utf-8", errors="replace").strip() or url
        else:
            raw = stdout
            final_url = url

        if raw and self._is_usable_partial_html(raw):
            return self._decode_html(raw, "utf-8"), final_url

        raise RuntimeError(
            f"curl не смог вернуть пригодный HTML. "
            f"returncode={result.returncode}; stderr={stderr[:500]}"
        )

    @staticmethod
    def _decode_html(raw: bytes, encoding: str | None) -> str:
        return raw.decode(encoding or "utf-8", errors="replace")

    @staticmethod
    def _has_enough_html_signals(tail: bytes) -> bool:
        low = tail.lower()
        signals = (
            b"</html>",
            b"</body>",
            b'class="page-catalog-product"',
            b'class="page-3d',
            b"catalog-product__offer-name",
            b"catalog-product__name",
            b"product-characteristics",
            b"catalog-product__3d-model__link-download",
            b"items-list-3d-models",
            b'item-link-image',
            b"download",
        )
        return any(x in low for x in signals)

    @staticmethod
    def _is_usable_partial_html(raw: bytes) -> bool:
        if len(raw) < 2048:
            return False

        head = raw[:32768].lower()
        body = raw.lower()

        has_html_shell = (
            b"<html" in head
            or b"<!doctype html" in head
            or b"<body" in head
        )
        if not has_html_shell:
            return False

        strong_signals = [
            b'class="page-catalog-product"',
            b'class="page-3d',
            b"catalog-product__offer-name",
            b"catalog-product__name",
            b"item-info-current-check-offer-price",
            b"product-characteristics",
            b"catalog-product__3d-model__link-download",
            b"catalog-product__3d-model__link-download",
            b"items-list-3d-models",
            b"item-link-image",
            b"item-price",
            b"bx_breadcrumbs",
        ]

        if any(x in body for x in strong_signals):
            return True

        if b"</body>" in body or b"</html>" in body:
            return True

        return False

    @staticmethod
    def now_utc_iso() -> str:
        return now_utc_iso()

    @staticmethod
    def norm_space(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def extract_meta_content(soup: BeautifulSoup, key: str) -> Optional[str]:
        node = soup.find("meta", attrs={"property": key})
        if node and node.get("content"):
            return SupplierAdapter.norm_space(node["content"])

        node = soup.find("meta", attrs={"name": key})
        if node and node.get("content"):
            return SupplierAdapter.norm_space(node["content"])

        return None

    @staticmethod
    def extract_jsonld_objects(soup: BeautifulSoup) -> list[dict]:
        out: list[dict] = []
        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = tag.get_text(strip=True)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            if isinstance(data, dict):
                out.append(data)
            elif isinstance(data, list):
                out.extend([x for x in data if isinstance(x, dict)])
        return out

    @staticmethod
    def extract_name_from_jsonld(soup: BeautifulSoup) -> Optional[str]:
        for item in SupplierAdapter.extract_jsonld_objects(soup):
            t = item.get("@type")
            if isinstance(t, list):
                ok = "Product" in t
            else:
                ok = t == "Product"
            if ok and item.get("name"):
                return SupplierAdapter.norm_space(str(item["name"]))
        return None

    @staticmethod
    def extract_brand_from_jsonld(soup: BeautifulSoup) -> Optional[str]:
        for item in SupplierAdapter.extract_jsonld_objects(soup):
            t = item.get("@type")
            if isinstance(t, list):
                ok = "Product" in t
            else:
                ok = t == "Product"
            if not ok:
                continue

            brand = item.get("brand")
            if isinstance(brand, dict) and brand.get("name"):
                return SupplierAdapter.norm_space(str(brand["name"]))
            if isinstance(brand, str):
                return SupplierAdapter.norm_space(brand)

        return None

    @staticmethod
    def parse_price_rub(text: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
        clean = text.replace("\xa0", " ")
        nums = re.findall(r"(\d[\d\s]{2,})\s*₽", clean)

        values: list[float] = []
        for x in nums:
            try:
                values.append(float(re.sub(r"\s+", "", x)))
            except Exception:
                pass

        if not values:
            return None, None, None
        if len(values) == 1:
            return values[0], None, "RUB"
        return values[-1], values[0], "RUB"

    @staticmethod
    def parse_dimension_cm(page_text: str, label: str) -> Optional[float]:
        m = re.search(
            rf"{re.escape(label)}\s+([0-9]+(?:[.,][0-9]+)?)\s*см",
            page_text,
            re.IGNORECASE,
        )
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    @staticmethod
    def parse_weight_kg(page_text: str) -> Optional[float]:
        m = re.search(
            r"Вес(?: в упаковке)?\s+([0-9]+(?:[.,][0-9]+)?)\s*кг",
            page_text,
            re.IGNORECASE,
        )
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    @staticmethod
    def extract_images(soup: BeautifulSoup, base_url: str, limit: int = 50) -> list[str]:
        out: list[str] = []
        for img in soup.find_all("img"):
            src = img.get("data-src") or img.get("src")
            if not src:
                continue
            out.append(urljoin(base_url, src))

        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq[:limit]

    @staticmethod
    def extract_onclick_download(tag, base_url: str) -> Optional[str]:
        onclick = tag.get("onclick")
        if not onclick:
            return None

        m = re.search(r"""window\.location\.href\s*=\s*['"]([^'"]+)['"]""", onclick)
        if m:
            return urljoin(base_url, m.group(1))

        m = re.search(r"""location\.href\s*=\s*['"]([^'"]+)['"]""", onclick)
        if m:
            return urljoin(base_url, m.group(1))

        return None

    @staticmethod
    def ext_from_url(url: str | None) -> Optional[str]:
        if not url:
            return None

        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()
        if ext:
            return ext

        qs = parse_qs(parsed.query)
        if "filename" in qs and qs["filename"]:
            return Path(qs["filename"][0]).suffix.lower() or None

        return None

    @staticmethod
    def filename_from_url(url: str | None) -> Optional[str]:
        if not url:
            return None

        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "filename" in qs and qs["filename"]:
            return unquote(qs["filename"][0])

        name = Path(parsed.path).name
        return unquote(name) if name else None

    @staticmethod
    def classify_category(title: str | None) -> Optional[str]:
        if not title:
            return None

        t = title.lower()

        if "диван" in t or "sofa" in t or "couch" in t:
            return "sofa"
        if "кресло" in t or "кресла" in t or "arm chair" in t or "armchair" in t:
            return "armchair"
        if "стул" in t or "chair" in t:
            return "chair"
        if "кровать" in t or "кровати" in t or "bed" in t:
            return "bed"
        if "журнальный стол" in t or "coffee table" in t:
            return "coffee_table"
        if "стол" in t or "table" in t or "desk" in t:
            return "table"
        if "тумба" in t or "nightstand" in t or "cabinet" in t:
            return "cabinet"
        if "комод" in t or "сервант" in t or "sideboard" in t or "chest of drawer" in t:
            return "sideboard"
        if "стеллаж" in t or "шкаф" in t or "bookcase" in t or "wardrobe" in t:
            return "bookcase"
        if (
            "люстра" in t or "бра" in t or "светильник" in t or "торшер" in t
            or "подвесной" in t
            or "lamp" in t or "light" in t or "sconce" in t or "chandelier" in t
        ):
            return "lamp"
        if "зеркало" in t or "mirror" in t:
            return "mirror"
        if "другая мягкая мебель" in t or "other soft seating" in t:
            return "ottoman"
        if "пуфик" in t or "банкетка" in t or "ottoman" in t or "bench" in t:
            return "ottoman"
        if "растение" in t or "дерево" in t or "plant" in t or "tree" in t:
            return "plant"
        if "кашпо" in t or "planter" in t:
            return "planter"

        return None

    def build_unique_key(self, final_url: str, external_id: str | None) -> str:
        if external_id:
            return f"{self.site_name}::id::{external_id}"
        return f"{self.site_name}::url::{final_url}"


# Обратная совместимость
BaseAdapter = SupplierAdapter
