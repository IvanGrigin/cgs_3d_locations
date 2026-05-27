#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/tools/translate_catalog_de_to_ru_en_googletrans.py

Pragmatic translator for supplier catalog JSON using googletrans.

What it does:
- reads supplier_catalog_canonical.json or another JSON catalog;
- adds *_en and *_ru fields near selected German/source fields;
- preserves original JSON structure;
- uses a persistent cache so repeated/interrupted runs do not retranslate the same strings;
- supports smoke runs with --limit;
- uses one asyncio event loop for googletrans 4.0.2, avoiding "Event loop is closed" errors;
- translates only product core fields by default; tags and heavy extra.ikea_de fields are opt-in.

Installation:
    python3 -m pip install -U googletrans tqdm

Smoke test:
    python3 src/tools/translate_catalog_de_to_ru_en_googletrans.py \
      --input data/sourse/suppliers/supplier_catalog_canonical.json \
      --output /tmp/supplier_catalog_test.googletrans.translated.json \
      --limit 10 \
      --batch-size 20 \
      --skip-existing

Dry-run stats:
    python3 src/tools/translate_catalog_de_to_ru_en_googletrans.py \
      --input data/sourse/suppliers/supplier_catalog_canonical.json \
      --output /tmp/unused.json \
      --dry-run-stats

Full product-core run:
    python3 src/tools/translate_catalog_de_to_ru_en_googletrans.py \
      --input data/sourse/suppliers/supplier_catalog_canonical.json \
      --output data/sourse/suppliers/supplier_catalog_canonical.translated.json \
      --batch-size 50 \
      --sleep-between-batches 0.2 \
      --skip-existing \
      --copy-source-on-fail

Optional heavier run with tags and IKEA extra fields:
    python3 src/tools/translate_catalog_de_to_ru_en_googletrans.py \
      --input data/sourse/suppliers/supplier_catalog_canonical.json \
      --output data/sourse/suppliers/supplier_catalog_canonical.translated.json \
      --batch-size 30 \
      --include-tags \
      --include-extra-ikea \
      --skip-existing \
      --copy-source-on-fail

Notes:
- googletrans is unofficial and can be rate-limited or broken by upstream changes.
- For 30k+ products do not use --profile all_strings unless you deliberately want a very long run.
- This script stores cache incrementally; after interruption rerun with the same --output/--cache and --skip-existing.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from googletrans import Translator
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[error] googletrans is not installed. Run: python3 -m pip install -U googletrans"
    ) from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

URL_OR_ID_KEY_RE = re.compile(
    r"(?:^|_)(?:url|uri|path|file|filename|id|key|hash|digest|currency|availability|timestamp|parsed_at|created_at|updated_at)(?:$|_)",
    re.IGNORECASE,
)

TECHNICAL_STRING_RE = re.compile(r"^[\d\s.,:;_+\-/%×x()]+$")

PRODUCT_CORE_STRING_FIELDS = [
    "title",
    "category_raw",
    "style",
    "color",
    "description",
    "materials",
]

# Intentionally empty by default. Tags are very numerous and often duplicate noisy text.
PRODUCT_CORE_LIST_FIELDS_DEFAULT: List[str] = []
PRODUCT_CORE_LIST_FIELDS_WITH_TAGS = ["tags"]

IKEA_EXTRA_STRING_FIELDS = [
    "image_alt",
    "type_name",
    "item_measure_reference_text",
]

IKEA_EXTRA_LIST_FIELDS = [
    "info_texts",
]


# -----------------------------------------------------------------------------
# Data refs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslationRef:
    parent: Dict[str, Any]
    key: str
    value: str

    @property
    def en_key(self) -> str:
        return f"{self.key}_en"

    @property
    def ru_key(self) -> str:
        return f"{self.key}_ru"


@dataclass
class TranslationStats:
    de_en_seconds: float = 0.0
    en_ru_seconds: float = 0.0
    batches_done: int = 0
    strings_done: int = 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate German/source catalog fields to English and Russian using googletrans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of catalog items for smoke tests.")
    parser.add_argument(
        "--profile",
        choices=("product_core", "all_strings"),
        default="product_core",
        help="Which fields to translate.",
    )
    parser.add_argument("--src-lang", default="de", help="Source language for googletrans.")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of unique strings per googletrans batch.")
    parser.add_argument("--cache", default=None, help="Cache JSON path. Default: output + '.cache.json'.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip field if both *_en and *_ru already exist.")
    parser.add_argument("--dry-run-stats", action="store_true", help="Only collect stats, do not translate/write output.")
    parser.add_argument("--max-chars-per-string", type=int, default=4500, help="Split long strings before translation.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per batch/string on googletrans errors.")
    parser.add_argument("--sleep-between-batches", type=float, default=0.0, help="Sleep after each batch in seconds.")
    parser.add_argument("--sleep-on-error", type=float, default=2.0, help="Base sleep after failed request.")
    parser.add_argument("--service-url", action="append", default=None, help="Custom Google Translate service URL; may be repeated.")
    parser.add_argument("--copy-source-on-fail", action="store_true", help="Use source text as translation if all retries fail.")
    parser.add_argument("--include-tags", action="store_true", help="Also translate tags list into tags_en/tags_ru.")
    parser.add_argument("--include-extra-ikea", action="store_true", help="Also translate extra.ikea_de selected fields.")
    parser.add_argument("--include-measurements", action="store_true", help="Also translate extra.ikea_de.measurements_raw[].name.")
    parser.add_argument("--cache-save-every", type=int, default=5, help="Save cache every N missing-string batches.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def limit_items_in_original_structure(data: Any, limit: Optional[int]) -> Any:
    if limit is None or limit <= 0:
        return data
    if isinstance(data, list):
        return data[:limit]
    if isinstance(data, dict):
        out = copy.deepcopy(data)
        if isinstance(out.get("items"), list):
            out["items"] = out["items"][:limit]
            return out
        if isinstance(out.get("data"), list):
            out["data"] = out["data"][:limit]
            return out
    return data


def get_items_container(data: Any) -> Tuple[List[Dict[str, Any]], str]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)], "list"
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return [x for x in data["items"] if isinstance(x, dict)], "items"
        if isinstance(data.get("data"), list):
            return [x for x in data["data"] if isinstance(x, dict)], "data"
        return [data], "single"
    raise ValueError("Input JSON must be a dict or list.")


# -----------------------------------------------------------------------------
# Field selection
# -----------------------------------------------------------------------------


def is_probably_url_or_path(value: str) -> bool:
    s = value.strip()
    if not s:
        return True
    if s.startswith(("http://", "https://", "s3://", "gs://", "file://")):
        return True
    if "/" in s and re.search(r"\.(?:jpg|jpeg|png|webp|glb|gltf|obj|fbx|bin|json|xml)(?:\?|$)", s, re.I):
        return True
    if re.fullmatch(r"[a-f0-9]{16,}", s, flags=re.I):
        return True
    return False


def is_translatable_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if len(s) < 2:
        return False
    if is_probably_url_or_path(s):
        return False
    if TECHNICAL_STRING_RE.fullmatch(s):
        return False
    return True


def add_ref(refs: List[TranslationRef], parent: Dict[str, Any], key: str, *, skip_existing: bool) -> None:
    value = parent.get(key)
    if not is_translatable_value(value):
        return
    if skip_existing and isinstance(parent.get(f"{key}_en"), str) and isinstance(parent.get(f"{key}_ru"), str):
        return
    refs.append(TranslationRef(parent=parent, key=key, value=str(value)))


def add_list_ref(refs: List[TranslationRef], parent: Dict[str, Any], key: str, *, skip_existing: bool) -> None:
    value = parent.get(key)
    if not isinstance(value, list):
        return
    if skip_existing and isinstance(parent.get(f"{key}_en"), list) and isinstance(parent.get(f"{key}_ru"), list):
        return
    for idx, item in enumerate(value):
        if is_translatable_value(item):
            synthetic_parent = {"__list_parent__": parent, "__list_key__": key, "__list_index__": idx}
            refs.append(TranslationRef(parent=synthetic_parent, key="value", value=str(item)))


def collect_product_core_refs(
    items: Sequence[Dict[str, Any]],
    *,
    skip_existing: bool,
    include_tags: bool,
    include_extra_ikea: bool,
    include_measurements: bool,
) -> List[TranslationRef]:
    refs: List[TranslationRef] = []
    list_fields = PRODUCT_CORE_LIST_FIELDS_WITH_TAGS if include_tags else PRODUCT_CORE_LIST_FIELDS_DEFAULT

    for item in items:
        for key in PRODUCT_CORE_STRING_FIELDS:
            add_ref(refs, item, key, skip_existing=skip_existing)
        for key in list_fields:
            add_list_ref(refs, item, key, skip_existing=skip_existing)

        if not include_extra_ikea:
            continue

        extra = item.get("extra")
        if isinstance(extra, dict):
            ikea = extra.get("ikea_de")
            if isinstance(ikea, dict):
                for key in IKEA_EXTRA_STRING_FIELDS:
                    add_ref(refs, ikea, key, skip_existing=skip_existing)
                for key in IKEA_EXTRA_LIST_FIELDS:
                    add_list_ref(refs, ikea, key, skip_existing=skip_existing)

                if include_measurements:
                    measurements = ikea.get("measurements_raw")
                    if isinstance(measurements, list):
                        for m in measurements:
                            if isinstance(m, dict):
                                add_ref(refs, m, "name", skip_existing=skip_existing)
    return refs


def collect_all_string_refs(data: Any, *, skip_existing: bool) -> List[TranslationRef]:
    refs: List[TranslationRef] = []

    def rec(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key.endswith("_en") or key.endswith("_ru"):
                    continue
                if URL_OR_ID_KEY_RE.search(key):
                    continue
                if isinstance(value, str):
                    add_ref(refs, node, key, skip_existing=skip_existing)
                elif isinstance(value, list):
                    add_list_ref(refs, node, key, skip_existing=skip_existing)
                    for x in value:
                        if isinstance(x, (dict, list)):
                            rec(x)
                elif isinstance(value, dict):
                    rec(value)
        elif isinstance(node, list):
            for x in node:
                rec(x)

    rec(data)
    return refs


def collect_refs(data: Any, args: argparse.Namespace) -> Tuple[List[TranslationRef], int, str]:
    items, container = get_items_container(data)
    if args.profile == "product_core":
        refs = collect_product_core_refs(
            items,
            skip_existing=args.skip_existing,
            include_tags=args.include_tags,
            include_extra_ikea=args.include_extra_ikea,
            include_measurements=args.include_measurements,
        )
    elif args.profile == "all_strings":
        refs = collect_all_string_refs(data, skip_existing=args.skip_existing)
    else:
        raise ValueError(f"unknown profile: {args.profile}")
    return refs, len(items), container


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------


def load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for key, val in data.items():
        if isinstance(key, str) and isinstance(val, dict):
            en = val.get("en")
            ru = val.get("ru")
            if isinstance(en, str) and isinstance(ru, str):
                out[key] = {"en": en, "ru": ru}
    return out


def save_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    write_json(path, cache)


# -----------------------------------------------------------------------------
# Long-string splitting
# -----------------------------------------------------------------------------


def split_long_text(text: str, max_chars: int) -> List[str]:
    s = text.strip()
    if len(s) <= max_chars:
        return [s]

    parts = re.split(r"(?<=[.!?;])\s+", s)
    chunks: List[str] = []
    cur = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            for i in range(0, len(part), max_chars):
                chunks.append(part[i : i + max_chars].strip())
            continue
        if not cur:
            cur = part
        elif len(cur) + 1 + len(part) <= max_chars:
            cur += " " + part
        else:
            chunks.append(cur.strip())
            cur = part
    if cur:
        chunks.append(cur.strip())
    return [x for x in chunks if x]


def expand_long_strings(strings: Sequence[str], max_chars: int) -> Tuple[List[str], Dict[str, List[str]]]:
    expanded: List[str] = []
    mapping: Dict[str, List[str]] = {}
    seen = set()
    for s in strings:
        chunks = split_long_text(s, max_chars)
        mapping[s] = chunks
        for c in chunks:
            if c not in seen:
                expanded.append(c)
                seen.add(c)
    return expanded, mapping


def join_chunk_translations(strings: Sequence[str], mapping: Dict[str, List[str]], translated: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in strings:
        chunks = mapping.get(s, [s])
        out[s] = " ".join(translated.get(c, c) for c in chunks).strip()
    return out


# -----------------------------------------------------------------------------
# googletrans async compatibility layer
# -----------------------------------------------------------------------------


def make_translator(service_urls: Optional[List[str]]) -> Any:
    if service_urls:
        return Translator(service_urls=service_urls)
    return Translator()


async def maybe_await_async(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(result, dict):
        text = result.get("text") or result.get("translatedText")
        if isinstance(text, str):
            return text
    return str(result)


async def translate_call(translator: Any, texts: Sequence[str] | str, *, src: str, dest: str) -> Any:
    return await maybe_await_async(translator.translate(texts, src=src, dest=dest))


async def close_translator(translator: Any) -> None:
    candidates: List[Any] = []
    for name in ("aclose", "close"):
        fn = getattr(translator, name, None)
        if callable(fn):
            candidates.append(fn)
    client = getattr(translator, "client", None)
    if client is not None:
        for name in ("aclose", "close"):
            fn = getattr(client, name, None)
            if callable(fn):
                candidates.append(fn)

    for fn in candidates:
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
            return
        except Exception:
            continue


async def translate_one_googletrans(
    translator: Any,
    text: str,
    *,
    src: str,
    dest: str,
    max_retries: int,
    sleep_on_error: float,
    copy_source_on_fail: bool,
) -> str:
    for attempt in range(max(1, max_retries)):
        try:
            result = await translate_call(translator, text, src=src, dest=dest)
            return extract_text(result)
        except Exception as exc:
            if attempt + 1 >= max_retries:
                print(f"[error] translate failed {src}->{dest}: {text[:120]!r}: {exc}", file=sys.stderr, flush=True)
                return text if copy_source_on_fail else ""
            delay = sleep_on_error * (attempt + 1) + random.random() * 0.25
            print(
                f"[warn] string {src}->{dest} failed attempt={attempt + 1}: {exc}; sleep={delay:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)
    return text if copy_source_on_fail else ""


async def translate_batch_googletrans(
    translator: Any,
    texts: Sequence[str],
    *,
    src: str,
    dest: str,
    max_retries: int,
    sleep_on_error: float,
    copy_source_on_fail: bool,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not texts:
        return out

    for attempt in range(max(1, max_retries)):
        try:
            result = await translate_call(translator, list(texts), src=src, dest=dest)
            if isinstance(result, list):
                if len(result) != len(texts):
                    raise RuntimeError(f"batch returned {len(result)} results for {len(texts)} inputs")
                for src_text, item in zip(texts, result):
                    out[src_text] = extract_text(item)
                return out

            # Some versions return one object even for one input.
            if len(texts) == 1:
                out[texts[0]] = extract_text(result)
                return out

            raise RuntimeError(f"unexpected googletrans batch result type: {type(result).__name__}")
        except Exception as exc:
            if attempt + 1 >= max_retries:
                break
            delay = sleep_on_error * (attempt + 1) + random.random() * 0.25
            print(
                f"[warn] batch {src}->{dest} failed attempt={attempt + 1}: {exc}; sleep={delay:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)

    # Fallback: per string, still inside the same event loop.
    for text in texts:
        out[text] = await translate_one_googletrans(
            translator,
            text,
            src=src,
            dest=dest,
            max_retries=max_retries,
            sleep_on_error=sleep_on_error,
            copy_source_on_fail=copy_source_on_fail,
        )
    return out


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    n = max(1, batch_size)
    for i in range(0, len(items), n):
        yield list(items[i : i + n])


def progress(iterable: Iterable[Any], *, total: int, desc: str) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


async def translate_many_googletrans(
    translator: Any,
    strings: Sequence[str],
    *,
    src: str,
    dest: str,
    batch_size: int,
    max_retries: int,
    sleep_between_batches: float,
    sleep_on_error: float,
    copy_source_on_fail: bool,
    desc: str,
) -> Dict[str, str]:
    unique = list(dict.fromkeys(strings))
    batches = list(iter_batches(unique, batch_size))
    out: Dict[str, str] = {}

    for batch in progress(batches, total=len(batches), desc=desc):
        translated = await translate_batch_googletrans(
            translator,
            batch,
            src=src,
            dest=dest,
            max_retries=max_retries,
            sleep_on_error=sleep_on_error,
            copy_source_on_fail=copy_source_on_fail,
        )
        out.update(translated)
        if sleep_between_batches > 0:
            await asyncio.sleep(sleep_between_batches)

    return out


async def translate_missing_progressive(
    translator: Any,
    missing: Sequence[str],
    cache: Dict[str, Dict[str, str]],
    cache_path: Path,
    args: argparse.Namespace,
) -> TranslationStats:
    stats = TranslationStats()
    unique_missing = list(dict.fromkeys(missing))
    outer_batches = list(iter_batches(unique_missing, args.batch_size))

    for batch_index, original_batch in enumerate(progress(outer_batches, total=len(outer_batches), desc="missing"), start=1):
        # A source string may be long. Translate chunks, then reconstruct the full string.
        expanded_de, de_map = expand_long_strings(original_batch, args.max_chars_per_string)

        t0 = time.perf_counter()
        de_en_chunks = await translate_many_googletrans(
            translator,
            expanded_de,
            src=args.src_lang,
            dest="en",
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            sleep_between_batches=args.sleep_between_batches,
            sleep_on_error=args.sleep_on_error,
            copy_source_on_fail=args.copy_source_on_fail,
            desc="de->en",
        )
        de_en_full = join_chunk_translations(original_batch, de_map, de_en_chunks)
        t1 = time.perf_counter()

        en_inputs = list(dict.fromkeys(de_en_full.values()))
        expanded_en, en_map = expand_long_strings(en_inputs, args.max_chars_per_string)
        en_ru_chunks = await translate_many_googletrans(
            translator,
            expanded_en,
            src="en",
            dest="ru",
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            sleep_between_batches=args.sleep_between_batches,
            sleep_on_error=args.sleep_on_error,
            copy_source_on_fail=args.copy_source_on_fail,
            desc="en->ru",
        )
        en_ru_full = join_chunk_translations(en_inputs, en_map, en_ru_chunks)
        t2 = time.perf_counter()

        for src_text in original_batch:
            en = de_en_full.get(src_text, src_text if args.copy_source_on_fail else "")
            ru = en_ru_full.get(en, src_text if args.copy_source_on_fail else "")
            cache[src_text] = {"en": en, "ru": ru}

        stats.de_en_seconds += t1 - t0
        stats.en_ru_seconds += t2 - t1
        stats.batches_done += 1
        stats.strings_done += len(original_batch)

        if args.cache_save_every > 0 and batch_index % args.cache_save_every == 0:
            save_cache(cache_path, cache)
            print(f"[cache] saved incremental entries={len(cache)}", flush=True)

    save_cache(cache_path, cache)
    return stats


# -----------------------------------------------------------------------------
# Apply translations
# -----------------------------------------------------------------------------


def ensure_list_translation_storage(parent: Dict[str, Any], key: str) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    value = parent.get(key)
    n = len(value) if isinstance(value, list) else 0
    en_key = f"{key}_en"
    ru_key = f"{key}_ru"
    en = parent.get(en_key)
    ru = parent.get(ru_key)
    if not isinstance(en, list) or len(en) != n:
        en = [None] * n
        parent[en_key] = en
    if not isinstance(ru, list) or len(ru) != n:
        ru = [None] * n
        parent[ru_key] = ru
    return en, ru


def apply_translations(refs: Sequence[TranslationRef], cache: Dict[str, Dict[str, str]]) -> int:
    applied = 0
    for ref in refs:
        tr = cache.get(ref.value)
        if not isinstance(tr, dict):
            continue
        en = tr.get("en")
        ru = tr.get("ru")
        if not isinstance(en, str) or not isinstance(ru, str):
            continue

        if "__list_parent__" in ref.parent:
            parent = ref.parent.get("__list_parent__")
            key = ref.parent.get("__list_key__")
            idx = ref.parent.get("__list_index__")
            if isinstance(parent, dict) and isinstance(key, str) and isinstance(idx, int):
                en_list, ru_list = ensure_list_translation_storage(parent, key)
                if 0 <= idx < len(en_list):
                    en_list[idx] = en
                    ru_list[idx] = ru
                    applied += 1
            continue

        ref.parent[ref.en_key] = en
        ref.parent[ref.ru_key] = ru
        applied += 1
    return applied


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


async def main_async() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve() if args.cache else Path(str(output_path) + ".cache.json")

    data = read_json(input_path)
    data = limit_items_in_original_structure(data, args.limit)

    refs, item_count, container = collect_refs(data, args)
    raw_strings = [r.value for r in refs]
    unique_strings = list(dict.fromkeys(raw_strings))

    print(f"[input] {input_path}", flush=True)
    print(f"[items] {item_count} container={container}", flush=True)
    print(
        f"[collect] refs={len(refs)} raw_strings={len(raw_strings)} unique_strings={len(unique_strings)} "
        f"profile={args.profile} include_tags={args.include_tags} include_extra_ikea={args.include_extra_ikea}",
        flush=True,
    )

    if args.dry_run_stats:
        print("[dry-run] translator was not loaded; output was not written", flush=True)
        return

    cache = load_cache(cache_path)
    hit = sum(1 for s in unique_strings if s in cache)
    missing = [s for s in unique_strings if s not in cache]

    print(f"[cache] {cache_path} entries={len(cache)}", flush=True)
    print(f"[cache] hit={hit} missing={len(missing)}", flush=True)

    stats = TranslationStats()
    translator = None
    try:
        if missing:
            translator = make_translator(args.service_url)
            stats = await translate_missing_progressive(translator, missing, cache, cache_path, args)
        else:
            print("[translate] nothing missing; using cache only", flush=True)
    finally:
        if translator is not None:
            await close_translator(translator)

    applied = apply_translations(refs, cache)
    write_json(output_path, data)

    print(f"[translate] de->en seconds={stats.de_en_seconds:.3f}", flush=True)
    print(f"[translate] en->ru seconds={stats.en_ru_seconds:.3f}", flush=True)
    print(f"[translate] batches_done={stats.batches_done} strings_done={stats.strings_done}", flush=True)
    print(f"[done] applied_refs={applied}", flush=True)
    print(f"[done] output={output_path}", flush=True)
    print(f"[done] cache={cache_path}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
