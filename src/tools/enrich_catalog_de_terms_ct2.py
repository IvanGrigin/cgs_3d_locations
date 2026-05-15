#!/usr/bin/env python3
"""Enrich German supplier catalog fields with short EN/RU search translations.

This is intentionally not a full JSON translator.  It translates only compact
product fields, prefers a furniture term dictionary, falls back to local CT2
Marian models, and leaves heavyweight supplier metadata untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ctranslate2
from transformers import AutoTokenizer


SHORT_FIELDS = ("title", "category_raw", "color", "materials")
TEXT_LIMIT = 300

GERMAN_FURNITURE_TERMS: Dict[str, Dict[str, str]] = {
    "Abfallsammler": {"en": "waste bin", "ru": "мусорное ведро"},
    "Arbeitsplatte": {"en": "worktop", "ru": "столешница"},
    "Aufbewahrung": {"en": "storage", "ru": "хранение"},
    "Badezimmer": {"en": "bathroom", "ru": "ванная комната"},
    "Bett": {"en": "bed", "ru": "кровать"},
    "Bettgestell": {"en": "bed frame", "ru": "каркас кровати"},
    "Bezug": {"en": "cover", "ru": "чехол"},
    "Birke": {"en": "birch", "ru": "берёза"},
    "Buche": {"en": "beech", "ru": "бук"},
    "Drehstuhl": {"en": "swivel chair", "ru": "вращающийся стул"},
    "Eiche": {"en": "oak", "ru": "дуб"},
    "Einlegeboden": {"en": "shelf", "ru": "полка"},
    "Esstisch": {"en": "dining table", "ru": "обеденный стол"},
    "Front": {"en": "front", "ru": "фасад"},
    "Garderobe": {"en": "coat rack", "ru": "вешалка"},
    "grau": {"en": "grey", "ru": "серый"},
    "graugrün": {"en": "grey-green", "ru": "серо-зелёный"},
    "Griff": {"en": "handle", "ru": "ручка"},
    "grün": {"en": "green", "ru": "зелёный"},
    "Hochglanz": {"en": "high-gloss", "ru": "глянцевый"},
    "Kissen": {"en": "cushion", "ru": "подушка"},
    "Kommode": {"en": "chest of drawers", "ru": "комод"},
    "Küchenfronten": {"en": "kitchen fronts", "ru": "кухонные фасады"},
    "Kunststoff": {"en": "plastic", "ru": "пластик"},
    "Kunststofffolie": {"en": "plastic foil", "ru": "пластиковая плёнка"},
    "Hängelampe": {"en": "pendant lamp", "ru": "подвесной светильник"},
    "Hängelampen": {"en": "pendant lamps", "ru": "подвесные светильники"},
    "Hängeleuchte": {"en": "pendant light", "ru": "подвесной светильник"},
    "Hängeleuchten": {"en": "pendant lights", "ru": "подвесные светильники"},
    "Hängeleuchten & Hängelampen": {"en": "pendant lights and lamps", "ru": "подвесные светильники"},
    "Lack": {"en": "lacquer", "ru": "лак"},
    "Lampe": {"en": "lamp", "ru": "лампа"},
    "Leuchte": {"en": "light", "ru": "светильник"},
    "Matratze": {"en": "mattress", "ru": "матрас"},
    "Metall": {"en": "metal", "ru": "металл"},
    "Nachttisch": {"en": "bedside table", "ru": "прикроватная тумба"},
    "Polypropylenkunststoff": {"en": "polypropylene plastic", "ru": "полипропиленовый пластик"},
    "Regal": {"en": "shelving unit", "ru": "стеллаж"},
    "Schrank": {"en": "cabinet", "ru": "шкаф"},
    "Schreibtisch": {"en": "desk", "ru": "письменный стол"},
    "Schublade": {"en": "drawer", "ru": "ящик"},
    "Schubladenfront": {"en": "drawer front", "ru": "фасад ящика"},
    "Sessel": {"en": "armchair", "ru": "кресло"},
    "Sitzkissen": {"en": "seat cushion", "ru": "подушка на сиденье"},
    "Sofa": {"en": "sofa", "ru": "диван"},
    "Spanplatte": {"en": "particleboard", "ru": "ДСП"},
    "Spiegel": {"en": "mirror", "ru": "зеркало"},
    "Stahl": {"en": "steel", "ru": "сталь"},
    "Stuhl": {"en": "chair", "ru": "стул"},
    "Tisch": {"en": "table", "ru": "стол"},
    "Tür": {"en": "door", "ru": "дверца"},
    "weiß": {"en": "white", "ru": "белый"},
    "dunkel graugrün": {"en": "dark grey-green", "ru": "тёмный серо-зелёный"},
    "dunkelgrau": {"en": "dark grey", "ru": "тёмно-серый"},
    "hellgrau": {"en": "light grey", "ru": "светло-серый"},
    "schwarz": {"en": "black", "ru": "чёрный"},
}

BRAND_TOKEN_RE = re.compile(r"^[A-ZÅÄÖÜÆØ0-9][A-ZÅÄÖÜÆØ0-9'._-]{1,}$")
UPPER_LETTER_RE = re.compile(r"[A-ZÅÄÖÜÆØ]")
WORD_RE = re.compile(r"(?<!\w)[A-ZÅÄÖÜÆØ0-9][A-ZÅÄÖÜÆØ0-9'._-]{1,}(?!\w)")
SPACING_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TECHNICAL_RE = re.compile(r"^[\d\s.,:;_+\-/%×x()]+$")


@dataclass
class CT2Pair:
    de_en_tokenizer: Any
    en_ru_tokenizer: Any
    de_en: ctranslate2.Translator
    en_ru: ctranslate2.Translator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add short de/en/ru furniture search fields to a supplier catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    parser.add_argument("--output", default="data/sourse/suppliers/supplier_catalog_canonical.enriched_de_en_ru.json")
    parser.add_argument("--cache", default="data/sourse/suppliers/supplier_catalog_canonical.enriched_de_en_ru.cache.json")
    parser.add_argument("--de-en-model", default="models/ct2/opus-mt-de-en-int8")
    parser.add_argument("--en-ru-model", default="models/ct2/opus-mt-en-ru-int8")
    parser.add_argument("--de-en-tokenizer", default="Helsinki-NLP/opus-mt-de-en")
    parser.add_argument("--en-ru-tokenizer", default="Helsinki-NLP/opus-mt-en-ru")
    parser.add_argument("--source-site-contains", default="ikea.com/de/de")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-save-every", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=TEXT_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


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


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return SPACING_RE.sub(" ", value.replace("\u00a0", " ")).strip()


def is_short_translatable(text: str, max_chars: int) -> bool:
    if len(text) < 2 or len(text) > max_chars:
        return False
    if TECHNICAL_RE.fullmatch(text):
        return False
    if text.startswith(("http://", "https://")):
        return False
    return True


def get_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [x for x in data["items"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError("Input catalog must be a JSON object with items[] or a JSON list.")


def matches_source(item: Dict[str, Any], needle: str) -> bool:
    if not needle:
        return True
    source_site = normalize_text(item.get("source_site"))
    product_url = normalize_text(item.get("product_url") or item.get("source_url"))
    return needle in source_site or needle in product_url


def first_sentence(text: str, max_chars: int = 220) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    sentence = SENTENCE_SPLIT_RE.split(text, maxsplit=1)[0].strip()
    if len(sentence) <= max_chars:
        return sentence
    cut = sentence[:max_chars].rsplit(" ", 1)[0].strip()
    return cut or sentence[:max_chars].strip()


def item_phrases(item: Dict[str, Any], max_chars: int) -> List[str]:
    phrases: List[str] = []
    for field in SHORT_FIELDS:
        text = normalize_text(item.get(field))
        if is_short_translatable(text, max_chars):
            phrases.append(text)
    description_short = first_sentence(normalize_text(item.get("description")))
    if is_short_translatable(description_short, max_chars):
        phrases.append(description_short)
    return list(dict.fromkeys(phrases))


def load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, dict):
            en = value.get("en")
            ru = value.get("ru")
            if isinstance(en, str) and isinstance(ru, str):
                out[key] = {"en": en, "ru": ru}
    return out


def term_translation(text: str) -> Optional[Dict[str, str]]:
    exact = GERMAN_FURNITURE_TERMS.get(text)
    if exact:
        return exact

    tokens = text.split()
    leading_brand_tokens: List[str] = []
    rest_tokens = list(tokens)
    while rest_tokens and BRAND_TOKEN_RE.fullmatch(rest_tokens[0]) and UPPER_LETTER_RE.search(rest_tokens[0]):
        leading_brand_tokens.append(rest_tokens.pop(0))
    if leading_brand_tokens and rest_tokens:
        rest = " ".join(rest_tokens)
        known_rest = GERMAN_FURNITURE_TERMS.get(rest)
        if known_rest:
            prefix = " ".join(leading_brand_tokens)
            return {
                "en": f"{prefix} {known_rest['en']}",
                "ru": f"{prefix} {known_rest['ru']}",
            }

    parts = [normalize_text(x) for x in re.split(r"[,/;]+", text) if normalize_text(x)]
    if len(parts) > 1 and all(part in GERMAN_FURNITURE_TERMS for part in parts):
        return {
            "en": ", ".join(GERMAN_FURNITURE_TERMS[part]["en"] for part in parts),
            "ru": ", ".join(GERMAN_FURNITURE_TERMS[part]["ru"] for part in parts),
        }
    return None


def refresh_known_term_cache(phrases: Sequence[str], cache: Dict[str, Dict[str, str]]) -> int:
    refreshed = 0
    for phrase in phrases:
        known = term_translation(phrase)
        if known and cache.get(phrase) != known:
            cache[phrase] = known
            refreshed += 1
    return refreshed


def load_ct2(args: argparse.Namespace) -> CT2Pair:
    return CT2Pair(
        de_en_tokenizer=AutoTokenizer.from_pretrained(args.de_en_tokenizer, local_files_only=True),
        en_ru_tokenizer=AutoTokenizer.from_pretrained(args.en_ru_tokenizer, local_files_only=True),
        de_en=ctranslate2.Translator(args.de_en_model, device="cpu"),
        en_ru=ctranslate2.Translator(args.en_ru_model, device="cpu"),
    )


def translate_batch(translator: ctranslate2.Translator, tokenizer: Any, texts: Sequence[str], batch_size: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for start in range(0, len(texts), max(1, batch_size)):
        batch = list(texts[start : start + batch_size])
        tokenized = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=True))
            for text in batch
        ]
        results = translator.translate_batch(tokenized, beam_size=1)
        for text, result in zip(batch, results):
            ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0])
            out[text] = normalize_text(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def protect_catalog_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    protected: Dict[str, str] = {}
    out = text
    for i, match in enumerate(dict.fromkeys(WORD_RE.findall(text))):
        if not BRAND_TOKEN_RE.fullmatch(match):
            continue
        if not UPPER_LETTER_RE.search(match):
            continue
        placeholder = f"AAA{i}"
        protected[placeholder] = match
        out = re.sub(rf"(?<!\w){re.escape(match)}(?!\w)", placeholder, out)
    return out, protected


def restore_catalog_tokens(text: str, protected: Dict[str, str]) -> str:
    out = text
    for placeholder, original in protected.items():
        latin_or_cyrillic_a = "".join("[AА]" if ch == "A" else re.escape(ch) for ch in placeholder)
        out = re.sub(rf"(?<!\w){latin_or_cyrillic_a}(?!\w)", original, out)
    return out


def translate_missing(
    missing: Sequence[str],
    cache: Dict[str, Dict[str, str]],
    cache_path: Path,
    args: argparse.Namespace,
) -> None:
    dict_hits = 0
    model_inputs: List[str] = []
    for text in missing:
        known = term_translation(text)
        if known:
            cache[text] = known
            dict_hits += 1
        else:
            model_inputs.append(text)

    print(f"[terms] dict_hits={dict_hits} model_inputs={len(model_inputs)}", flush=True)
    if not model_inputs:
        return

    t0 = time.perf_counter()
    pair = load_ct2(args)
    chunk_size = max(args.batch_size, args.cache_save_every)
    done = 0
    for start in range(0, len(model_inputs), chunk_size):
        chunk = model_inputs[start : start + chunk_size]
        protected_by_source: Dict[str, Dict[str, str]] = {}
        protected_inputs: List[str] = []
        protected_to_source: Dict[str, str] = {}
        for source in chunk:
            protected, mapping = protect_catalog_tokens(source)
            protected_by_source[source] = mapping
            protected_inputs.append(protected)
            protected_to_source[protected] = source

        de_en_protected = translate_batch(pair.de_en, pair.de_en_tokenizer, protected_inputs, args.batch_size)
        de_en: Dict[str, str] = {}
        for protected, en_text in de_en_protected.items():
            source = protected_to_source[protected]
            de_en[source] = restore_catalog_tokens(en_text, protected_by_source[source])

        en_inputs = list(dict.fromkeys(de_en.values()))
        en_protected_inputs: List[str] = []
        en_protected_to_source: Dict[str, str] = {}
        en_to_source = {value: source for source, value in de_en.items()}
        for en_text in en_inputs:
            source = en_to_source[en_text]
            protected, _ = protect_catalog_tokens(en_text)
            en_protected_inputs.append(protected)
            en_protected_to_source[protected] = source

        en_ru_protected = translate_batch(pair.en_ru, pair.en_ru_tokenizer, en_protected_inputs, args.batch_size)
        en_ru: Dict[str, str] = {}
        for protected, ru_text in en_ru_protected.items():
            source = en_protected_to_source[protected]
            en_text = de_en[source]
            en_ru[en_text] = restore_catalog_tokens(ru_text, protected_by_source[source])

        for source in chunk:
            en = de_en.get(source, source)
            ru = en_ru.get(en, en)
            cache[source] = {"en": en, "ru": ru}

        done += len(chunk)
        write_json(cache_path, cache)
        print(
            f"[ct2] translated={done}/{len(model_inputs)} cache_entries={len(cache)} "
            f"seconds={time.perf_counter() - t0:.2f}",
            flush=True,
        )


def compact_join(values: Iterable[str]) -> str:
    seen: Dict[str, None] = {}
    for value in values:
        for part in re.split(r"[,;]+", normalize_text(value)):
            if part:
                seen.setdefault(part, None)
    return " ".join(seen.keys())


def apply_item(item: Dict[str, Any], cache: Dict[str, Dict[str, str]], *, overwrite: bool, max_chars: int) -> int:
    changed = 0
    for field in SHORT_FIELDS:
        source = normalize_text(item.get(field))
        tr = cache.get(source)
        if not source or not tr:
            continue
        for lang in ("en", "ru"):
            key = f"{field}_{lang}"
            if overwrite or not normalize_text(item.get(key)):
                item[key] = tr[lang]
                changed += 1

    description_short_de = first_sentence(normalize_text(item.get("description")))
    if description_short_de:
        tr = cache.get(description_short_de)
        if overwrite or not normalize_text(item.get("description_short_de")):
            item["description_short_de"] = description_short_de
            changed += 1
        if tr:
            for lang in ("en", "ru"):
                key = f"description_short_{lang}"
                if overwrite or not normalize_text(item.get(key)):
                    item[key] = tr[lang]
                    changed += 1

    search_de_parts = [
        normalize_text(item.get("title")),
        normalize_text(item.get("category_raw")),
        normalize_text(item.get("color")),
        normalize_text(item.get("materials")),
        normalize_text(item.get("brand")),
        normalize_text(item.get("collection")),
    ]
    search_en_parts = [
        normalize_text(item.get("title_en")),
        normalize_text(item.get("category_raw_en")),
        normalize_text(item.get("color_en")),
        normalize_text(item.get("materials_en")),
        normalize_text(item.get("brand")),
        normalize_text(item.get("collection")),
    ]
    search_ru_parts = [
        normalize_text(item.get("title_ru")),
        normalize_text(item.get("category_raw_ru")),
        normalize_text(item.get("color_ru")),
        normalize_text(item.get("materials_ru")),
        normalize_text(item.get("brand")),
        normalize_text(item.get("collection")),
    ]
    for key, value in (
        ("search_text_de", compact_join(search_de_parts)),
        ("search_text_en", compact_join(search_en_parts)),
        ("search_text_ru", compact_join(search_ru_parts)),
    ):
        if value and (overwrite or not normalize_text(item.get(key))):
            item[key] = value
            changed += 1

    return changed


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache)

    data = read_json(input_path)
    items = get_items(data)
    selected = [item for item in items if matches_source(item, args.source_site_contains)]
    if args.limit is not None:
        selected = selected[: args.limit]

    phrases: List[str] = []
    for item in selected:
        phrases.extend(item_phrases(item, args.max_chars))
    unique_phrases = list(dict.fromkeys(phrases))
    cache = load_cache(cache_path)
    refreshed = refresh_known_term_cache(unique_phrases, cache)
    missing = [phrase for phrase in unique_phrases if phrase not in cache]

    print(f"[input] {input_path}", flush=True)
    print(f"[items] total={len(items)} selected={len(selected)} source_site_contains={args.source_site_contains!r}", flush=True)
    print(
        f"[collect] phrases={len(phrases)} unique={len(unique_phrases)} "
        f"cache={len(cache)} refreshed_terms={refreshed} missing={len(missing)}",
        flush=True,
    )

    if args.dry_run:
        print("[dry-run] output was not written", flush=True)
        return

    translate_missing(missing, cache, cache_path, args)
    write_json(cache_path, cache)

    changed = 0
    for item in selected:
        changed += apply_item(item, cache, overwrite=args.overwrite_existing, max_chars=args.max_chars)

    meta = data.setdefault("meta", {}) if isinstance(data, dict) else {}
    if isinstance(meta, dict):
        meta["de_en_ru_catalog_enrichment"] = {
            "source": str(input_path),
            "created_at_unix": time.time(),
            "source_language": "de",
            "selected_items": len(selected),
            "unique_phrases": len(unique_phrases),
            "cache_path": str(cache_path),
            "fields": [
                "title_en",
                "title_ru",
                "category_raw_en",
                "category_raw_ru",
                "color_en",
                "color_ru",
                "materials_en",
                "materials_ru",
                "description_short_de",
                "description_short_en",
                "description_short_ru",
                "search_text_de",
                "search_text_en",
                "search_text_ru",
            ],
        }

    write_json(output_path, data)
    print(f"[done] changed_fields={changed}", flush=True)
    print(f"[done] output={output_path}", flush=True)
    print(f"[done] cache={cache_path}", flush=True)


if __name__ == "__main__":
    main()
