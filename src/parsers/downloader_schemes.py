import os
import re
import time
import json
import hashlib
import urllib.parse as urlparse
from collections import deque, defaultdict
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from urllib import robotparser

# ---------- Настройки по умолчанию ----------
KEYWORDS = [
    # русские
    "планировка", "планировки", "план квартиры", "поэтажный план",
    "план дома", "план секции", "типовой план", "чертеж", "чертёж",
    "pdf план", "поэтажный", "квартирография", "брошюра", "каталог",
    # английские
    "floor plan", "floorplan", "layout", "apartment plan", "unit plan", "brochure"
]
FILE_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
USER_AGENT = "FloorplanCrawler/1.0 (+https://example.local)"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 1.0  # сек
MAX_BYTES = 80 * 1024 * 1024   # 80MB защитный лимит
MIN_IMAGE_SIDE = 500           # отсечь иконки (по факту после скачивания)

# ---------- Вспомогательные ----------
def norm_domain(url: str) -> str:
    try:
        return urlparse.urlparse(url).netloc.lower()
    except Exception:
        return ""

def same_domain(a: str, b: str) -> bool:
    return norm_domain(a) == norm_domain(b)

def is_allowed_by_robots(url: str, rp_cache: dict) -> bool:
    base = f"{urlparse.urlparse(url).scheme}://{urlparse.urlparse(url).netloc}"
    if base not in rp_cache:
        robots_url = urlparse.urljoin(base, "/robots.txt")
        rp = robotparser.RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            # Если robots.txt не читается, действуем консервативно и позволяем,
            # но можно поменять на return False для строгого режима
            rp_cache[base] = None
            return True
        rp_cache[base] = rp
    rp = rp_cache[base]
    if rp is None:
        return True
    return rp.can_fetch(USER_AGENT, url)

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:120] if name else "file"

def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def pick_best_from_srcset(srcset_value: str) -> str:
    # Берём ссылку с максимальной шириной
    best = None
    max_w = -1
    for part in srcset_value.split(","):
        url_part = part.strip().split()
        if not url_part:
            continue
        u = url_part[0]
        w = -1
        if len(url_part) >= 2 and url_part[1].endswith("w"):
            try:
                w = int(url_part[1][:-1])
            except ValueError:
                w = -1
        if w > max_w:
            max_w = w
            best = u
    return best or srcset_value.split(",")[0].strip()

def looks_like_floorplan_link(a_text: str, href: str) -> bool:
    text = (a_text or "").lower()
    href_l = (href or "").lower()
    if any(k in text for k in KEYWORDS):
        return True
    if any(k in href_l for k in KEYWORDS):
        return True
    if href_l.endswith(".pdf"):
        # pdf часто с планами/буклетами
        return True
    return False

def absolutize(base_url: str, maybe_url: str) -> str:
    return urlparse.urljoin(base_url, maybe_url)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_bytes(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

def try_open_image(binary: bytes) -> Image.Image | None:
    try:
        img = Image.open(BytesIO(binary))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        return None

# ---------- Извлечение ассетов со страницы ----------
def extract_assets(page_url: str, soup: BeautifulSoup) -> list[dict]:
    found = []

    # 1) Явные ссылки <a> на pdf/jpg/png + эвристика по тексту
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        text = a.get_text(separator=" ", strip=True)
        if not href:
            continue
        if looks_like_floorplan_link(text, href) or href.lower().endswith(FILE_EXTS):
            found.append({
                "type": "file",
                "href": href,
                "text": text,
                "context": a.parent.get_text(" ", strip=True)[:500] if a.parent else ""
            })

    # 2) Картинки <img> — берём src/srcset/data-src и фильтруем по alt/классам/названию
    for img in soup.find_all("img"):
        src = img.get("src")
        alt = (img.get("alt") or "").lower()
        data_src = img.get("data-src") or img.get("data-original") or img.get("data-lazy")
        srcset = img.get("srcset")
        candidate = None
        if srcset:
            candidate = pick_best_from_srcset(srcset)
        elif data_src:
            candidate = data_src
        else:
            candidate = src

        if not candidate:
            continue

        hint = " ".join([
            alt,
            " ".join(img.get("class") or []),
            img.get("id") or ""
        ]).lower()

        if any(k in hint for k in ["plan", "layout", "план", "квартир", "этаж", "секц"]):
            found.append({
                "type": "image",
                "href": candidate,
                "text": alt,
                "context": img.parent.get_text(" ", strip=True)[:500] if img.parent else ""
            })
        elif candidate.lower().endswith(FILE_EXTS):
            # возможно, это всё равно полезная картинка/пдф
            found.append({
                "type": "image",
                "href": candidate,
                "text": alt,
                "context": img.parent.get_text(" ", strip=True)[:500] if img.parent else ""
            })

    # 3) Вставленные iframe и object (часто любят встраивать pdf)
    for tag in soup.find_all(["iframe", "object", "embed"]):
        src = tag.get("src") or tag.get("data")
        if src and (src.lower().endswith(".pdf") or "pdf" in src.lower()):
            found.append({
                "type": "file",
                "href": src,
                "text": "embedded pdf",
                "context": tag.parent.get_text(" ", strip=True)[:500] if tag.parent else ""
            })

    # Удаляем явные дубликаты по href
    uniq = {}
    for x in found:
        href_abs = absolutize(page_url, x["href"])
        uniq[href_abs] = {
            **x,
            "href": href_abs
        }
    return list(uniq.values())

# ---------- Краулер ----------
def crawl(start_url: str, out_dir: str, max_pages: int = 300):
    session = build_session()
    rp_cache = {}
    visited = set()
    q = deque([start_url])
    base_domain = norm_domain(start_url)
    ensure_dir(out_dir)

    # Индекс по хешам для дедупликации
    seen_hashes = set()
    hash_index_path = os.path.join(out_dir, "_hashes.json")
    if os.path.exists(hash_index_path):
        try:
            seen_hashes = set(json.load(open(hash_index_path)))
        except Exception:
            seen_hashes = set()

    page_count = 0
    per_domain_last = defaultdict(lambda: 0.0)

    with tqdm(total=max_pages, desc=f"Crawling {base_domain}") as pbar:
        while q and page_count < max_pages:
            url = q.popleft()
            if url in visited:
                continue
            visited.add(url)

            if not same_domain(start_url, url):
                continue

            if not is_allowed_by_robots(url, rp_cache):
                print(f"[robots] SKIP {url}")
                continue

            # rate limit для домена
            dom = norm_domain(url)
            dt = time.time() - per_domain_last[dom]
            if dt < SLEEP_BETWEEN_REQUESTS:
                time.sleep(SLEEP_BETWEEN_REQUESTS - dt)

            try:
                r = session.get(url, timeout=REQUEST_TIMEOUT)
                per_domain_last[dom] = time.time()
            except requests.RequestException:
                continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype:
                # Не HTML — возможно, файл. Скачаем, если это планировка.
                if url.lower().endswith(FILE_EXTS):
                    try_download_asset(url, r.content, r.headers, out_dir, seen_hashes, page_url=url, page_title="")
                    json.dump(list(seen_hashes), open(hash_index_path, "w"))
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            title = (soup.title.string.strip() if soup.title and soup.title.string else "")

            # 1) Извлечь ассеты на странице
            assets = extract_assets(url, soup)
            saved_items = []
            for a in assets:
                href = a["href"]
                try:
                    ar = session.get(href, timeout=REQUEST_TIMEOUT, stream=True)
                except requests.RequestException:
                    continue

                # ограничение размера
                total = 0
                chunks = []
                for chunk in ar.iter_content(chunk_size=16384):
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_BYTES:
                        chunks = []
                        break
                if not chunks:
                    continue

                binary = b"".join(chunks)
                saved_as = try_download_asset(href, binary, ar.headers, out_dir, seen_hashes, page_url=url, page_title=title, context=a.get("context", ""), anchor_text=a.get("text", ""))
                if saved_as:
                    saved_items.append(saved_as)
                    json.dump(list(seen_hashes), open(hash_index_path, "w"))

            # 2) Добавить новые ссылки в очередь (узкое BFS)
            for link in soup.find_all("a", href=True):
                nxt = absolutize(url, link["href"])
                # держим только тот же домен, убираем якоря/почту/телефон
                if nxt.startswith("mailto:") or nxt.startswith("tel:"):
                    continue
                if "#" in nxt:
                    nxt = nxt.split("#", 1)[0]
                if same_domain(start_url, nxt) and nxt not in visited:
                    # слегка приоритизируем страницы «планировок»
                    txt = (link.get_text(" ", strip=True) or "").lower()
                    if any(k in (txt + " " + nxt.lower()) for k in ["plan", "layout", "план", "квартир", "этаж", "pdf"]):
                        q.appendleft(nxt)
                    else:
                        q.append(nxt)

            # 3) Записать meta по странице, если что-то сохранили
            if saved_items:
                page_dir = os.path.join(out_dir, "_pages")
                ensure_dir(page_dir)
                page_meta = {
                    "url": url,
                    "title": title,
                    "saved": saved_items
                }
                meta_name = sanitize_filename(title) or sha1(url.encode())
                with open(os.path.join(page_dir, f"{meta_name}.json"), "w", encoding="utf-8") as f:
                    json.dump(page_meta, f, ensure_ascii=False, indent=2)

            page_count += 1
            pbar.update(1)

def guess_extension(content_type: str, url: str) -> str:
    url = url.lower()
    if url.endswith(".pdf"):
        return ".pdf"
    if url.endswith(".png"):
        return ".png"
    if url.endswith(".jpg") or url.endswith(".jpeg"):
        return ".jpg"
    if url.endswith(".webp"):
        return ".webp"
    if url.endswith(".tif") or url.endswith(".tiff"):
        return ".tif"

    ct = (content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "webp" in ct:
        return ".webp"
    if "tiff" in ct or "tif" in ct:
        return ".tif"
    return ".bin"

def try_download_asset(asset_url: str, binary: bytes, headers: dict, out_dir: str, seen_hashes: set,
                       page_url: str, page_title: str, context: str = "", anchor_text: str = "") -> dict | None:
    h = sha1(binary)
    if h in seen_hashes:
        return None

    ext = guess_extension(headers.get("Content-Type", ""), asset_url)
    # Мини-фильтр: если картинка слишком маленькая — пропускаем
    if ext in [".png", ".jpg", ".webp", ".tif", ".jpeg"]:
        img = try_open_image(binary)
        if img is None:
            # бывает, что «картинка» — SVG или битая; пропускаем
            return None
        w, h_img = img.size
        if w < MIN_IMAGE_SIDE or h_img < MIN_IMAGE_SIDE:
            return None

    # Структура: out/<domain>/<YYYY>/<mm> можно простую: out/<domain>/
    domain = norm_domain(asset_url) or norm_domain(page_url)
    asset_dir = os.path.join(out_dir, domain)
    ensure_dir(asset_dir)

    # Имя — по заголовку страницы + хеш, чтобы уникально
    base = sanitize_filename(page_title) or "plan"
    fname = f"{base}__{h[:10]}{ext}"
    fpath = os.path.join(asset_dir, fname)
    save_bytes(fpath, binary)

    # meta по файлу
    meta = {
        "saved_as": fpath,
        "source_url": asset_url,
        "page_url": page_url,
        "page_title": page_title,
        "sha1": h,
        "content_type": headers.get("Content-Type", ""),
        "context_snippet": context,
        "anchor_text": anchor_text
    }
    with open(f"{fpath}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    seen_hashes.add(meta["sha1"])
    return meta

# ---------- Точка входа (input) ----------
if __name__ == "__main__":
    print("Введите стартовые URL (через запятую), например:")
    print("https://developer.example/realty/complex-1/floorplans, https://developer.example/complex-2/")
    start_urls_raw = input("start_urls: ").strip()
    start_urls = [u.strip() for u in start_urls_raw.split(",") if u.strip()]

    out_dir = input("Папка для сохранения (по умолчанию output_floorplans): ").strip() or "output_floorplans"
    try:
        max_pages = int(input("Макс. страниц на домен (например 300): ").strip() or "300")
    except ValueError:
        max_pages = 300

    for u in start_urls:
        crawl(u, out_dir, max_pages=max_pages)

    print(f"Готово. Проверьте папку: {out_dir}")