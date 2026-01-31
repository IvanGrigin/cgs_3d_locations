#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

ARCHIVE_EXT = (".rar",)  # если нужно: (".rar", ".zip")


def is_archive_url(u: str) -> bool:
    u2 = u.split("?", 1)[0].lower()
    return any(u2.endswith(ext) for ext in ARCHIVE_EXT)


def to_absolute(u: str, base: str) -> str:
    u = u.strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return urljoin(base, u)
    # уже абсолютная
    return u


def url_to_local_path(url: str, out_root: Path) -> Path:
    """
    https://imodern.ru/upload/iblock/d8d/....rar
      -> out_root / upload/iblock/d8d/....rar
    """
    p = urlparse(url)
    if not p.path or p.path == "/":
        raise ValueError(f"Плохой путь в URL: {url}")
    rel = p.path.lstrip("/")
    return out_root / rel


def download_one(session: requests.Session, url: str, dst: Path, retries: int, timeout: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            # HEAD чтобы понять размер (не всегда поддерживается, но часто)
            expected = None
            try:
                h = session.head(url, allow_redirects=True, timeout=timeout)
                if h.ok and "Content-Length" in h.headers:
                    expected = int(h.headers["Content-Length"])
            except Exception:
                pass

            if dst.exists() and expected is not None and dst.stat().st_size == expected and expected > 0:
                return True  # уже скачано

            # stream download
            with session.get(url, stream=True, allow_redirects=True, timeout=timeout) as r:
                r.raise_for_status()

                # если файл есть — попробуем докачку
                resume_from = dst.stat().st_size if dst.exists() else 0
                # если сервер поддерживает Range — запросим с позиции
                if resume_from > 0:
                    # повторим запрос с Range
                    headers = {"Range": f"bytes={resume_from}-"}
                    r.close()
                    with session.get(url, stream=True, allow_redirects=True, timeout=timeout, headers=headers) as rr:
                        if rr.status_code in (206, 200):
                            mode = "ab" if rr.status_code == 206 else "wb"
                            if rr.status_code == 200:
                                resume_from = 0
                            with open(dst, mode) as f:
                                for chunk in rr.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        f.write(chunk)
                        else:
                            rr.raise_for_status()
                else:
                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

            # финальная проверка размера
            if expected is not None and dst.exists() and dst.stat().st_size != expected:
                # возможно, докачка не сработала/сервер не дал размер — попробуем ещё раз
                raise IOError(f"Размер не совпал: {dst.stat().st_size} != {expected}")

            return True

        except Exception as e:
            if attempt == retries:
                print(f"[FAIL] {url} -> {dst} | {e}", file=sys.stderr)
                return False
            time.sleep(1.5 * attempt)

    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Скачивание архивов imodern по списку ссылок.")
    ap.add_argument("input", type=Path, help="Файл со ссылками (по одной в строке), например links_urls.txt")
    ap.add_argument("--base", default="https://imodern.ru", help="Базовый домен для относительных ссылок")
    ap.add_argument("--out", type=Path, default=Path("src/data/sourse/imodern"), help="Папка для скачивания")
    ap.add_argument("--retries", type=int, default=3, help="Кол-во попыток на файл")
    ap.add_argument("--timeout", type=int, default=60, help="Таймаут (сек)")
    args = ap.parse_args()

    inp: Path = args.input
    out_root: Path = args.out
    base: str = args.base.rstrip("/") + "/"

    if not inp.exists():
        raise SystemExit(f"Нет входного файла: {inp}")

    lines = inp.read_text(encoding="utf-8", errors="ignore").splitlines()
    urls = []
    for line in lines:
        u = to_absolute(line, base)
        if u and is_archive_url(u):
            urls.append(u)

    # дедупликация с сохранением порядка
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    print(f"Архивных ссылок (.rar): {len(uniq)}")
    out_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) downloader/1.0"
    })

    ok = 0
    fail = 0
    for i, url in enumerate(uniq, 1):
        try:
            dst = url_to_local_path(url, out_root)
        except Exception as e:
            print(f"[SKIP] {url} | {e}", file=sys.stderr)
            fail += 1
            continue

        if dst.exists():
            # быстрый пропуск: если файл уже есть и не пустой
            if dst.stat().st_size > 0:
                print(f"[{i}/{len(uniq)}] SKIP exists: {dst}")
                ok += 1
                continue

        print(f"[{i}/{len(uniq)}] GET {url}")
        if download_one(session, url, dst, retries=args.retries, timeout=args.timeout):
            ok += 1
        else:
            fail += 1

    print(f"Готово. Успешно: {ok}, ошибок: {fail}")
    print(f"Скачано в: {out_root.resolve()}")


if __name__ == "__main__":
    main()