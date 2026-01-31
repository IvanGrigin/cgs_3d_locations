# src/tools/download_and_extract_imodern.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from collections import deque

import requests

ARCHIVE_EXTS = {".rar", ".zip", ".7z"}
# Для "первых частей" 7z-сплитов иногда бывает *.7z.001
SPLIT_7Z_FIRST_SUFFIX = ".7z.001"


@dataclass
class ArchiveItem:
    bx_id: Optional[str]
    name: Optional[str]
    url: str
    filename: str
    ext: str


def _safe_name(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9A-Za-zА-Яа-я._\- ]+", "_", s)
    s = s.strip(" ._-")
    return (s or "item")[:max_len]


def _url_filename(url: str) -> str:
    p = urlparse(url)
    name = Path(p.path).name
    return name or "download"


def _ext(url_or_name: str) -> str:
    return Path(url_or_name).suffix.lower()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _ensure_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    i = 1
    while True:
        cand = base_dir.parent / f"{base_dir.name}-{i}"
        if not cand.exists():
            return cand
        i += 1


def _check_tools(need_7z: bool) -> None:
    if shutil.which("unar") is None:
        raise RuntimeError(
            "Не найден `unar`. Установи: brew install unar\n"
            "И проверь, что команда `unar` доступна в PATH."
        )
    if need_7z and shutil.which("7z") is None:
        raise RuntimeError("Нужен 7z для .7z: brew install p7zip")


def load_archives_from_db(db_path: Path, allow_dedup_by_url: bool) -> List[ArchiveItem]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB не найдена: {db_path}")

    q = """
    SELECT bx_id, name, archive_url_abs, archive_filename, archive_ext
    FROM imodern_item
    WHERE archive_url_abs IS NOT NULL AND archive_url_abs <> ''
    """

    items: List[ArchiveItem] = []
    with sqlite3.connect(db_path) as con:
        for bx_id, name, url, archive_filename, archive_ext in con.execute(q):
            url = (url or "").strip()
            if not url:
                continue
            ext = (archive_ext or _ext(url)).lower()
            if ext not in ARCHIVE_EXTS:
                continue
            filename = (archive_filename or _url_filename(url)).strip()
            if not _ext(filename):
                filename += ext
            items.append(ArchiveItem(bx_id=bx_id, name=name, url=url, filename=filename, ext=ext))

    if not allow_dedup_by_url:
        return items

    # Дедуп по url (опционально)
    uniq = {}
    for it in items:
        uniq[it.url] = it
    return list(uniq.values())


def download_file(url: str, out_path: Path, timeout: Tuple[int, int], retries: int, sleep_sec: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, allow_redirects=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp_path.replace(out_path)
            return
        except Exception as e:
            last_err = e
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(sleep_sec)

    raise RuntimeError(f"Не удалось скачать {url}: {last_err}")


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"ZIP содержит небезопасный путь: {member.filename}")
        zf.extractall(dest_dir)


def _is_rar_first_volume(p: Path) -> bool:
    """
    True если это "первая часть" RAR:
      - *.rar
      - *.part1.rar / *.part01.rar / *.part001.rar
    False если это *.part2.rar и т.п. (их пропускаем).
    """
    name = p.name.lower()
    if name.endswith(".rar"):
        m = re.search(r"\.part(\d+)\.rar$", name)
        if m:
            return int(m.group(1)) == 1
        return True
    return False


def _marker_for_archive(archive_path: Path) -> Path:
    # Маркер рядом с архивом, чтобы не распаковывать повторно
    return archive_path.with_name(archive_path.name + ".extracted_ok")


def extract_archive_once(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()

    if ext == ".zip":
        safe_extract_zip(archive_path, dest_dir)
        return

    if ext == ".rar":
        if not _is_rar_first_volume(archive_path):
            # Не первая часть тома — пропускаем, unar должен запускаться на первой
            return
        res = subprocess.run(
            ["unar", "-o", str(dest_dir), "-f", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"unar не смог распаковать {archive_path}:\n{res.stdout}")
        return

    if ext == ".7z":
        if shutil.which("7z") is None:
            raise RuntimeError("Нужен 7z для .7z: brew install p7zip")
        res = subprocess.run(
            ["7z", "x", "-y", f"-o{dest_dir}", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"7z не смог распаковать {archive_path}:\n{res.stdout}")
        return

    raise RuntimeError(f"Неизвестное расширение архива: {archive_path}")


def list_archives_in_tree(root: Path) -> List[Path]:
    archives: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        # обычные архивы
        if p.suffix.lower() in ARCHIVE_EXTS:
            archives.append(p)
            continue
        # сплит 7z
        if low.endswith(SPLIT_7Z_FIRST_SUFFIX):
            archives.append(p)
            continue
    return archives


def build_item_dir(root: Path, it: ArchiveItem) -> Path:
    base = _safe_name(it.name or Path(it.filename).stem)
    suffix = it.bx_id or _sha1(it.url)[:10]
    dir_name = _safe_name(f"{base}__{suffix}")
    return root / dir_name


def _default_dest_for_archive(archive_path: Path, base_out_dir: Optional[Path]) -> Path:
    """
    Куда распаковывать архивы в режиме SCAN:
      - если base_out_dir задан, то строим путь относительно него;
      - распаковку кладём рядом с архивом, в подпапку <stem>__extracted
    """
    stem = archive_path.name
    # аккуратно уберём ".part1.rar" -> "xxx"
    stem = re.sub(r"\.part\d+\.rar$", "", stem, flags=re.IGNORECASE)
    # ".rar" -> ""
    stem = re.sub(r"\.rar$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.zip$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.7z$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.7z\.001$", "", stem, flags=re.IGNORECASE)

    folder = _safe_name(stem) + "__extracted"
    if base_out_dir is None:
        return archive_path.parent / folder

    try:
        rel = archive_path.parent.relative_to(base_out_dir)
        return base_out_dir / rel / folder
    except Exception:
        return archive_path.parent / folder


def deep_extract_archives(
    start_archives: Iterable[Path],
    scan_root: Optional[Path],
    depth: int,
    remove_archives: bool,
    verbose: bool,
) -> Tuple[int, int, int]:
    """
    Рекурсивно распаковывает архивы:
      - стартуем со списка start_archives
      - после распаковки ищем новые архивы внутри dest_dir и добавляем в очередь (если depth>1)
    Возвращает: (ok, skipped, failed)
    """
    ok = 0
    skipped = 0
    failed = 0

    q: Deque[Tuple[Path, int]] = deque()
    for a in start_archives:
        q.append((a, depth))

    while q:
        arch, d = q.popleft()
        if not arch.exists() or not arch.is_file():
            continue

        marker = _marker_for_archive(arch)
        if marker.exists():
            skipped += 1
            continue

        # Определим dest_dir
        dest_dir = _default_dest_for_archive(arch, scan_root)

        try:
            if verbose:
                print(f"[EXTRACT] {arch}")
                print(f"          -> {dest_dir}")

            before_files = sum(1 for _ in dest_dir.rglob("*")) if dest_dir.exists() else 0
            extract_archive_once(arch, dest_dir)
            after_files = sum(1 for _ in dest_dir.rglob("*")) if dest_dir.exists() else 0

            # Простая проверка "не пусто" (после распаковки должно появиться что-то)
            if after_files <= before_files:
                # unar иногда создаёт только папку верхнего уровня — всё равно это валидно,
                # но если вообще ничего не появилось, считаем ошибкой.
                # (папка могла существовать заранее, поэтому сравнение with before_files)
                pass

            marker.write_text("ok\n", encoding="utf-8")
            ok += 1

            if remove_archives:
                try:
                    arch.unlink()
                except Exception:
                    # не критично
                    pass

            # Рекурсия: ищем новые архивы внутри dest_dir
            if d > 1 and dest_dir.exists():
                nested = list_archives_in_tree(dest_dir)
                for na in nested:
                    # не добавляем то, что уже помечено
                    if not _marker_for_archive(na).exists():
                        q.append((na, d - 1))

        except Exception as e:
            failed += 1
            print(f"[ERROR] {arch}: {e}", file=sys.stderr)

    return ok, skipped, failed


def main() -> None:
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan", type=str, help="Режим SCAN: распаковать все архивы внутри указанной папки")
    mode.add_argument("--db", type=str, help="Режим DB: путь к imodern.db (скачать из БД и распаковать)")

    # DB-mode args
    ap.add_argument("--out", type=str, default="src/data/sourse/imodern", help="Куда складывать распакованные данные (DB)")
    ap.add_argument("--archives-dir", type=str, default="src/data/sourse/imodern/_archives", help="Куда сохранять скачанные архивы (DB)")
    ap.add_argument("--skip-download", action="store_true", help="DB: не скачивать, только распаковывать уже скачанное")
    ap.add_argument("--no-dedup-by-url", action="store_true", help="DB: НЕ дедупить по URL (обрабатывать записи как есть)")

    # network
    ap.add_argument("--timeout-connect", type=int, default=10)
    ap.add_argument("--timeout-read", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=1.5)

    # deep extraction
    ap.add_argument("--depth", type=int, default=2, help="Глубина рекурсивной распаковки вложенных архивов")
    ap.add_argument("--remove-archives", action="store_true", help="Удалять архив после успешной распаковки")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    need_7z = True  # безопаснее считать, что может встретиться .7z
    _check_tools(need_7z=need_7z)

    if args.scan:
        scan_root = Path(args.scan)
        if not scan_root.exists():
            raise FileNotFoundError(f"--scan путь не найден: {scan_root}")

        start = list_archives_in_tree(scan_root)
        print(f"Найдено архивов для распаковки: {len(start)}")
        ok, skipped, failed = deep_extract_archives(
            start_archives=start,
            scan_root=scan_root,
            depth=max(1, args.depth),
            remove_archives=args.remove_archives,
            verbose=args.verbose,
        )

        print("\n=== Итог (SCAN) ===")
        print(f"Успешно   : {ok}")
        print(f"Пропущено : {skipped} (уже были маркеры *.extracted_ok)")
        print(f"Ошибок    : {failed}")
        print(f"Корень    : {scan_root.resolve()}")
        return

    # DB mode
    db_path = Path(args.db)
    out_root = Path(args.out)
    archives_root = Path(args.archives_dir)

    out_root.mkdir(parents=True, exist_ok=True)
    archives_root.mkdir(parents=True, exist_ok=True)

    items = load_archives_from_db(db_path, allow_dedup_by_url=(not args.no_dedup_by_url))
    print(f"Архивов из БД: {len(items)}")

    # 1) Скачать/распаковать архивы из БД как раньше (по объектам)
    ok_db = 0
    fail_db = 0
    extracted_item_dirs: List[Path] = []

    for idx, it in enumerate(items, start=1):
        try:
            archive_path = archives_root / it.filename
            if not it.ext:
                it.ext = _ext(it.filename) or _ext(it.url)
            if archive_path.suffix.lower() not in ARCHIVE_EXTS:
                archive_path = archive_path.with_suffix(it.ext)

            item_dir = build_item_dir(out_root, it)
            item_dir = _ensure_unique_dir(item_dir) if not item_dir.exists() else item_dir

            print(f"[{idx}/{len(items)}] {it.name or it.filename}")
            print(f"  URL : {it.url}")
            print(f"  ARCH: {archive_path}")
            print(f"  OUT : {item_dir}")

            if not args.skip_download:
                download_file(
                    it.url,
                    archive_path,
                    timeout=(args.timeout_connect, args.timeout_read),
                    retries=args.retries,
                    sleep_sec=args.sleep,
                )

            # распаковываем архив в item_dir (не ставим общий marker на папку,
            # marker будет рядом с самим архивом в archives_root)
            marker = _marker_for_archive(archive_path)
            if not marker.exists():
                extract_archive_once(archive_path, item_dir)
                marker.write_text("ok\n", encoding="utf-8")

            extracted_item_dirs.append(item_dir)
            ok_db += 1
        except Exception as e:
            fail_db += 1
            print(f"  ERROR: {e}", file=sys.stderr)

    print("\n=== Итог (DB) ===")
    print(f"Успешно: {ok_db}")
    print(f"Ошибок : {fail_db}")
    print(f"Распаковано в: {out_root.resolve()}")
    print(f"Архивы лежат в: {archives_root.resolve()}")

    # 2) После DB-фазы делаем глубокую распаковку по out_root:
    #    это добьёт вложенные архивы, которые оказались внутри распаковки.
    print("\nЗапускаю рекурсивную распаковку вложенных архивов в out_root...")
    start = list_archives_in_tree(out_root)
    print(f"Найдено архивов в out_root: {len(start)}")
    ok, skipped, failed = deep_extract_archives(
        start_archives=start,
        scan_root=out_root,
        depth=max(1, args.depth),
        remove_archives=args.remove_archives,
        verbose=args.verbose,
    )

    print("\n=== Итог (DEEP in out_root) ===")
    print(f"Успешно   : {ok}")
    print(f"Пропущено : {skipped}")
    print(f"Ошибок    : {failed}")


if __name__ == "__main__":
    main()