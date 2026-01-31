from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

URL_ATTRS = {"href", "src", "data-src"}
ARCHIVE_EXT = (".rar", ".zip", ".7z")


class UrlExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in URL_ATTRS and value:
                v = value.strip()
                if v:
                    self.urls.append(v)


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default="links.html", help="путь к HTML (по умолчанию links.html)")
    p.add_argument("--out", default="", help="путь к выходному файлу (по умолчанию рядом links_urls.txt)")
    p.add_argument("--base", default="", help="база для абсолютных ссылок, например https://site.ru")
    p.add_argument("--archives-only", action="store_true", help="оставить только .rar/.zip/.7z")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Файл не найден: {in_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else in_path.with_name("links_urls.txt")

    html = in_path.read_text(encoding="utf-8", errors="replace")

    parser = UrlExtractor()
    parser.feed(html)

    urls = unique_preserve_order(parser.urls)

    if args.archives_only:
        urls = [u for u in urls if u.lower().split("?")[0].endswith(ARCHIVE_EXT)]

    if args.base:
        base = args.base.rstrip("/") + "/"
        urls = [urljoin(base, u) for u in urls]

    out_path.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")

    print(f"Найдено ссылок: {len(urls)}")
    print(f"Записано в: {out_path}")


if __name__ == "__main__":
    main()