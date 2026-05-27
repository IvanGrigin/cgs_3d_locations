# -*- coding: utf-8 -*-
"""
This module fetches supplier product pages and resolves downloadable model files.
It handles direct downloads, Yandex Disk links, and authenticated 3ddd flows.
The code also records download metadata back into supplier databases.
This is the main network entrypoint for asset acquisition.
Keep auth handling and download notes explicit and traceable.
"""
from __future__ import annotations

import argparse
import base64
import http.cookiejar
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse

import requests

from src.suppliers.adapters.base import SupplierAdapter
from src.suppliers.db_core import init_db, insert_download, upsert_product
from src.suppliers.models import ProductRecord
from src.suppliers.registry import find_adapter
from src.suppliers.site_models import DownloadResult
from src.suppliers.utils import DEFAULT_HEADERS, json_loads_or, now_utc_iso


THREEDDD_DEFAULT_SESSION_DIR = Path("data/sourse/suppliers/3ddd_session")


def resolve_yadisk_public_download(public_url: str) -> tuple[Optional[str], Optional[str]]:
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    response = requests.get(api_url, params={"public_key": public_url}, headers=DEFAULT_HEADERS, timeout=45)
    response.raise_for_status()
    data = response.json()
    return data.get("href"), data.get("filename")


def _merge_extra_json(record: ProductRecord, patch: dict[str, object]) -> None:
    base = json_loads_or(record.extra_json, {})
    if not isinstance(base, dict):
        base = {}
    base.update(patch)
    record.extra_json = json.dumps(base, ensure_ascii=False)


def _three_ddd_slug(record: ProductRecord) -> Optional[str]:
    for raw in (
        record.external_id,
        _extra_json_value(record, "api_slug"),
        _slug_from_url(record.product_url),
        _slug_from_url(record.model_page_url),
        _slug_from_url(record.source_url),
    ):
        value = (str(raw or "").strip())
        if value:
            return value
    return None


def _slug_from_url(url: str | None) -> Optional[str]:
    if not url:
        return None
    path = urlparse(url).path.strip("/")
    if not path:
        return None
    return path.split("/")[-1] or None


def _extra_json_value(record: ProductRecord, key: str) -> object | None:
    try:
        extra = json.loads(record.extra_json or "{}")
    except Exception:
        return None
    if not isinstance(extra, dict):
        return None
    return extra.get(key)


def _default_3ddd_session_file(filename: str) -> str:
    return str((_three_ddd_session_dir() / filename).resolve())


def _three_ddd_session_dir() -> Path:
    value = (
        os.environ.get("SUPPLIER_3DDD_SESSION_DIR")
        or os.environ.get("THREEDDD_SESSION_DIR")
    )
    return Path(value or THREEDDD_DEFAULT_SESSION_DIR).expanduser().resolve()


def _cookie_sources_for_3ddd() -> tuple[str | None, str | None]:
    header = (
        os.environ.get("SUPPLIER_3DDD_COOKIES")
        or os.environ.get("SUPPLIER_3DDD_COOKIE")
        or os.environ.get("THREEDDD_COOKIES")
        or os.environ.get("THREEDDD_COOKIE")
    )
    file_path = (
        os.environ.get("SUPPLIER_3DDD_COOKIES_FILE")
        or os.environ.get("SUPPLIER_3DDD_COOKIE_FILE")
        or os.environ.get("THREEDDD_COOKIES_FILE")
        or os.environ.get("THREEDDD_COOKIE_FILE")
        or _default_3ddd_session_file("cookie.txt")
    )
    return header, file_path


def _auth_sources_for_3ddd() -> tuple[str | None, str | None]:
    token = (
        os.environ.get("SUPPLIER_3DDD_AUTH")
        or os.environ.get("SUPPLIER_3DDD_AUTHORIZATION")
        or os.environ.get("THREEDDD_AUTH")
        or os.environ.get("THREEDDD_AUTHORIZATION")
    )
    file_path = (
        os.environ.get("SUPPLIER_3DDD_AUTH_FILE")
        or os.environ.get("SUPPLIER_3DDD_AUTHORIZATION_FILE")
        or os.environ.get("THREEDDD_AUTH_FILE")
        or os.environ.get("THREEDDD_AUTHORIZATION_FILE")
        or _default_3ddd_session_file("auth.txt")
    )
    return token, file_path


def _refresh_sources_for_3ddd() -> tuple[str | None, str | None]:
    token = (
        os.environ.get("SUPPLIER_3DDD_REFRESH")
        or os.environ.get("SUPPLIER_3DDD_REFRESH_TOKEN")
        or os.environ.get("THREEDDD_REFRESH")
        or os.environ.get("THREEDDD_REFRESH_TOKEN")
    )
    file_path = (
        os.environ.get("SUPPLIER_3DDD_REFRESH_FILE")
        or os.environ.get("SUPPLIER_3DDD_REFRESH_TOKEN_FILE")
        or os.environ.get("THREEDDD_REFRESH_FILE")
        or os.environ.get("THREEDDD_REFRESH_TOKEN_FILE")
        or _default_3ddd_session_file("refresh.txt")
    )
    return token, file_path


def _3ddd_model_headers() -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://3ddd.ru",
        "Referer": "https://3ddd.ru/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Expires": "Sat, 01 Jan 2000 00:00:00 GMT",
        "X-Requested-With": "XMLHttpRequest",
    }


def _truthy_env(name: str) -> bool:
    value = str(os.environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _3ddd_auto_bootstrap_enabled() -> bool:
    return _truthy_env("SUPPLIER_3DDD_AUTO_BOOTSTRAP") or _truthy_env("THREEDDD_AUTO_BOOTSTRAP")


def _load_env_text(primary: list[str], fallback_files: list[str]) -> str | None:
    for name in primary:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    for name in fallback_files:
        value = _load_text_if_file(os.environ.get(name))
        if value:
            return value
    return None


def _normalize_bearer_token(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not text.lower().startswith("bearer "):
        text = f"Bearer {text}"
    return text


def _load_text_if_file(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _persist_text_if_path(path_str: str | None, value: str | None) -> None:
    if not path_str or value is None:
        return
    path = Path(path_str).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def _jwt_expiration_ts(auth_value: str | None) -> int | None:
    token = _normalize_bearer_token(auth_value)
    if not token:
        return None
    raw = token.split(" ", 1)[1].strip()
    parts = raw.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        data = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    exp = data.get("exp")
    try:
        return int(exp)
    except Exception:
        return None


def _auth_token_is_expired(auth_value: str | None, skew_sec: int = 90) -> bool:
    exp = _jwt_expiration_ts(auth_value)
    if exp is None:
        return not bool(_normalize_bearer_token(auth_value))
    now_ts = int(datetime.now(timezone.utc).timestamp())
    return exp <= (now_ts + skew_sec)


def _set_3ddd_cookie(session: requests.Session, name: str, value: str) -> None:
    session.cookies.set(name, value)
    for domain in ("3ddd.ru", ".3ddd.ru", "models.3ddd.ru", "b4.3ddd.ru", "b5.3ddd.ru"):
        session.cookies.set(name, value, domain=domain, path="/")


def _bootstrap_3ddd_session(reason: str) -> list[str]:
    notes = [f"3ddd_bootstrap_reason:{reason}"]
    if not _3ddd_auto_bootstrap_enabled():
        notes.append("3ddd_bootstrap_disabled")
        return notes

    email = _load_env_text(
        ["SUPPLIER_3DDD_EMAIL", "THREEDDD_EMAIL"],
        ["SUPPLIER_3DDD_EMAIL_FILE", "THREEDDD_EMAIL_FILE"],
    )
    password = _load_env_text(
        ["SUPPLIER_3DDD_PASSWORD", "THREEDDD_PASSWORD"],
        ["SUPPLIER_3DDD_PASSWORD_FILE", "THREEDDD_PASSWORD_FILE"],
    )
    if not email or not password:
        notes.append("3ddd_bootstrap_missing_credentials")
        return notes

    script_path = Path("src/suppliers/bootstrap_3ddd_session.py").resolve()
    if not script_path.is_file():
        notes.append("3ddd_bootstrap_script_missing")
        return notes

    env = os.environ.copy()
    env["SUPPLIER_3DDD_EMAIL"] = email
    env["SUPPLIER_3DDD_PASSWORD"] = password

    browser = (
        os.environ.get("SUPPLIER_3DDD_BROWSER")
        or os.environ.get("THREEDDD_BROWSER")
        or "safari"
    )
    timeout_sec = (
        os.environ.get("SUPPLIER_3DDD_BOOTSTRAP_TIMEOUT_SEC")
        or os.environ.get("THREEDDD_BOOTSTRAP_TIMEOUT_SEC")
        or "180"
    )
    session_dir = str(_three_ddd_session_dir())
    cmd = [
        sys.executable,
        str(script_path),
        "--browser",
        browser,
        "--out-dir",
        session_dir,
        "--timeout-sec",
        timeout_sec,
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(int(timeout_sec) + 60, 240),
            env=env,
        )
    except Exception as exc:
        notes.append(f"3ddd_bootstrap_failed:{type(exc).__name__}:{exc}")
        return notes

    if completed.returncode == 0:
        notes.append("3ddd_bootstrap_ok")
    else:
        stderr = (completed.stderr or "").strip().replace("\n", " ")[:400]
        stdout = (completed.stdout or "").strip().replace("\n", " ")[:400]
        detail = stderr or stdout or f"returncode={completed.returncode}"
        notes.append(f"3ddd_bootstrap_failed:{detail}")
    return notes


def _load_3ddd_session(allow_bootstrap: bool = True) -> tuple[requests.Session, list[str]]:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    notes: list[str] = []

    cookie_header, cookie_file = _cookie_sources_for_3ddd()
    if cookie_file:
        path = Path(cookie_file).expanduser()
        if path.is_file():
            loaded = False
            try:
                jar = http.cookiejar.MozillaCookieJar(str(path))
                jar.load(ignore_discard=True, ignore_expires=True)
                for cookie in jar:
                    session.cookies.set(cookie.name, cookie.value, domain=cookie.domain, path=cookie.path)
                loaded = True
                notes.append("3ddd_cookie_file:mozilla_cookiejar")
            except Exception:
                loaded = False

            if not loaded:
                raw = path.read_text(encoding="utf-8", errors="replace").strip()
                cookie_header = f"{cookie_header}; {raw}" if cookie_header and raw else (cookie_header or raw)
                notes.append("3ddd_cookie_file:raw_header")
        else:
            notes.append("3ddd_cookie_file_missing")

    if cookie_header:
        parsed = False
        text = cookie_header.strip()
        if text.startswith("{"):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    for key, value in loaded.items():
                        if key:
                            _set_3ddd_cookie(session, str(key), str(value))
                    parsed = True
                    notes.append("3ddd_cookie_header:json")
            except Exception:
                parsed = False
        if not parsed:
            pairs = [part.strip() for part in text.split(";") if "=" in part]
            for pair in pairs:
                name, value = pair.split("=", 1)
                name = name.strip()
                value = value.strip()
                if name:
                    _set_3ddd_cookie(session, name, value)
            if pairs:
                notes.append("3ddd_cookie_header:kv")

    auth_value, auth_file = _auth_sources_for_3ddd()
    if auth_file:
        file_value = _load_text_if_file(auth_file)
        if file_value:
            auth_value = file_value or auth_value
            notes.append("3ddd_auth_file:loaded")
        else:
            notes.append("3ddd_auth_file_missing")

    refresh_value, refresh_file = _refresh_sources_for_3ddd()
    if refresh_file:
        file_value = _load_text_if_file(refresh_file)
        if file_value:
            refresh_value = file_value or refresh_value
            notes.append("3ddd_refresh_file:loaded")
        else:
            notes.append("3ddd_refresh_file_missing")

    auth_text = _normalize_bearer_token(auth_value)
    if auth_text:
        session.headers["Authorization"] = auth_text
        notes.append("3ddd_auth_header:loaded")

    if refresh_value and _auth_token_is_expired(auth_text):
        try:
            response = session.post(
                "https://auth.3ddd.ru/api/token/refresh",
                headers=_3ddd_model_headers(),
                json={"refresh_token": str(refresh_value).strip()},
                timeout=(20, 60),
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else {}
            fresh_auth = _normalize_bearer_token((data or {}).get("token"))
            fresh_refresh = str((data or {}).get("refresh_token") or "").strip() or str(refresh_value).strip()
            if fresh_auth:
                session.headers["Authorization"] = fresh_auth
                _persist_text_if_path(auth_file, fresh_auth)
                _persist_text_if_path(refresh_file, fresh_refresh)
                notes.append("3ddd_auth_refreshed")
            else:
                notes.append("3ddd_auth_refresh_missing_token")
        except Exception as exc:
            notes.append(f"3ddd_auth_refresh_failed:{type(exc).__name__}:{exc}")

    has_auth = bool(session.headers.get("Authorization"))
    has_cookie = bool(list(session.cookies))
    if allow_bootstrap and (not has_auth or not has_cookie):
        notes.extend(_bootstrap_3ddd_session("missing_auth_or_cookie"))
        if "3ddd_bootstrap_ok" in notes:
            retry_session, retry_notes = _load_3ddd_session(allow_bootstrap=False)
            return retry_session, notes + retry_notes

    return session, notes


def _should_retry_3ddd_after_bootstrap(error: str | None, notes: list[str]) -> bool:
    haystack = " | ".join([*(notes or []), str(error or "")]).lower()
    retry_tokens = (
        "user/account",
        "login",
        "expired jwt token",
        "token is expired",
        "3ddd_cookie_file_missing",
        "3ddd_auth_file_missing",
        "3ddd_download_requires_auth",
        "3ddd session download failed",
    )
    return any(token in haystack for token in retry_tokens)


def _extract_download_url_candidates(payload: object) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for inner in value.values():
                walk(inner)
            return
        if isinstance(value, list):
            for inner in value:
                walk(inner)
            return
        if not isinstance(value, str):
            return

        text = value.strip()
        if not text:
            return  # pragma: no cover
        if text.startswith("http://") or text.startswith("https://"):
            if text not in seen:
                seen.add(text)
                candidates.append(text)
            return
        if text.startswith("/"):
            absolute = urljoin("https://3ddd.ru", text)
            if absolute not in seen:
                seen.add(absolute)
                candidates.append(absolute)

    walk(payload)
    return candidates


def _extract_response_filename(response: requests.Response, fallback: str | None = None) -> Optional[str]:
    filename = filename_from_headers(response.headers.get("Content-Disposition"))
    if filename:
        return filename
    final_url = response.url
    if final_url:
        inferred = SupplierAdapter.filename_from_url(final_url)
        if inferred:
            return inferred
    return fallback


def _save_response_binary(response: requests.Response, target_dir: Path, filename_hint: str | None = None) -> DownloadResult:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = _extract_response_filename(response, fallback=filename_hint or "model.bin") or "model.bin"
    target_path = target_dir / filename
    total = 0
    with open(target_path, "wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            file_obj.write(chunk)
            total += len(chunk)
    return DownloadResult(
        final_url=response.url,
        local_path=str(target_path),
        filename=filename,
        content_type=response.headers.get("Content-Type"),
        ok=True,
        size_bytes=total,
        error=None,
    )


def resolve_3ddd_download(record: ProductRecord) -> tuple[Optional[str], Optional[str], list[str]]:
    slug = _three_ddd_slug(record)
    if not slug:
        return None, None, ["3ddd_slug_missing"]

    session, notes = _load_3ddd_session()
    _merge_extra_json(record, {"api_slug": slug})
    filename_hint = record.model_download_filename

    attempts: list[tuple[str, str, dict[str, object] | None, dict[str, str] | None]] = [
        (
            "post_button_slug",
            "https://models.3ddd.ru/api/models/button",
            {"slug": slug},
            _3ddd_model_headers(),
        ),
        (
            "get_direct_slug",
            f"https://3ddd.ru/api/models/download/{slug}",
            None,
            _3ddd_model_headers(),
        ),
        (
            "post_json_slug",
            "https://3ddd.ru/api/models/download",
            {"slug": slug},
            _3ddd_model_headers(),
        ),
        (
            "get_slug_query",
            f"https://3ddd.ru/api/models/download?slug={slug}",
            None,
            {"Accept": "*/*", "X-Requested-With": "XMLHttpRequest"},
        ),
    ]

    external_id = str(record.external_id or "").strip()
    if external_id and external_id != slug:
        attempts.insert(  # pragma: no cover
            1,
            (
                "post_json_id",
                "https://3ddd.ru/api/models/download",
                {"id": external_id},
                _3ddd_model_headers(),
            ),
        )

    for strategy, url, payload, headers in attempts:
        try:
            if payload is None:
                response = session.get(url, headers=headers, timeout=(20, 60), allow_redirects=True, stream=True)
            else:
                response = session.post(url, headers=headers, json=payload, timeout=(20, 60), allow_redirects=True, stream=True)
        except Exception as exc:
            notes.append(f"3ddd_download_attempt_failed:{strategy}:{type(exc).__name__}:{exc}")
            continue

        content_type = (response.headers.get("Content-Type") or "").lower()
        final_url = response.url or url
        notes.append(f"3ddd_download_attempt:{strategy}:status={response.status_code}")

        if response.status_code >= 400:
            try:
                body_preview = response.text[:300]
            except Exception:
                body_preview = ""
            if body_preview:
                notes.append(f"3ddd_download_attempt_body:{strategy}:{body_preview}")
            response.close()
            continue

        if any(token in content_type for token in ("application/zip", "application/octet-stream", "application/x-rar", "application/x-7z")):
            filename = _extract_response_filename(response, fallback=filename_hint)
            response.close()
            return final_url, filename, notes

        if "application/json" in content_type:
            try:
                payload_data = response.json()
            except Exception as exc:
                notes.append(f"3ddd_download_json_parse_failed:{strategy}:{type(exc).__name__}:{exc}")
                response.close()
                continue
            response.close()

            if strategy == "post_button_slug":
                state = (((payload_data or {}).get("data") or {}).get("state") or "")
                if state:
                    notes.append(f"3ddd_button_state:{state}")
                disabled = (((payload_data or {}).get("data") or {}).get("disabled"))
                if disabled is not None:
                    notes.append(f"3ddd_button_disabled:{disabled}")

            urls = _extract_download_url_candidates(payload_data)
            urls = [candidate for candidate in urls if "/api/models/download" not in candidate]
            if urls:
                _merge_extra_json(record, {"download_resolver_payload": payload_data})
                return urls[0], SupplierAdapter.filename_from_url(urls[0]), notes

            notes.append(f"3ddd_download_json_no_url:{strategy}")
            continue

        if response.history:
            redirect_url = final_url
            if redirect_url and redirect_url != url:
                response.close()
                return redirect_url, SupplierAdapter.filename_from_url(redirect_url), notes

        if "text/html" in content_type:
            try:
                body_preview = response.text[:300]
            except Exception:  # pragma: no cover
                body_preview = ""  # pragma: no cover
            if "auth" in final_url or "login" in final_url or "login" in body_preview.lower():
                notes.append(f"3ddd_download_requires_auth:{strategy}")
            response.close()
            continue

        response.close()  # pragma: no cover

    return None, None, notes


def enrich_model_download(record: ProductRecord) -> ProductRecord:
    if record.source_site == "loftdesigne" and record.model_download_landing_url:
        try:
            direct_url, filename = resolve_yadisk_public_download(record.model_download_landing_url)
            record.model_download_url = direct_url
            record.model_download_filename = filename or SupplierAdapter.filename_from_url(direct_url)
            record.model_format = SupplierAdapter.ext_from_url(filename or direct_url)
            record.model_extraction_method = "yadisk_public_api"
        except Exception as exc:  # pragma: no cover
            _merge_extra_json(record, {"model_resolution_error": f"{type(exc).__name__}: {exc}"})  # pragma: no cover

    if record.source_site == "3ddd" and not record.model_download_url:
        try:
            direct_url, filename, notes = resolve_3ddd_download(record)
            if direct_url:
                record.model_download_url = direct_url
                record.model_download_filename = filename or SupplierAdapter.filename_from_url(direct_url)
                record.model_format = SupplierAdapter.ext_from_url(record.model_download_filename or direct_url)
                record.model_extraction_method = "3ddd_download_resolver"
            _merge_extra_json(
                record,
                {
                    "download_resolver_notes": notes,
                    "download_requires_auth": not bool(direct_url),
                },
            )
        except Exception as exc:  # pragma: no cover
            _merge_extra_json(record, {"model_resolution_error": f"{type(exc).__name__}: {exc}"})  # pragma: no cover

    return record


def filename_from_headers(content_disposition: str | None) -> Optional[str]:
    if not content_disposition:
        return None

    match = re.search(r"""filename\*=UTF-8''([^;]+)""", content_disposition, re.IGNORECASE)
    if match:
        return unquote(match.group(1))

    match = re.search(r'''filename="?([^"]+)"?''', content_disposition, re.IGNORECASE)
    if match:
        return match.group(1)

    return None  # pragma: no cover


def _download_binary_curl(url: str, target_dir: Path, filename_hint: str | None = None) -> DownloadResult:
    final_filename = filename_hint or SupplierAdapter.filename_from_url(url) or "model.bin"
    target_path = target_dir / final_filename

    cmd = [
        "curl",
        "-L",
        "--fail",
        "--connect-timeout",
        "30",
        "--max-time",
        "900",
        "--speed-time",
        "60",
        "--speed-limit",
        "10240",
        "-A",
        DEFAULT_HEADERS.get("User-Agent", "Mozilla/5.0"),
        "-o",
        str(target_path),
        url,
    ]

    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=960)
    except Exception as exc:
        return DownloadResult(
            final_url=None,
            local_path=None,
            filename=final_filename,
            content_type=None,
            ok=False,
            size_bytes=0,
            error=f"curl {type(exc).__name__}: {exc}",
        )

    if completed.returncode != 0 or not target_path.is_file():
        return DownloadResult(
            final_url=None,
            local_path=None,
            filename=final_filename,
            content_type=None,
            ok=False,
            size_bytes=0,
            error=f"curl returncode={completed.returncode}: {completed.stderr.strip()[:500]}",
        )

    return DownloadResult(
        final_url=url,
        local_path=str(target_path),
        filename=final_filename,
        content_type=None,
        ok=True,
        size_bytes=target_path.stat().st_size,
        error=None,
    )


def download_binary(url: str, target_dir: Path, filename_hint: str | None = None) -> DownloadResult:
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=(30, 300),
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type")
            final_url = response.url
            content_disposition = response.headers.get("Content-Disposition")

            final_filename = (
                filename_from_headers(content_disposition)
                or filename_hint
                or SupplierAdapter.filename_from_url(final_url)
                or "model.bin"
            )

            target_path = target_dir / final_filename
            total = 0

            with open(target_path, "wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue  # pragma: no cover
                    file_obj.write(chunk)
                    total += len(chunk)

            return DownloadResult(
                final_url=final_url,
                local_path=str(target_path),
                filename=final_filename,
                content_type=content_type,
                ok=True,
                size_bytes=total,
                error=None,
            )
    except Exception as exc:
        curl_result = _download_binary_curl(url, target_dir, filename_hint=filename_hint)
        if curl_result.ok:
            return curl_result
        return DownloadResult(
            final_url=None,
            local_path=None,
            filename=filename_hint,
            content_type=None,
            ok=False,
            size_bytes=0,
            error=f"{type(exc).__name__}: {exc}; {curl_result.error}",
        )


def _download_binary_with_session(
    session: requests.Session,
    url: str,
    target_dir: Path,
    filename_hint: str | None = None,
) -> DownloadResult:
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with session.get(
            url,
            timeout=(30, 300),
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            return _save_response_binary(response, target_dir, filename_hint=filename_hint)
    except Exception as exc:
        return DownloadResult(
            final_url=None,
            local_path=None,
            filename=filename_hint,
            content_type=None,
            ok=False,
            size_bytes=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def download_binary_for_record(record: ProductRecord, target_dir: Path, filename_hint: str | None = None) -> DownloadResult:
    record = enrich_model_download(record)

    if record.model_download_url:
        if record.source_site == "3ddd":
            session, notes = _load_3ddd_session()
            result = _download_binary_with_session(
                session,
                record.model_download_url,
                target_dir,
                filename_hint=filename_hint or record.model_download_filename,
            )
            if result.ok:
                return result
            if _should_retry_3ddd_after_bootstrap(result.error, notes):
                retry_notes = _bootstrap_3ddd_session("session_download_failed")
                notes.extend(retry_notes)
                if "3ddd_bootstrap_ok" in retry_notes:
                    retry_session, reloaded_notes = _load_3ddd_session(allow_bootstrap=False)
                    notes.extend(reloaded_notes)
                    retry_result = _download_binary_with_session(
                        retry_session,
                        record.model_download_url,
                        target_dir,
                        filename_hint=filename_hint or record.model_download_filename,
                    )
                    if retry_result.ok:
                        return retry_result
                    result = retry_result
            return DownloadResult(
                final_url=result.final_url,
                local_path=result.local_path,
                filename=result.filename,
                content_type=result.content_type,
                ok=False,
                size_bytes=result.size_bytes,
                error="3ddd session download failed; "
                + " | ".join(notes[-4:])
                + (f" | {result.error}" if result.error else ""),
            )
        return download_binary(record.model_download_url, target_dir, filename_hint=filename_hint or record.model_download_filename)

    if record.source_site == "3ddd":
        slug = _three_ddd_slug(record)
        if not slug:
            return DownloadResult(
                final_url=None,
                local_path=None,
                filename=filename_hint,
                content_type=None,
                ok=False,
                size_bytes=0,
                error="3ddd slug is empty",
            )

        session, notes = _load_3ddd_session()
        attempts = [
            (
                "post_button_slug",
                "https://models.3ddd.ru/api/models/button",
                {"slug": slug},
                _3ddd_model_headers(),
            ),
            (
                "get_direct_slug",
                f"https://3ddd.ru/api/models/download/{slug}",
                None,
                _3ddd_model_headers(),
            ),
            (
                "post_json_slug",
                "https://3ddd.ru/api/models/download",
                {"slug": slug},
                _3ddd_model_headers(),
            ),
            (
                "get_slug_query",
                f"https://3ddd.ru/api/models/download?slug={slug}",
                None,
                {"Accept": "*/*", "X-Requested-With": "XMLHttpRequest"},
            ),
        ]

        external_id = str(record.external_id or "").strip()
        if external_id and external_id != slug:
            attempts.insert(  # pragma: no cover
                1,
                (
                    "post_json_id",
                    "https://3ddd.ru/api/models/download",
                    {"id": external_id},
                    _3ddd_model_headers(),
                ),
            )

        for strategy, url, payload, headers in attempts:
            try:
                if payload is None:
                    response = session.get(url, headers=headers, timeout=(20, 120), allow_redirects=True, stream=True)
                else:
                    response = session.post(url, headers=headers, json=payload, timeout=(20, 120), allow_redirects=True, stream=True)
            except Exception as exc:
                notes.append(f"{strategy}:{type(exc).__name__}:{exc}")
                continue

            content_type = (response.headers.get("Content-Type") or "").lower()
            final_url = response.url or url

            if response.status_code >= 400:
                try:
                    body_preview = response.text[:300]
                except Exception:  # pragma: no cover
                    body_preview = ""  # pragma: no cover
                notes.append(f"{strategy}:status={response.status_code}:{body_preview}")
                response.close()
                continue

            if any(token in content_type for token in ("application/zip", "application/octet-stream", "application/x-rar", "application/x-7z")):
                return _save_response_binary(response, target_dir, filename_hint=filename_hint or record.model_download_filename)

            if "application/json" in content_type:
                try:
                    payload_data = response.json()
                except Exception as exc:
                    notes.append(f"{strategy}:json_parse_failed:{type(exc).__name__}:{exc}")
                    response.close()
                    continue
                response.close()

                if strategy == "post_button_slug":
                    state = (((payload_data or {}).get("data") or {}).get("state") or "")
                    if state:
                        notes.append(f"{strategy}:state={state}")
                    disabled = (((payload_data or {}).get("data") or {}).get("disabled"))
                    if disabled is not None:
                        notes.append(f"{strategy}:disabled={disabled}")

                urls = _extract_download_url_candidates(payload_data)
                urls = [candidate for candidate in urls if "/api/models/download" not in candidate]
                if urls:
                    record.model_download_url = urls[0]
                    record.model_download_filename = record.model_download_filename or SupplierAdapter.filename_from_url(urls[0])
                    record.model_format = record.model_format or SupplierAdapter.ext_from_url(record.model_download_filename or urls[0])
                    _merge_extra_json(record, {"download_resolver_notes": notes, "download_resolver_payload": payload_data})
                    result = _download_binary_with_session(
                        session,
                        urls[0],
                        target_dir,
                        filename_hint=filename_hint or record.model_download_filename,
                    )
                    if result.ok:
                        return result
                    if _should_retry_3ddd_after_bootstrap(result.error, notes):
                        retry_notes = _bootstrap_3ddd_session(f"{strategy}_session_download_failed")
                        notes.extend(retry_notes)
                        if "3ddd_bootstrap_ok" in retry_notes:
                            retry_session, reloaded_notes = _load_3ddd_session(allow_bootstrap=False)
                            notes.extend(reloaded_notes)
                            retry_result = _download_binary_with_session(
                                retry_session,
                                urls[0],
                                target_dir,
                                filename_hint=filename_hint or record.model_download_filename,
                            )
                            if retry_result.ok:
                                return retry_result
                            result = retry_result  # pragma: no cover
                    notes.append(f"{strategy}:session_download_failed:{result.error}")  # pragma: no cover
                    continue  # pragma: no cover

                notes.append(f"{strategy}:json_no_direct_url")
                continue

            if response.history and final_url and final_url != url:
                response.close()
                result = _download_binary_with_session(
                    session,
                    final_url,
                    target_dir,
                    filename_hint=filename_hint or record.model_download_filename,
                )
                if result.ok:
                    return result
                notes.append(f"{strategy}:redirect_session_download_failed:{result.error}")  # pragma: no cover
                continue  # pragma: no cover

            if "text/html" in content_type:
                try:
                    body_preview = response.text[:300]
                except Exception:  # pragma: no cover
                    body_preview = ""  # pragma: no cover
                notes.append(f"{strategy}:html:{final_url}:{body_preview}")
                response.close()
                continue

            response.close()  # pragma: no cover

        return DownloadResult(
            final_url=None,
            local_path=None,
            filename=filename_hint or record.model_download_filename,
            content_type=None,
            ok=False,
            size_bytes=0,
            error="3ddd download unresolved; " + " | ".join(notes[-8:]),
        )

    return DownloadResult(
        final_url=None,
        local_path=None,
        filename=filename_hint or record.model_download_filename,
        content_type=None,
        ok=False,
        size_bytes=0,
        error="model_download_url is empty",
    )


def save_metadata_json(record: ProductRecord, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^\w\-\.]+", "_", (record.title or record.unique_key), flags=re.UNICODE).strip("_")
    suffix_source = record.external_id or record.unique_key
    suffix = re.sub(r"[^\w\-\.]+", "_", str(suffix_source), flags=re.UNICODE).strip("_")
    slug = f"{base}__{suffix}" if base and suffix and suffix != base else (base or suffix or "product")
    path = out_dir / f"{slug}.metadata.json"
    path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_url(url: str, html: str, final_url: str | None = None) -> ProductRecord:
    effective_url = final_url or url
    adapter = find_adapter(effective_url)
    raw_items = adapter.parse(url, html, effective_url)
    if not raw_items:
        raise ValueError(f"Адаптер {adapter.site_name} не вернул ни одной записи")
    item = raw_items[0]
    if isinstance(item, ProductRecord):
        return item
    raise TypeError(f"Unsupported adapter result type: {type(item).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="URL карточки товара")
    ap.add_argument("--db", default="out/supplier_ingest/suppliers.db")
    ap.add_argument("--out-dir", default="out/supplier_ingest/items")
    ap.add_argument("--no-download-model", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    init_db(db_path)

    adapter = find_adapter(args.url)
    html, final_page_url = adapter.fetch_html(args.url)
    raw_items = adapter.parse(args.url, html, final_page_url)
    if not raw_items:
        if getattr(adapter, "empty_parse_is_skip", False):
            print("status: skipped_empty_result")
            print(f"site: {adapter.site_name}")
            return
        raise ValueError(f"Адаптер {adapter.site_name} не вернул ни одной записи")

    first_item = raw_items[0]
    if isinstance(first_item, ProductRecord):
        record = first_item
    else:
        raise TypeError(f"Unsupported adapter result type: {type(first_item).__name__}")
    record = enrich_model_download(record)

    upsert_product(db_path, record)
    meta_path = save_metadata_json(record, out_dir)

    print(f"metadata_json: {meta_path}")
    print(f"title: {record.title}")
    print(f"model_download_url: {record.model_download_url}")

    if args.no_download_model:
        return

    if not record.model_download_url:
        insert_download(
            db_path=db_path,
            unique_key=record.unique_key,
            downloaded_at=now_utc_iso(),
            final_url=None,
            local_path=None,
            filename=None,
            content_type=None,
            ok=False,
            size_bytes=0,
            error="model_download_url is empty",
        )
        print("model: no downloadable url")
        return

    item_slug = re.sub(r"[^\w\-\.]+", "_", (record.title or record.unique_key), flags=re.UNICODE).strip("_")
    model_dir = out_dir / item_slug

    result = download_binary_for_record(record, model_dir, filename_hint=record.model_download_filename)

    insert_download(
        db_path=db_path,
        unique_key=record.unique_key,
        downloaded_at=now_utc_iso(),
        final_url=result.final_url,
        local_path=result.local_path,
        filename=result.filename,
        content_type=result.content_type,
        ok=result.ok,
        size_bytes=result.size_bytes,
        error=result.error,
    )

    if result.ok:
        print(f"model_saved: {result.local_path}")
        print(f"size_bytes: {result.size_bytes}")
    else:
        print(f"model_download_error: {result.error}")


if __name__ == "__main__":
    main()  # pragma: no cover
