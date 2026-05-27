from __future__ import annotations

import base64
import json
import sys
import time
import types
from pathlib import Path

import pytest

from src.suppliers.models import ProductRecord
from src.suppliers.site_models import DownloadResult
from src.suppliers import fetch_product_and_model as fetcher


def record(**overrides) -> ProductRecord:
    data = {
        "unique_key": "site::1",
        "source_site": "3ddd",
        "source_url": "https://3ddd.ru/3dmodels/show/model_slug",
        "parsed_at": "2026-01-01T00:00:00Z",
        "external_id": "123",
        "title": "Model Name",
        "product_url": "https://3ddd.ru/3dmodels/show/model_slug",
        "extra_json": "{}",
    }
    data.update(overrides)
    return ProductRecord(**data)


class FakeCookies:
    def __init__(self):
        self.items = []

    def set(self, name, value, domain=None, path=None):
        self.items.append((name, value, domain, path))

    def __iter__(self):
        return iter(self.items)


class FakeResponse:
    def __init__(
        self,
        *,
        url="https://example.com/file.zip",
        status_code=200,
        headers=None,
        chunks=(b"abc",),
        payload=None,
        text="",
        history=None,
        raise_error=None,
    ):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks
        self._payload = payload
        self.text = text
        self.history = history or []
        self._raise_error = raise_error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload

    def iter_content(self, chunk_size=1024):
        yield from self._chunks

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses=None):
        self.headers = {}
        self.cookies = FakeCookies()
        self.responses = list(responses or [])
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if not self.responses:
            raise RuntimeError("no fake response")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if not self.responses:
            raise RuntimeError("no fake response")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def jwt_with_exp(exp: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"Bearer aaa.{payload}.zzz"


def test_basic_record_env_token_and_cookie_helpers(monkeypatch, tmp_path):
    rec = record(extra_json='{"old": 1}', external_id="", product_url="", model_page_url="https://3ddd.ru/3dmodels/show/from_page")
    fetcher._merge_extra_json(rec, {"new": 2})
    assert json.loads(rec.extra_json) == {"old": 1, "new": 2}
    assert fetcher._three_ddd_slug(rec) == "from_page"
    assert fetcher._slug_from_url("https://site/path/to/file.obj?x=1") == "file.obj"
    assert fetcher._extra_json_value(record(extra_json="{bad"), "x") is None

    monkeypatch.setenv("SUPPLIER_3DDD_SESSION_DIR", str(tmp_path / "session"))
    assert fetcher._three_ddd_session_dir() == (tmp_path / "session").resolve()
    assert fetcher._default_3ddd_session_file("auth.txt").endswith("auth.txt")
    assert fetcher._3ddd_model_headers()["X-Requested-With"] == "XMLHttpRequest"

    monkeypatch.setenv("SUPPLIER_3DDD_AUTO_BOOTSTRAP", "yes")
    assert fetcher._3ddd_auto_bootstrap_enabled()
    monkeypatch.setenv("SUPPLIER_3DDD_AUTH", "token")
    assert fetcher._load_env_text(["SUPPLIER_3DDD_AUTH"], []) == "token"
    assert fetcher._normalize_bearer_token("token") == "Bearer token"
    assert fetcher._normalize_bearer_token("Bearer token") == "Bearer token"

    text_file = tmp_path / "value.txt"
    fetcher._persist_text_if_path(str(text_file), " value ")
    assert fetcher._load_text_if_file(str(text_file)) == "value"
    assert fetcher._load_text_if_file(str(tmp_path / "missing.txt")) is None

    future = jwt_with_exp(int(time.time()) + 3600)
    past = jwt_with_exp(int(time.time()) - 3600)
    assert fetcher._jwt_expiration_ts(future) is not None
    assert not fetcher._auth_token_is_expired(future)
    assert fetcher._auth_token_is_expired(past)
    assert fetcher._jwt_expiration_ts("bad-token") is None

    session = FakeSession()
    fetcher._set_3ddd_cookie(session, "sid", "abc")
    domains = {domain for _name, _value, domain, _path in session.cookies.items}
    assert {"3ddd.ru", ".3ddd.ru", "models.3ddd.ru"} <= domains


def test_load_3ddd_session_reads_cookie_auth_and_refreshes(monkeypatch, tmp_path):
    cookie_file = tmp_path / "cookie.txt"
    cookie_file.write_text("sid=abc; user=ivan", encoding="utf-8")
    auth_file = tmp_path / "auth.txt"
    refresh_file = tmp_path / "refresh.txt"
    auth_file.write_text(jwt_with_exp(int(time.time()) - 1000), encoding="utf-8")
    refresh_file.write_text("refresh-token", encoding="utf-8")

    monkeypatch.setenv("SUPPLIER_3DDD_COOKIES_FILE", str(cookie_file))
    monkeypatch.setenv("SUPPLIER_3DDD_AUTH_FILE", str(auth_file))
    monkeypatch.setenv("SUPPLIER_3DDD_REFRESH_FILE", str(refresh_file))
    fresh = jwt_with_exp(int(time.time()) + 5000)
    fake_session = FakeSession(
        [
            FakeResponse(
                headers={"Content-Type": "application/json"},
                payload={"data": {"token": fresh.removeprefix("Bearer "), "refresh_token": "fresh-refresh"}},
            )
        ]
    )
    monkeypatch.setattr(fetcher.requests, "Session", lambda: fake_session)

    session, notes = fetcher._load_3ddd_session(allow_bootstrap=False)
    assert session is fake_session
    assert "3ddd_cookie_file:raw_header" in notes
    assert "3ddd_auth_file:loaded" in notes
    assert "3ddd_auth_refreshed" in notes
    assert auth_file.read_text(encoding="utf-8").startswith("Bearer ")
    assert refresh_file.read_text(encoding="utf-8").strip() == "fresh-refresh"


def test_resolve_3ddd_download_handles_binary_json_redirect_and_failures(monkeypatch):
    binary_session = FakeSession(
        [
            FakeResponse(
                url="https://cdn.3ddd.ru/model.rar",
                headers={"Content-Type": "application/x-rar", "Content-Disposition": 'attachment; filename="model.rar"'},
            )
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (binary_session, ["loaded"]))
    direct_url, filename, notes = fetcher.resolve_3ddd_download(record(model_download_filename="hint.rar"))
    assert direct_url == "https://cdn.3ddd.ru/model.rar"
    assert filename == "model.rar"
    assert any("post_button_slug:status=200" in note for note in notes)

    json_session = FakeSession(
        [
            FakeResponse(status_code=403, text="login required"),
            FakeResponse(
                url="https://3ddd.ru/api/models/download/model_slug",
                headers={"Content-Type": "application/json"},
                payload={"data": {"url": "https://cdn.3ddd.ru/from-json.zip"}},
            ),
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (json_session, []))
    rec = record(external_id="")
    direct_url, filename, notes = fetcher.resolve_3ddd_download(rec)
    assert direct_url == "https://cdn.3ddd.ru/from-json.zip"
    assert filename == "from-json.zip"
    assert json.loads(rec.extra_json)["download_resolver_payload"]["data"]["url"].endswith(".zip")

    assert fetcher.resolve_3ddd_download(record(source_url="", product_url="", model_page_url="", external_id="", extra_json="{}")) == (
        None,
        None,
        ["3ddd_slug_missing"],
    )
    assert fetcher._should_retry_3ddd_after_bootstrap("expired jwt token", [])
    assert fetcher._extract_download_url_candidates({"a": ["/download/file.zip", "https://x/y.rar", "https://x/y.rar"]}) == [
        "https://3ddd.ru/download/file.zip",
        "https://x/y.rar",
    ]


def test_download_binary_paths_and_response_filename_helpers(monkeypatch, tmp_path):
    assert fetcher.filename_from_headers("attachment; filename*=UTF-8''%D1%82%D0%B5%D1%81%D1%82.rar") == "тест.rar"
    assert fetcher.filename_from_headers('attachment; filename="model.zip"') == "model.zip"
    response = FakeResponse(
        url="https://example.com/path/final.glb",
        headers={"Content-Disposition": 'attachment; filename="asset.glb"', "Content-Type": "model/gltf-binary"},
        chunks=(b"aa", b"", b"bb"),
    )
    saved = fetcher._save_response_binary(response, tmp_path, filename_hint="fallback.bin")
    assert saved.ok
    assert saved.filename == "asset.glb"
    assert Path(saved.local_path).read_bytes() == b"aabb"

    monkeypatch.setattr(fetcher.requests, "get", lambda *args, **kwargs: FakeResponse(url="https://cdn/file.obj", chunks=(b"obj",)))
    result = fetcher.download_binary("https://cdn/file.obj", tmp_path)
    assert result.ok
    assert Path(result.local_path).read_bytes() == b"obj"

    monkeypatch.setattr(fetcher.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network")))
    monkeypatch.setattr(
        fetcher,
        "_download_binary_curl",
        lambda url, target_dir, filename_hint=None: DownloadResult(url, str(target_dir / "curl.bin"), "curl.bin", None, True, 3),
    )
    assert fetcher.download_binary("https://cdn/fallback.bin", tmp_path).filename == "curl.bin"

    session = FakeSession([RuntimeError("nope")])
    failed = fetcher._download_binary_with_session(session, "https://cdn/x.zip", tmp_path, filename_hint="x.zip")
    assert not failed.ok
    assert "RuntimeError" in failed.error


def test_download_binary_for_record_and_enrichment_branches(monkeypatch, tmp_path):
    loft = record(source_site="loftdesigne", model_download_landing_url="https://yadi.sk/d/model", model_download_url=None)
    monkeypatch.setattr(fetcher, "resolve_yadisk_public_download", lambda url: ("https://downloader/model.zip", "model.zip"))
    enriched = fetcher.enrich_model_download(loft)
    assert enriched.model_download_url == "https://downloader/model.zip"
    assert enriched.model_format == ".zip"
    assert enriched.model_extraction_method == "yadisk_public_api"

    three = record(model_download_url=None)
    monkeypatch.setattr(fetcher, "resolve_3ddd_download", lambda rec: ("https://cdn/model.rar", "model.rar", ["ok"]))
    enriched = fetcher.enrich_model_download(three)
    assert enriched.model_download_url == "https://cdn/model.rar"
    assert json.loads(enriched.extra_json)["download_requires_auth"] is False

    direct = record(
        source_site="homeconcept",
        model_download_url="https://cdn/direct.zip",
        model_download_filename="direct.zip",
    )
    monkeypatch.setattr(fetcher, "download_binary", lambda url, target_dir, filename_hint=None: DownloadResult(url, str(target_dir / filename_hint), filename_hint, None, True, 10))
    assert fetcher.download_binary_for_record(direct, tmp_path).ok

    retry = record(model_download_url="https://cdn/secure.zip", model_download_filename="secure.zip")
    sessions = [(FakeSession(), ["expired jwt token"]), (FakeSession(), ["reloaded"])]
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: sessions.pop(0))
    results = [
        DownloadResult(None, None, "secure.zip", None, False, 0, "expired jwt token"),
        DownloadResult("https://cdn/secure.zip", str(tmp_path / "secure.zip"), "secure.zip", None, True, 5),
    ]
    monkeypatch.setattr(fetcher, "_download_binary_with_session", lambda *args, **kwargs: results.pop(0))
    monkeypatch.setattr(fetcher, "_bootstrap_3ddd_session", lambda reason: ["3ddd_bootstrap_ok"])
    assert fetcher.download_binary_for_record(retry, tmp_path).ok

    empty = record(
        source_site="homeconcept",
        model_download_url=None,
        external_id="",
        product_url="",
        model_page_url="",
        source_url="",
    )
    failed = fetcher.download_binary_for_record(empty, tmp_path, filename_hint="missing.bin")
    assert not failed.ok
    assert failed.error == "model_download_url is empty"


def test_download_binary_for_3ddd_without_prefilled_url_covers_resolver_loop(monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher, "enrich_model_download", lambda rec: rec)

    binary_session = FakeSession(
        [
            FakeResponse(
                url="https://cdn.3ddd.ru/direct.rar",
                headers={"Content-Type": "application/octet-stream", "Content-Disposition": 'attachment; filename="direct.rar"'},
                chunks=(b"rar",),
            )
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (binary_session, ["loaded"]))
    binary = fetcher.download_binary_for_record(record(model_download_url=None), tmp_path / "binary")
    assert binary.ok
    assert binary.filename == "direct.rar"
    assert Path(binary.local_path).read_bytes() == b"rar"

    json_session = FakeSession(
        [
            FakeResponse(
                headers={"Content-Type": "application/json"},
                payload={"data": {"download": "https://cdn.3ddd.ru/json.zip", "api": "https://3ddd.ru/api/models/download/skip"}},
            ),
            FakeResponse(url="https://cdn.3ddd.ru/json.zip", headers={"Content-Disposition": 'attachment; filename="json.zip"'}, chunks=(b"zip",)),
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (json_session, ["loaded"]))
    json_result = fetcher.download_binary_for_record(record(model_download_url=None), tmp_path / "json")
    assert json_result.ok
    assert json_result.filename == "json.zip"

    redirect_session = FakeSession(
        [
            FakeResponse(
                url="https://cdn.3ddd.ru/redirected.7z",
                headers={"Content-Type": "text/plain"},
                history=[object()],
            ),
            FakeResponse(url="https://cdn.3ddd.ru/redirected.7z", chunks=(b"7z",)),
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (redirect_session, ["loaded"]))
    redirected = fetcher.download_binary_for_record(record(model_download_url=None), tmp_path / "redirect")
    assert redirected.ok
    assert Path(redirected.local_path).read_bytes() == b"7z"

    bad_json_session = FakeSession(
        [
            FakeResponse(headers={"Content-Type": "application/json"}, payload=ValueError("bad json")),
            FakeResponse(status_code=404, text="missing"),
            RuntimeError("network down"),
            FakeResponse(headers={"Content-Type": "text/html"}, text="<html>login</html>", url="https://3ddd.ru/login"),
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (bad_json_session, ["loaded"]))
    unresolved = fetcher.download_binary_for_record(record(model_download_url=None, external_id=""), tmp_path / "unresolved")
    assert not unresolved.ok
    assert "3ddd download unresolved" in unresolved.error


def test_metadata_parse_url_and_yadisk_api_mocks(monkeypatch, tmp_path):
    metadata = fetcher.save_metadata_json(record(title="Товар / test", external_id="id 1"), tmp_path)
    assert metadata.name.startswith("Товар_test__id_1")
    assert json.loads(metadata.read_text(encoding="utf-8"))["unique_key"] == "site::1"

    parsed_record = record(source_site="homeconcept")
    adapter = types.SimpleNamespace(site_name="unit", parse=lambda url, html, final_url: [parsed_record])
    monkeypatch.setattr(fetcher, "find_adapter", lambda url: adapter)
    assert fetcher.parse_url("https://site/item", "<html>", final_url="https://site/final") is parsed_record

    empty_adapter = types.SimpleNamespace(site_name="empty", parse=lambda url, html, final_url: [])
    monkeypatch.setattr(fetcher, "find_adapter", lambda url: empty_adapter)
    with pytest.raises(ValueError):
        fetcher.parse_url("https://site/item", "<html>")

    monkeypatch.setattr(
        fetcher.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(payload={"href": "https://download", "filename": "asset.rar"}),
    )
    assert fetcher.resolve_yadisk_public_download("https://yadi.sk/d/x") == ("https://download", "asset.rar")


def test_bootstrap_curl_and_main_paths_are_fully_mocked(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("SUPPLIER_3DDD_AUTO_BOOTSTRAP", raising=False)
    assert fetcher._bootstrap_3ddd_session("unit") == ["3ddd_bootstrap_reason:unit", "3ddd_bootstrap_disabled"]

    monkeypatch.setenv("SUPPLIER_3DDD_AUTO_BOOTSTRAP", "1")
    monkeypatch.delenv("SUPPLIER_3DDD_EMAIL", raising=False)
    monkeypatch.delenv("SUPPLIER_3DDD_PASSWORD", raising=False)
    assert "3ddd_bootstrap_missing_credentials" in fetcher._bootstrap_3ddd_session("unit")

    monkeypatch.setenv("SUPPLIER_3DDD_EMAIL", "mail@example.test")
    monkeypatch.setenv("SUPPLIER_3DDD_PASSWORD", "secret")
    monkeypatch.setattr(fetcher.Path, "is_file", lambda self: False)
    assert "3ddd_bootstrap_script_missing" in fetcher._bootstrap_3ddd_session("unit")

    monkeypatch.setattr(fetcher.Path, "is_file", lambda self: True)
    monkeypatch.setattr(fetcher.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    assert "3ddd_bootstrap_ok" in fetcher._bootstrap_3ddd_session("unit")
    monkeypatch.setattr(fetcher.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=2, stdout="", stderr="bad login"))
    assert any(note.startswith("3ddd_bootstrap_failed:bad login") for note in fetcher._bootstrap_3ddd_session("unit"))
    monkeypatch.setattr(fetcher.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert any(note.startswith("3ddd_bootstrap_failed:RuntimeError:boom") for note in fetcher._bootstrap_3ddd_session("unit"))

    target = tmp_path / "curl"
    target.mkdir()
    monkeypatch.setattr(fetcher.Path, "is_file", lambda self: self.exists() and not self.is_dir())
    monkeypatch.setattr(fetcher.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    ok = fetcher._download_binary_curl("https://cdn.example.test/model.bin", target, filename_hint="asset.bin")
    assert not ok.ok
    assert "returncode=0" in ok.error
    (target / "asset.bin").write_bytes(b"asset")
    ok = fetcher._download_binary_curl("https://cdn.example.test/model.bin", target, filename_hint="asset.bin")
    assert ok.ok and ok.size_bytes == 5
    monkeypatch.setattr(fetcher.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=23, stdout="", stderr="curl failed"))
    failed = fetcher._download_binary_curl("https://cdn.example.test/model.bin", target, filename_hint="missing.bin")
    assert not failed.ok and "curl failed" in failed.error

    calls = {"init": 0, "upsert": 0, "downloads": []}
    product = record(source_site="unit", model_download_url=None, model_download_filename=None)
    adapter = types.SimpleNamespace(
        site_name="unit",
        fetch_html=lambda url: ("<html>", url + "/final"),
        parse=lambda url, html, final_url: [product],
    )
    monkeypatch.setattr(fetcher, "find_adapter", lambda url: adapter)
    monkeypatch.setattr(fetcher, "init_db", lambda db_path: calls.__setitem__("init", calls["init"] + 1))
    monkeypatch.setattr(fetcher, "upsert_product", lambda db_path, rec: calls.__setitem__("upsert", calls["upsert"] + 1))
    monkeypatch.setattr(fetcher, "insert_download", lambda **kwargs: calls["downloads"].append(kwargs))
    monkeypatch.setattr(fetcher, "enrich_model_download", lambda rec: rec)

    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_product_and_model", "--url", "https://example.test/item", "--db", str(tmp_path / "db.sqlite"), "--out-dir", str(tmp_path / "out"), "--no-download-model"],
    )
    fetcher.main()
    assert calls["init"] == 1
    assert calls["upsert"] == 1
    assert not calls["downloads"]
    assert "metadata_json:" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_product_and_model", "--url", "https://example.test/item", "--db", str(tmp_path / "db.sqlite"), "--out-dir", str(tmp_path / "out")],
    )
    fetcher.main()
    assert calls["downloads"][-1]["ok"] is False
    assert calls["downloads"][-1]["error"] == "model_download_url is empty"
    assert "model: no downloadable url" in capsys.readouterr().out

    product.model_download_url = "https://cdn.example.test/model.glb"
    product.model_download_filename = "model.glb"
    monkeypatch.setattr(
        fetcher,
        "download_binary_for_record",
        lambda rec, model_dir, filename_hint=None: DownloadResult(
            final_url=rec.model_download_url,
            local_path=str(model_dir / "model.glb"),
            filename=filename_hint,
            content_type="model/gltf-binary",
            ok=True,
            size_bytes=10,
        ),
    )
    fetcher.main()
    assert calls["downloads"][-1]["ok"] is True
    assert "model_saved:" in capsys.readouterr().out

    monkeypatch.setattr(
        fetcher,
        "download_binary_for_record",
        lambda rec, model_dir, filename_hint=None: DownloadResult(
            final_url=None,
            local_path=None,
            filename=filename_hint,
            content_type=None,
            ok=False,
            size_bytes=0,
            error="unit failure",
        ),
    )
    fetcher.main()
    assert calls["downloads"][-1]["ok"] is False
    assert calls["downloads"][-1]["error"] == "unit failure"
    assert "model_download_error: unit failure" in capsys.readouterr().out


def test_fetcher_session_cookie_error_and_download_failure_edges(monkeypatch, tmp_path):
    rec = record(extra_json="[]", product_url="")
    fetcher._merge_extra_json(rec, {"patched": True})
    assert json.loads(rec.extra_json) == {"patched": True}
    assert fetcher._slug_from_url(None) is None
    assert fetcher._slug_from_url("https://3ddd.ru/") is None
    assert fetcher._extra_json_value(record(extra_json="[]"), "x") is None

    for name in [
        "SUPPLIER_3DDD_COOKIES",
        "SUPPLIER_3DDD_COOKIE",
        "THREEDDD_COOKIES",
        "THREEDDD_COOKIE",
        "SUPPLIER_3DDD_COOKIES_FILE",
        "SUPPLIER_3DDD_COOKIE_FILE",
        "THREEDDD_COOKIES_FILE",
        "THREEDDD_COOKIE_FILE",
        "SUPPLIER_3DDD_AUTH",
        "SUPPLIER_3DDD_AUTHORIZATION",
        "THREEDDD_AUTH",
        "THREEDDD_AUTHORIZATION",
        "SUPPLIER_3DDD_AUTH_FILE",
        "SUPPLIER_3DDD_AUTHORIZATION_FILE",
        "THREEDDD_AUTH_FILE",
        "THREEDDD_AUTHORIZATION_FILE",
        "SUPPLIER_3DDD_REFRESH",
        "SUPPLIER_3DDD_REFRESH_TOKEN",
        "THREEDDD_REFRESH",
        "THREEDDD_REFRESH_TOKEN",
        "SUPPLIER_3DDD_REFRESH_FILE",
        "SUPPLIER_3DDD_REFRESH_TOKEN_FILE",
        "THREEDDD_REFRESH_FILE",
        "THREEDDD_REFRESH_TOKEN_FILE",
    ]:
        monkeypatch.delenv(name, raising=False)
    assert fetcher._cookie_sources_for_3ddd()[1].endswith("cookie.txt")
    assert fetcher._auth_sources_for_3ddd()[1].endswith("auth.txt")
    assert fetcher._refresh_sources_for_3ddd()[1].endswith("refresh.txt")

    fallback_file = tmp_path / "fallback.txt"
    fallback_file.write_text("from-file", encoding="utf-8")
    monkeypatch.setenv("VALUE_FILE", str(fallback_file))
    assert fetcher._load_env_text(["MISSING_VALUE"], ["VALUE_FILE"]) == "from-file"
    assert fetcher._normalize_bearer_token("") is None
    fetcher._persist_text_if_path(None, "x")
    assert fetcher._jwt_expiration_ts(None) is None
    bad_payload = "Bearer aaa." + base64.urlsafe_b64encode(b"not-json").decode().rstrip("=") + ".zzz"
    assert fetcher._jwt_expiration_ts(bad_payload) is None
    no_exp = "Bearer aaa." + base64.urlsafe_b64encode(json.dumps({"exp": "bad"}).encode()).decode().rstrip("=") + ".zzz"
    assert fetcher._jwt_expiration_ts(no_exp) is None
    assert fetcher._auth_token_is_expired(None)

    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        "# Netscape HTTP Cookie File\n"
        ".3ddd.ru\tTRUE\t/\tFALSE\t2145916800\tsid\tabc\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPPLIER_3DDD_COOKIES_FILE", str(cookie_file))
    monkeypatch.setenv("SUPPLIER_3DDD_COOKIES", '{"json_sid": "j"}')
    monkeypatch.setenv("SUPPLIER_3DDD_AUTH_FILE", str(tmp_path / "missing_auth.txt"))
    monkeypatch.setenv("SUPPLIER_3DDD_REFRESH_FILE", str(tmp_path / "missing_refresh.txt"))
    session, notes = fetcher._load_3ddd_session(allow_bootstrap=False)
    assert "3ddd_cookie_file:mozilla_cookiejar" in notes
    assert "3ddd_cookie_header:json" in notes
    assert "3ddd_auth_file_missing" in notes
    assert "3ddd_refresh_file_missing" in notes
    assert list(session.cookies)

    expired_auth = tmp_path / "expired_auth.txt"
    expired_auth.write_text(jwt_with_exp(int(time.time()) - 1000), encoding="utf-8")
    refresh = tmp_path / "refresh_token.txt"
    refresh.write_text("refresh", encoding="utf-8")
    monkeypatch.setenv("SUPPLIER_3DDD_COOKIES_FILE", str(cookie_file))
    monkeypatch.setenv("SUPPLIER_3DDD_AUTH_FILE", str(expired_auth))
    monkeypatch.setenv("SUPPLIER_3DDD_REFRESH_FILE", str(refresh))
    missing_token_session = FakeSession([FakeResponse(headers={"Content-Type": "application/json"}, payload={"data": {}})])
    monkeypatch.setattr(fetcher.requests, "Session", lambda: missing_token_session)
    assert "3ddd_auth_refresh_missing_token" in fetcher._load_3ddd_session(allow_bootstrap=False)[1]

    failing_refresh_session = FakeSession([RuntimeError("refresh failed")])
    monkeypatch.setattr(fetcher.requests, "Session", lambda: failing_refresh_session)
    assert any(note.startswith("3ddd_auth_refresh_failed:RuntimeError") for note in fetcher._load_3ddd_session(allow_bootstrap=False)[1])

    monkeypatch.setattr(fetcher.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("curl boom")))
    curl_failed = fetcher._download_binary_curl("https://cdn.example.test/model.bin", tmp_path / "curl-error", filename_hint="model.bin")
    assert not curl_failed.ok and "curl RuntimeError" in curl_failed.error

    anon = FakeResponse(url="https://cdn.example.test/", headers={}, chunks=(b"x",))
    saved = fetcher._save_response_binary(anon, tmp_path / "anon")
    assert saved.filename == "model.bin"

    monkeypatch.setattr(fetcher.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("requests down")))
    monkeypatch.setattr(fetcher, "_download_binary_curl", lambda *args, **kwargs: DownloadResult(None, None, kwargs.get("filename_hint"), None, False, 0, "curl down"))
    failed = fetcher.download_binary("https://cdn.example.test/fail.bin", tmp_path / "failed", filename_hint="fail.bin")
    assert not failed.ok and "curl down" in failed.error

    monkeypatch.setattr(fetcher, "_bootstrap_3ddd_session", lambda reason: ["3ddd_bootstrap_ok"])
    sessions = [
        (FakeSession(), ["expired jwt token"]),
        (FakeSession(), ["reloaded"]),
    ]
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: sessions.pop(0))
    attempts = [
        DownloadResult("https://cdn.example.test/model.rar", None, "model.rar", None, False, 0, "login required"),
        DownloadResult("https://cdn.example.test/model.rar", None, "model.rar", None, False, 0, "still denied"),
    ]
    monkeypatch.setattr(fetcher, "_download_binary_with_session", lambda *args, **kwargs: attempts.pop(0))
    three_fail = fetcher.download_binary_for_record(record(model_download_url="https://cdn.example.test/model.rar"), tmp_path / "three-fail")
    assert not three_fail.ok
    assert "3ddd session download failed" in three_fail.error

    assert fetcher.download_binary_for_record(record(model_download_url=None, external_id="", product_url="", model_page_url="", source_url=""), tmp_path / "no-slug").error == "3ddd slug is empty"

    unsupported_adapter = types.SimpleNamespace(site_name="unit", parse=lambda url, html, final_url: [object()])
    monkeypatch.setattr(fetcher, "find_adapter", lambda url: unsupported_adapter)
    with pytest.raises(TypeError):
        fetcher.parse_url("https://x", "<html>")


def test_fetcher_remaining_3ddd_resolver_session_and_main_edges(monkeypatch, tmp_path, capsys):
    class TextRaisesResponse(FakeResponse):
        @property
        def text(self):
            raise RuntimeError("text failed")

        @text.setter
        def text(self, _value):
            return None

    monkeypatch.setenv("SUPPLIER_3DDD_COOKIES_FILE", str(tmp_path / "missing-cookie.txt"))
    monkeypatch.setenv("SUPPLIER_3DDD_AUTH_FILE", str(tmp_path / "missing-auth.txt"))
    monkeypatch.setenv("SUPPLIER_3DDD_REFRESH_FILE", str(tmp_path / "missing-refresh.txt"))
    monkeypatch.setenv("SUPPLIER_3DDD_COOKIES", "{bad-json")
    session_1 = FakeSession()
    session_2 = FakeSession()
    sessions = [session_1, session_2]
    monkeypatch.setattr(fetcher.requests, "Session", lambda: sessions.pop(0))
    monkeypatch.setattr(fetcher, "_bootstrap_3ddd_session", lambda reason: ["3ddd_bootstrap_ok", reason])
    loaded_session, notes = fetcher._load_3ddd_session(allow_bootstrap=True)
    assert loaded_session is session_2
    assert "3ddd_cookie_file_missing" in notes
    assert "3ddd_bootstrap_ok" in notes

    inserted_session = FakeSession(
        [
            RuntimeError("temporary outage"),
            TextRaisesResponse(status_code=500),
            FakeResponse(headers={"Content-Type": "application/json"}, payload=ValueError("bad json")),
            FakeResponse(
                url="https://cdn.example.test/redirected.zip",
                headers={"Content-Type": "text/plain"},
                history=[object()],
            ),
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (inserted_session, []))
    direct_url, filename, notes = fetcher.resolve_3ddd_download(
        record(external_id="external-id", product_url="https://3ddd.ru/3dmodels/show/model_slug")
    )
    assert direct_url == "https://cdn.example.test/redirected.zip"
    assert filename == "redirected.zip"
    assert any("3ddd_download_attempt_failed:post_button_slug" in note for note in notes)
    assert any("3ddd_download_json_parse_failed:" in note for note in notes)

    no_url_session = FakeSession(
        [
            FakeResponse(
                headers={"Content-Type": "application/json"},
                payload={"data": {"state": "disabled", "disabled": True}},
            ),
            FakeResponse(headers={"Content-Type": "text/html"}, text="login required"),
            FakeResponse(headers={"Content-Type": "text/html"}, text="login required"),
            FakeResponse(headers={"Content-Type": "text/html"}, text="login required"),
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (no_url_session, ["loaded"]))
    direct_url, filename, notes = fetcher.resolve_3ddd_download(record(external_id=""))
    assert direct_url is None and filename is None
    assert "3ddd_button_state:disabled" in notes
    assert "3ddd_button_disabled:True" in notes
    assert any("3ddd_download_requires_auth" in note for note in notes)

    ok_download = DownloadResult("https://cdn.example.test/model.rar", str(tmp_path / "model.rar"), "model.rar", None, True, 3)
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (FakeSession(), ["loaded"]))
    monkeypatch.setattr(fetcher, "_download_binary_with_session", lambda *args, **kwargs: ok_download)
    assert fetcher.download_binary_for_record(record(model_download_url="https://cdn.example.test/model.rar"), tmp_path / "prefilled").ok

    monkeypatch.setattr(fetcher, "enrich_model_download", lambda rec: rec)
    json_session = FakeSession(
        [
            FakeResponse(
                headers={"Content-Type": "application/json"},
                payload={"data": {"url": "https://cdn.example.test/from-json.zip"}},
            )
        ]
    )
    reloaded_session = FakeSession()
    sessions = [(json_session, ["login"]), (reloaded_session, ["reloaded"])]
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: sessions.pop(0))
    monkeypatch.setattr(fetcher, "_should_retry_3ddd_after_bootstrap", lambda error, notes: True)
    monkeypatch.setattr(fetcher, "_bootstrap_3ddd_session", lambda reason: ["3ddd_bootstrap_ok", reason])
    attempts = [
        DownloadResult(None, None, "from-json.zip", None, False, 0, "login required"),
        DownloadResult("https://cdn.example.test/from-json.zip", str(tmp_path / "from-json.zip"), "from-json.zip", None, True, 4),
    ]
    monkeypatch.setattr(fetcher, "_download_binary_with_session", lambda *args, **kwargs: attempts.pop(0))
    retried = fetcher.download_binary_for_record(record(model_download_url=None, external_id="external-id"), tmp_path / "retry-json")
    assert retried.ok

    unresolved_session = FakeSession(
        [
            FakeResponse(headers={"Content-Type": "application/json"}, payload={"data": {"state": "free", "disabled": False}}),
            RuntimeError("post id failed"),
            FakeResponse(status_code=500, text="server down"),
            FakeResponse(headers={"Content-Type": "application/json"}, payload={"data": {}}),
            TextRaisesResponse(headers={"Content-Type": "text/html"}, url="https://3ddd.ru/login"),
        ]
    )
    monkeypatch.setattr(fetcher, "_load_3ddd_session", lambda *args, **kwargs: (unresolved_session, []))
    unresolved = fetcher.download_binary_for_record(record(model_download_url=None, external_id="external-id"), tmp_path / "unresolved-more")
    assert not unresolved.ok
    assert "3ddd download unresolved" in unresolved.error

    class EmptyAdapter:
        site_name = "empty"
        empty_parse_is_skip = True

        def fetch_html(self, url):
            return "<html>", url

        def parse(self, url, html, final_url):
            return []

    monkeypatch.setattr(fetcher, "init_db", lambda _path: None)
    monkeypatch.setattr(fetcher, "find_adapter", lambda _url: EmptyAdapter())
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_product_and_model", "--url", "https://example.test/empty", "--db", str(tmp_path / "db.sqlite"), "--out-dir", str(tmp_path / "out")],
    )
    fetcher.main()
    assert "skipped_empty_result" in capsys.readouterr().out

    class BadEmptyAdapter(EmptyAdapter):
        empty_parse_is_skip = False

    monkeypatch.setattr(fetcher, "find_adapter", lambda _url: BadEmptyAdapter())
    with pytest.raises(ValueError, match="не вернул"):
        fetcher.main()

    class ObjectAdapter(EmptyAdapter):
        empty_parse_is_skip = False

        def parse(self, url, html, final_url):
            return [object()]

    monkeypatch.setattr(fetcher, "find_adapter", lambda _url: ObjectAdapter())
    with pytest.raises(TypeError):
        fetcher.main()
