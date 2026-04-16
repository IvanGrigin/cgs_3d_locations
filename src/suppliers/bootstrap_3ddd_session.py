# -*- coding: utf-8 -*-
"""
This script bootstraps an authenticated 3ddd browser session on disk.
It uses Selenium to log in once and save tokens plus cookies locally.
Those session artifacts are reused by automated model download flows.
The script is operational and intentionally explicit about browser behavior.
Keep the saved session format compatible with download tooling.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


DEFAULT_OUT_DIR = Path("data/sourse/suppliers/3ddd_session")


def _prompt_if_missing(value: str | None, prompt: str, secret: bool = False) -> str:
    if value:
        return value
    if secret:
        return getpass.getpass(prompt)
    return input(prompt).strip()


def _build_driver(browser: str) -> webdriver.Remote:
    browser = browser.strip().lower()
    if browser == "safari":
        return webdriver.Safari()
    raise ValueError(f"Unsupported browser: {browser}")


def _find_submit_button(driver: webdriver.Remote) -> object | None:
    candidates = driver.find_elements(By.CSS_SELECTOR, "button, input[type='submit']")
    for element in candidates:
        try:
            text = (element.text or "").strip().lower()
            value = (element.get_attribute("value") or "").strip().lower()
            combined = f"{text} {value}"
            if not element.is_displayed() or not element.is_enabled():
                continue
            if any(token in combined for token in ("войти", "sign in", "login", "log in")):
                return element
        except Exception:
            continue
    for element in candidates:
        try:
            if element.is_displayed() and element.is_enabled():
                return element
        except Exception:
            continue
    return None


def _cookie_header(driver: webdriver.Remote) -> str:
    pairs: list[str] = []
    seen: set[str] = set()
    for cookie in driver.get_cookies():
        domain = str(cookie.get("domain") or "")
        if "3ddd.ru" not in domain:
            continue
        name = str(cookie.get("name") or "").strip()
        value = str(cookie.get("value") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def _save_session(out_dir: Path, auth: str, refresh: str, cookie_header: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "auth.txt").write_text(f"Bearer {auth.strip()}\n", encoding="utf-8")
    (out_dir / "refresh.txt").write_text(refresh.strip() + "\n", encoding="utf-8")
    (out_dir / "cookie.txt").write_text(cookie_header.strip() + "\n", encoding="utf-8")
    meta = {
        "saved_at_epoch": int(time.time()),
        "has_auth": bool(auth),
        "has_refresh": bool(refresh),
        "cookie_count": len([p for p in cookie_header.split(";") if "=" in p]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=os.environ.get("THREEDDD_EMAIL") or os.environ.get("SUPPLIER_3DDD_EMAIL"))
    ap.add_argument("--password", default=os.environ.get("THREEDDD_PASSWORD") or os.environ.get("SUPPLIER_3DDD_PASSWORD"))
    ap.add_argument("--browser", default="safari")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--timeout-sec", type=int, default=90)
    ap.add_argument("--login-url", default="https://3ddd.ru/auth/login?referer_url=/")
    args = ap.parse_args()

    email = _prompt_if_missing(args.email, "3ddd email: ")
    password = _prompt_if_missing(args.password, "3ddd password: ", secret=True)
    out_dir = Path(args.out_dir).expanduser().resolve()

    driver = _build_driver(args.browser)
    wait = WebDriverWait(driver, args.timeout_sec)
    driver.set_window_size(1440, 1100)

    try:
        driver.get(args.login_url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        auth_token = driver.execute_script("return window.localStorage.getItem('skyAuthToken');")
        refresh_token = driver.execute_script("return window.localStorage.getItem('skyRefreshToken');")
        if not (auth_token and refresh_token):
            email_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']")))
            password_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))

            email_input.clear()
            email_input.send_keys(email)
            password_input.clear()
            password_input.send_keys(password)

            button = _find_submit_button(driver)
            if button is not None:
                button.click()
            else:
                password_input.send_keys(Keys.ENTER)

        def _tokens_ready(_: webdriver.Remote) -> bool:
            auth = driver.execute_script("return window.localStorage.getItem('skyAuthToken');")
            refresh = driver.execute_script("return window.localStorage.getItem('skyRefreshToken');")
            return bool(auth and refresh)

        wait.until(_tokens_ready)
        auth_token = driver.execute_script("return window.localStorage.getItem('skyAuthToken');")
        refresh_token = driver.execute_script("return window.localStorage.getItem('skyRefreshToken');")
        cookie_header = _cookie_header(driver)

        if not auth_token or not refresh_token:
            raise RuntimeError("3ddd tokens not found in localStorage after login")
        if not cookie_header:
            raise RuntimeError("3ddd cookies not found after login")

        _save_session(out_dir, auth_token, refresh_token, cookie_header)

        print(f"saved_auth: {out_dir / 'auth.txt'}")
        print(f"saved_refresh: {out_dir / 'refresh.txt'}")
        print(f"saved_cookie: {out_dir / 'cookie.txt'}")
        print(f"saved_meta: {out_dir / 'meta.json'}")
        return 0
    except TimeoutException:
        print("Timed out waiting for 3ddd login to complete. This usually means wrong credentials, 2FA, or Safari remote automation is disabled.", file=sys.stderr)
        return 2
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())
