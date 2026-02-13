# src/LLMModule/probe_models.py

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from .keys_manager import KeyManager


@dataclass
class ProbeResult:
    model: str
    name: str
    ok: bool
    status: str
    latency_s: float
    chars: int
    preview: str


# ---------------------------

def datetime_now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def os_makedirs_for_file(path: str) -> None:
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _is_zero(v: Any) -> bool:
    return v == 0 or v == "0" or v == 0.0 or v == "0.0"


# ---------------------------

def list_models(timeout_s: float = 30.0) -> List[Dict[str, Any]]:
    url = "https://openrouter.ai/api/v1/models"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    items = data.get("data", [])
    return items if isinstance(items, list) else []


def is_free_model(item: Dict[str, Any]) -> bool:
    mid = str(item.get("id", ""))
    pricing = item.get("pricing") or {}
    p_prompt = pricing.get("prompt")
    p_comp = pricing.get("completion")

    return (":free" in mid) or (_is_zero(p_prompt) and _is_zero(p_comp))


# ---------------------------

def classify_error(e: Exception) -> str:
    s = str(e).lower()

    if "402" in s or "insufficient credits" in s:
        return "402"

    if "404" in s:
        return "404"

    if "401" in s:
        return "401"

    if "429" in s or "rate limit" in s:
        return "rl"

    if "timeout" in s:
        return "timeout"

    return "other"


# ---------------------------

def probe_one_model(client: OpenAI, model: str, prompt: str, verbose: bool):

    t0 = time.time()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,
        )

        dt = time.time() - t0
        content = (resp.choices[0].message.content or "").strip()

        if verbose:
            print("\n--- PROMPT ---")
            print(prompt)
            print("\n--- RESPONSE ---")
            print(content if content else "<EMPTY>")
            print()

        if not content:
            return False, "empty", dt, 0, ""

        try:
            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                return False, "not_json_object", dt, len(content), content[:120]

        except Exception:
            return False, "invalid_json", dt, len(content), content[:120]

        preview = content[:120].replace("\n", "\\n")

        return True, "ok", dt, len(content), preview

    except Exception as e:
        dt = time.time() - t0
        status = classify_error(e)

        if verbose:
            print("\n--- PROMPT ---")
            print(prompt)
            print("\n--- ERROR ---")
            print(e)
            print()

        return False, status, dt, 0, ""


# ---------------------------

def main() -> None:

    ap = argparse.ArgumentParser(description="Probe OpenRouter models (JSON stability test).")

    ap.add_argument("--free-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--skip-thinking", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="data/output/model_probe_results.json")
    ap.add_argument("--env-prefix", default="ivangrigin_OPENROUTER_API_KEY_")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")

    args = ap.parse_args()

    km = KeyManager.from_env_prefix(args.env_prefix)

    test_prompt = (
        "Вычисли выражение 1 + 1.\n\n"
        "Верни ответ строго в JSON формате:\n\n"
        "{\n"
        '  "answer": 2\n'
        "}\n\n"
        "Без пояснений."
    )

    items = list_models()

    pairs: List[Tuple[str, str]] = []

    for it in items:
        mid = it.get("id")
        name = it.get("name") or ""

        if not isinstance(mid, str):
            continue

        if args.skip_thinking and "thinking" in mid:
            continue

        if args.free_only and not is_free_model(it):
            continue

        pairs.append((mid, str(name)))

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Models to probe: {len(pairs)}")

    results: List[ProbeResult] = []

    for idx, (mid, name) in enumerate(pairs, start=1):

        print(f"\n[{idx}/{len(pairs)}] probing: {mid}")

        ok = False
        final_status = "other"
        final_dt = 0.0
        final_chars = 0
        final_preview = ""

        for attempt in range(args.retries):

            key = km.get_key()

            client = OpenAI(
                api_key=key,
                base_url=args.base_url,
                timeout=args.timeout,
            )

            ok, st, dt, ch, prev = probe_one_model(
                client,
                mid,
                test_prompt,
                args.verbose,
            )

            final_status = st
            final_dt = dt
            final_chars = ch
            final_preview = prev

            print(f"status={st} latency={dt:.2f}s chars={ch}")

            if ok:
                break

            if st in ("401", "rl"):
                km.mark_exhausted(key)
                time.sleep(0.5)
                continue

            time.sleep(0.2)

        results.append(
            ProbeResult(
                model=mid,
                name=name,
                ok=ok,
                status=final_status,
                latency_s=final_dt,
                chars=final_chars,
                preview=final_preview,
            )
        )

        time.sleep(args.sleep)

    alive = [r for r in results if r.ok]
    alive.sort(key=lambda r: r.latency_s)

    print("\n=== BEST MODELS ===")

    for r in alive[:20]:
        print(f"{r.model}\t{r.latency_s:.2f}s")

    out_obj = {
        "generated_at": datetime_now_iso(),
        "count": len(results),
        "alive": len(alive),
        "results": [r.__dict__ for r in results],
    }

    os_makedirs_for_file(args.out)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {args.out}")

    if alive:
        print("\nBEST MODEL:")
        print(alive[0].model)
    else:
        print("\nNo stable JSON models.")


if __name__ == "__main__":
    main()
