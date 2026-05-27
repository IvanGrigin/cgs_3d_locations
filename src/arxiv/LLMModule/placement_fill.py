# src/LLMModule/placement_fill.py
"""
python -m src.LLMModule.placement_fill \
  --input data/input/test_for_llm.json \
  --output data/output/test_for_llm_filled.json
"""
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

from openai import OpenAI
from .keys_manager import KeyManager


JsonDict = Dict[str, Any]


# ---------------------------
# Logger
# ---------------------------

class _Logger:
    def info(self, msg: str) -> None:
        print(msg)

    def warning(self, msg: str) -> None:
        print("WARNING:", msg)

    def error(self, msg: str) -> None:
        print("ERROR:", msg)


logger = _Logger()


# ---------------------------
# Geometry
# ---------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (
        isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    )


def point_in_polygon(x: float, z: float, polygon: List[Dict[str, float]]) -> bool:
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]["x"], polygon[i]["z"]
        xj, zj = polygon[j]["x"], polygon[j]["z"]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def aabb_corners(cx: float, cz: float, sx: float, sz: float) -> List[Tuple[float, float]]:
    hx = sx / 2.0
    hz = sz / 2.0
    return [
        (cx - hx, cz - hz),
        (cx - hx, cz + hz),
        (cx + hx, cz - hz),
        (cx + hx, cz + hz),
    ]


def aabb_inside_polygon(cx: float, cz: float, sx: float, sz: float, poly: List[Dict[str, float]]) -> bool:
    for x, z in aabb_corners(cx, cz, sx, sz):
        if not point_in_polygon(x, z, poly):
            return False
    return True


# ---------------------------
# JSON input stabilizer (for "x:" newline "z:" etc.)
# ---------------------------

# 1) "x":   ,  OR  "x":\n   }  -> insert null
_EMPTY_VALUE_RE = re.compile(r'("(?P<k>x|z)"\s*:\s*)(?=(,|\}|\]|\n|\r))')
# 2) "x":\n null  -> "x": null
_SPLIT_NULL_RE = re.compile(r'("(?P<k>x|z)"\s*:\s*)\n\s*null\b')
# 3) Missing comma between x and z:  "x": null "z":
_MISSING_COMMA_XZ_RE = re.compile(r'("x"\s*:\s*(?:null|-?\d+(?:\.\d+)?))(\s*"z"\s*:)')
# 4) Rare: colon then immediate next key:  : "z":  -> : null, "z":
_KEY_AFTER_COLON_RE = re.compile(r'(:\s*)"([a-zA-Z_]+)"\s*:')


def stabilize_json(raw: str) -> str:
    s = raw
    s = _EMPTY_VALUE_RE.sub(r"\1null", s)
    s = _SPLIT_NULL_RE.sub(r"\1null", s)
    s = _MISSING_COMMA_XZ_RE.sub(r"\1,\2", s)
    s = _KEY_AFTER_COLON_RE.sub(r'\1null, "\2":', s)
    return s

def stabilize_json_aggressive(raw: str) -> str:
    """
    Агрессивный ремонт JSON:
    - обрезает всё до первого '{'
    - обрезает мусор после последнего '}' если он есть
    - балансирует скобки {}, []
    - удаляет лишние закрывающие скобки в конце
    Важно: не пытается «угадать» пропущенные запятые (это делает stabilize_json).
    """
    s = raw.strip()
    if not s:
        return s

    # Обрезаем префикс до первого объекта
    first = s.find("{")
    if first != -1:
        s = s[first:]

    # Если есть явный хвост после последней }, попробуем обрезать
    last_obj_end = s.rfind("}")
    if last_obj_end != -1 and last_obj_end + 1 < len(s):
        # но только если после } нет ничего похожего на продолжение JSON
        tail = s[last_obj_end + 1:].strip()
        if tail:
            s = s[: last_obj_end + 1]

    # Балансировка: считаем, сколько не закрыто
    open_curly = 0
    open_square = 0
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            open_curly += 1
        elif ch == "}":
            # лишние закрывающие просто игнорируем на этапе подсчёта
            if open_curly > 0:
                open_curly -= 1
        elif ch == "[":
            open_square += 1
        elif ch == "]":
            if open_square > 0:
                open_square -= 1

    # Если строка закончилась внутри строки — закрывать кавычки НЕ пытаемся (слишком опасно)
    # Дозакрываем только контейнеры.
    if open_square > 0:
        s += "]" * open_square
    if open_curly > 0:
        s += "}" * open_curly

    # Иногда модель/файл добавляют лишние закрывающие в самом конце: '}}]]'
    # Удалим лишнее с конца, пока json.loads не перестанет ругаться "Extra data"/лишняя скобка.
    s2 = s
    for _ in range(32):
        try:
            json.loads(s2)
            return s2
        except json.JSONDecodeError as e:
            # Если ошибка про лишние данные, попробуем обрезать хвост
            msg = (e.msg or "").lower()
            if "extra data" in msg and e.pos > 0:
                s2 = s2[: e.pos].rstrip()
                continue

            # Частый кейс: "Expecting ',' delimiter" в самом конце из-за лишней скобки
            # Попробуем убрать один символ с конца, если он закрывающий.
            if s2 and s2[-1] in ("]", "}"):
                s2 = s2[:-1].rstrip()
                continue

            break

    return s2


def load_json_lenient(path: str) -> JsonDict:
    raw = open(path, "r", encoding="utf-8").read()

    # 1) как есть
    try:
        obj = json.loads(raw)
    except Exception:
        # 2) стабилизация локальных поломок (ваш текущий стабилизатор)
        logger.warning("Битый JSON → стабилизация")
        fixed = stabilize_json(raw)

        # 3) попытка «дозакрыть» скобки и вырезать мусор до/после JSON
        fixed2 = stabilize_json_aggressive(fixed)

        try:
            obj = json.loads(fixed2)
        except json.JSONDecodeError as e:
            start = max(0, e.pos - 160)
            end = min(len(fixed2), e.pos + 160)
            frag = fixed2[start:end].replace("\n", "\\n")
            raise ValueError(f"JSON не восстановлен: {e.msg} pos={e.pos}\nФрагмент:\n{frag}")

    if not isinstance(obj, dict):
        raise ValueError("Входной JSON должен быть объектом верхнего уровня (dict).")
    return obj


def sanitize_scene_in(scene: JsonDict) -> JsonDict:
    # Гарантируем, что все pos.x/pos.z либо число, либо None
    out = json.loads(json.dumps(scene))
    rooms = out.get("rooms")
    if not isinstance(rooms, list):
        return out

    for room in rooms:
        if not isinstance(room, dict):
            continue
        objs = room.get("objects")
        if not isinstance(objs, list):
            continue
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            pos = obj.get("pos")
            if not isinstance(pos, dict):
                pos = {}
                obj["pos"] = pos
            x = pos.get("x", None)
            z = pos.get("z", None)
            pos["x"] = x if _is_number(x) else None
            pos["z"] = z if _is_number(z) else None
    return out


# ---------------------------
# JSON extraction (balanced braces, supports ```json ... ``` too)
# ---------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _extract_first_json_object(text: str) -> str:
    if not text:
        raise ValueError("Пустой ответ модели")

    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("JSON не найден в ответе модели")

    # Пытаемся найти сбалансированный объект. Если не получилось — вернём "до конца",
    # дальше его починит stabilize_json_aggressive.
    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        else:
            if c == '"':
                in_str = True
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1].strip()

    # обрезанный JSON: вернём всё до конца
    return text[start:].strip()


def extract_json(text: str) -> JsonDict:
    s = _extract_first_json_object(text)

    # локальная стабилизация (ваши правила)
    s1 = stabilize_json(s)

    # агрессивный ремонт скобок/хвостов
    s2 = stabilize_json_aggressive(s1)

    obj = json.loads(s2)
    if not isinstance(obj, dict):
        raise ValueError("Ожидался JSON-объект верхнего уровня (dict).")
    return obj



# ---------------------------
# Contract checks / merge
# ---------------------------

def is_same_structure(original: JsonDict, candidate: JsonDict) -> bool:
    if not isinstance(candidate, dict):
        return False

    # candidate должен содержать минимум ключи original (лишние допускаем)
    for k in original.keys():
        if k not in candidate:
            return False

    or_rooms = original.get("rooms", [])
    ca_rooms = candidate.get("rooms", [])
    if not isinstance(or_rooms, list) or not isinstance(ca_rooms, list):
        return False
    if len(or_rooms) != len(ca_rooms):
        return False

    for r1, r2 in zip(or_rooms, ca_rooms):
        if not isinstance(r1, dict) or not isinstance(r2, dict):
            return False
        o1 = r1.get("objects", [])
        o2 = r2.get("objects", [])
        if not isinstance(o1, list) or not isinstance(o2, list):
            return False
        if len(o1) != len(o2):
            return False

    return True


def merge_positions(original: JsonDict, partial: JsonDict) -> JsonDict:
    """
    Мержит найденные pos.x/pos.z в исходную сцену.
    Стратегии:
    A) partial как сцена с rooms/objects
    B) partial содержит список объектов в любом месте (flat)
    C) partial содержит только наборы pos (вытащим по порядку)
    """
    merged = json.loads(json.dumps(original))

    def assign_by_index(pairs: List[Tuple[Optional[float], Optional[float]]]) -> None:
        if not pairs:
            return
        rooms = merged.get("rooms")
        if not isinstance(rooms, list):
            return
        k = 0
        for r in rooms:
            if not isinstance(r, dict):
                continue
            objs = r.get("objects")
            if not isinstance(objs, list):
                continue
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                if k >= len(pairs):
                    return
                x, z = pairs[k]
                k += 1
                pos = obj.setdefault("pos", {})
                if _is_number(x):
                    pos["x"] = float(x)  # type: ignore[arg-type]
                if _is_number(z):
                    pos["z"] = float(z)  # type: ignore[arg-type]

    # --- A) ожидаемая структура rooms/objects ---
    rooms_p = partial.get("rooms")
    if isinstance(rooms_p, list):
        any_set = False
        for r_i, room_p in enumerate(rooms_p):
            if not isinstance(room_p, dict):
                continue
            objs_p = room_p.get("objects")
            if not isinstance(objs_p, list):
                continue
            for o_i, obj_p in enumerate(objs_p):
                if not isinstance(obj_p, dict):
                    continue
                pos_p = obj_p.get("pos")
                if not isinstance(pos_p, dict):
                    continue
                x = pos_p.get("x")
                z = pos_p.get("z")
                if r_i < len(merged.get("rooms", [])) and o_i < len(merged["rooms"][r_i].get("objects", [])):
                    pos_m = merged["rooms"][r_i]["objects"][o_i].setdefault("pos", {})
                    if _is_number(x):
                        pos_m["x"] = float(x)
                        any_set = True
                    if _is_number(z):
                        pos_m["z"] = float(z)
                        any_set = True
        if any_set:
            return merged

    # --- B) попробуем собрать "flat list" объектов из partial ---
    def collect_all_objects(node: Any, out: List[Dict[str, Any]]) -> None:
        if isinstance(node, dict):
            # если это похоже на объект сцены
            if "pos" in node and "size" in node:
                out.append(node)
            for v in node.values():
                collect_all_objects(v, out)
        elif isinstance(node, list):
            for it in node:
                collect_all_objects(it, out)

    flat_objs: List[Dict[str, Any]] = []
    collect_all_objects(partial, flat_objs)

    pairs: List[Tuple[Optional[float], Optional[float]]] = []
    for obj_p in flat_objs:
        pos_p = obj_p.get("pos")
        if not isinstance(pos_p, dict):
            continue
        x = pos_p.get("x")
        z = pos_p.get("z")
        fx = float(x) if _is_number(x) else None
        fz = float(z) if _is_number(z) else None
        # берём даже частичные, чтобы хоть что-то записать
        pairs.append((fx, fz))

    if pairs:
        assign_by_index(pairs)
        return merged

    # --- C) если вообще нет объектов, но есть "pos" где-то ---
    def collect_pos_pairs(node: Any, out: List[Tuple[Optional[float], Optional[float]]]) -> None:
        if isinstance(node, dict):
            if "pos" in node and isinstance(node["pos"], dict):
                p = node["pos"]
                x = p.get("x")
                z = p.get("z")
                out.append((float(x) if _is_number(x) else None, float(z) if _is_number(z) else None))
            for v in node.values():
                collect_pos_pairs(v, out)
        elif isinstance(node, list):
            for it in node:
                collect_pos_pairs(it, out)

    pairs2: List[Tuple[Optional[float], Optional[float]]] = []
    collect_pos_pairs(partial, pairs2)
    if pairs2:
        assign_by_index(pairs2)

    return merged


# ---------------------------
# Validation
# ---------------------------

def find_null_positions(scene: JsonDict) -> List[str]:
    errors: List[str] = []
    rooms = scene.get("rooms", [])
    if not isinstance(rooms, list):
        return ["rooms отсутствует или не список"]

    for r_i, room in enumerate(rooms):
        if not isinstance(room, dict):
            errors.append(f"rooms[{r_i}] не dict")
            continue
        objs = room.get("objects", [])
        if not isinstance(objs, list):
            errors.append(f"rooms[{r_i}].objects invalid")
            continue
        for o_i, obj in enumerate(objs):
            if not isinstance(obj, dict):
                errors.append(f"rooms[{r_i}].objects[{o_i}] не dict")
                continue
            pos = obj.get("pos")
            if not isinstance(pos, dict):
                errors.append(f"rooms[{r_i}].objects[{o_i}].pos invalid")
                continue
            x = pos.get("x")
            z = pos.get("z")
            if x is None or z is None:
                errors.append(f"rooms[{r_i}].objects[{o_i}] null pos")
            elif not _is_number(x) or not _is_number(z):
                errors.append(f"rooms[{r_i}].objects[{o_i}] invalid pos")
    return errors


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def validate_scene(scene: JsonDict) -> ValidationResult:
    errors: List[str] = []

    # 0) В первую очередь: все null должны быть заполнены (и вообще числа)
    null_errs = find_null_positions(scene)
    if null_errs:
        return ValidationResult(False, null_errs)

    rooms = scene.get("rooms", [])
    if not isinstance(rooms, list) or not rooms:
        return ValidationResult(False, ["rooms отсутствует или не список"])

    for r_i, room in enumerate(rooms):
        if not isinstance(room, dict):
            errors.append(f"rooms[{r_i}] не dict")
            continue

        poly = room.get("floor_polygon_xz")
        if not isinstance(poly, list) or len(poly) < 3:
            errors.append(f"rooms[{r_i}].floor_polygon_xz invalid")
            continue

        objs = room.get("objects", [])
        if not isinstance(objs, list) or not objs:
            errors.append(f"rooms[{r_i}].objects invalid")
            continue

        for o_i, obj in enumerate(objs):
            if not isinstance(obj, dict):
                errors.append(f"rooms[{r_i}].objects[{o_i}] не dict")
                continue

            pos = obj.get("pos")
            size = obj.get("size")

            if not isinstance(pos, dict):
                errors.append(f"rooms[{r_i}].objects[{o_i}].pos invalid")
                continue

            x = pos.get("x")
            z = pos.get("z")

            # тут уже гарантировано числами (find_null_positions), но оставим защиту
            if not _is_number(x) or not _is_number(z):
                errors.append(f"rooms[{r_i}].objects[{o_i}] invalid pos")
                continue

            if not isinstance(size, list) or len(size) != 2 or not _is_number(size[0]) or not _is_number(size[1]):
                errors.append(f"rooms[{r_i}].objects[{o_i}] invalid size")
                continue

            if not aabb_inside_polygon(float(x), float(z), float(size[0]), float(size[1]), poly):
                errors.append(f"rooms[{r_i}].objects[{o_i}] outside polygon")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


# ---------------------------
# Probe Integration
# ---------------------------

def load_models_from_probe(path: str) -> List[str]:
    if not os.path.exists(path):
        logger.warning("probe file not found")
        return []

    data = json.load(open(path, "r", encoding="utf-8"))

    alive: List[Tuple[str, float]] = []
    for r in data.get("results", []):
        if r.get("ok") and r.get("status") == "ok":
            mid = r.get("model")
            lat = r.get("latency_s", 1e9)
            if isinstance(mid, str) and isinstance(lat, (int, float)):
                alive.append((mid, float(lat)))

    alive.sort(key=lambda x: x[1])
    models = [m for m, _ in alive]
    logger.info(f"Alive models: {len(models)}")
    return models


# ---------------------------
# Prompting
# ---------------------------

def build_prompt(scene: JsonDict) -> str:
    rules = (
        "You are an interior layout engine. Fill object coordinates.\n\n"
        "OUTPUT (STRICT)\n"
        "- Return EXACTLY ONE JSON object.\n"
        "- No markdown, no comments, no explanations.\n"
        "- Keep JSON structure IDENTICAL.\n"
        "- You may modify ONLY: pos.x and pos.z.\n"
        "- pos.x and pos.z must be numbers (null forbidden).\n\n"
        "CONSTRAINTS\n"
        "1) Every object footprint (axis-aligned rectangle size[0] x size[1] in XZ) must be fully inside floor_polygon_xz.\n"
        "2) No object-object overlaps (even partial).\n"
        "3) Keep a margin from walls >= 0.10 and between objects >= 0.10 (prefer 0.20–0.40 if space allows).\n"
        "4) Human-like layout: large items near walls; group TV+sofa; dining table+chairs; small items near related furniture; keep plausible walkways.\n\n"
        "Return ONLY the JSON.\n\n"
        "Scene:\n"
    )

    return rules + json.dumps(scene, ensure_ascii=False, indent=2) + "\n"

# ---------------------------
# Error Classifier
# ---------------------------

def classify_error(e: Exception) -> str:
    s = str(e)
    sl = s.lower()

    if "error code: 402" in sl or "insufficient credits" in sl or "402" in sl:
        return "402"
    if "error code: 404" in sl or "no endpoints found" in sl or "404" in sl:
        return "404"
    if "error code: 401" in sl or " 401" in sl:
        return "401"
    if "error code: 429" in sl or "rate limit" in sl or "429" in sl:
        return "rl"
    if "timeout" in sl or "timed out" in sl or "read timeout" in sl or "connect timeout" in sl:
        return "timeout"
    return "other"


# ---------------------------
# LLM Runner
# ---------------------------

class PlacementLLM:
    def __init__(
        self,
        models: List[str],
        timeout_s: float,
        env_prefix: str = "ivangrigin_OPENROUTER_API_KEY_",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int = 5000,
        sleep_between_models_s: float = 0.2,
    ):
        if load_dotenv:
            load_dotenv()

        self.key_manager = KeyManager.from_env_prefix(env_prefix)
        self.base_url = base_url
        self.models = [m for m in models if isinstance(m, str) and m.strip()]
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.sleep_between_models_s = float(sleep_between_models_s)

        self._model_idx = 0

    def _next_model(self) -> str:
        m = self.models[self._model_idx % len(self.models)]
        self._model_idx += 1 * 0
        return m

    def _call_model(self, model: str, prompt: str) -> str:
        """
        На одной модели перебираем ключи только для 401/429.
        Для 402/404/timeout считаем, что ключ не виноват -> пусть пайплайн сменит модель.
        """
        max_key_tries = len(self.key_manager._keys)
        last_exc: Optional[Exception] = None

        for _ in range(max_key_tries):
            key = self.key_manager.get_key()
            client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout_s)

            t0 = time.time()
            try:
                logger.info(f"model={model}")
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                dt = time.time() - t0
                content = resp.choices[0].message.content or ""
                logger.info(f"ok {dt:.2f}s chars={len(content)}")
                return content

            except Exception as e:
                dt = time.time() - t0
                last_exc = e
                status = classify_error(e)
                logger.warning(f"{model} → {status} ({dt:.2f}s)")

                if status in ("401", "rl"):
                    # ключ "плохой/лимитный" -> переключаемся
                    self.key_manager.mark_exhausted(key)
                    time.sleep(0.4)
                    continue

                # 402/404/timeout/other -> ключ не трогаем, пусть вызывающий сменит модель
                raise

        raise RuntimeError(f"All keys exhausted. Last error: {last_exc}")

    def generate(self, prompt: str) -> str:
        last_exc: Optional[Exception] = None

        for _ in range(len(self.models)):
            model = self._next_model()
            try:
                return self._call_model(model, prompt)
            except Exception as e:
                last_exc = e
                time.sleep(self.sleep_between_models_s)
                continue

        raise RuntimeError(f"All models failed. Last error: {last_exc}")


# ---------------------------
# Pipeline
# ---------------------------
def run_pipeline(
    scene: JsonDict,
    models: List[str],
    timeout_s: float,
    attempts: int,
) -> JsonDict:
    llm = PlacementLLM(models=models, timeout_s=timeout_s)

    prompt_base = build_prompt(scene)
    prompt = prompt_base

    for attempt in range(1, attempts + 1):
        logger.info(f"attempt {attempt}/{attempts}")

        answer = llm.generate(prompt)

        # 1) parse
        try:
            candidate = extract_json(answer)
        except Exception:
            logger.warning("JSON parse fail")
            logger.info(answer)
            # усиливаем ремонт и пробуем снова
            prompt = prompt_base + (
                "\nREPAIR:\n"
                "Your previous output was not valid JSON.\n"
                "Return ONLY one JSON object matching the template exactly.\n"
                "Modify ONLY pos.x and pos.z.\n"
            )
            continue

        # 2) если структура другая -> merge
        if not is_same_structure(scene, candidate):
            logger.warning("Model broke JSON contract → attempting merge")
            candidate = merge_positions(scene, candidate)

            if not is_same_structure(scene, candidate):
                # даже после merge структура может быть оригинальной (merge делает копию original),
                # но если original тоже не dict/rooms, лучше продолжить
                logger.warning("Merge did not restore structure")
                prompt = prompt_base + (
                    "\nREPAIR:\n"
                    "You changed the JSON structure. Do NOT change any keys or arrays.\n"
                    "Return the SAME JSON template with only pos.x and pos.z updated.\n"
                )
                continue

        # 2.5) приоритет: null
        null_errs = find_null_positions(candidate)
        if null_errs:
            logger.warning("null/invalid pos remain")
            logger.info("\n".join(null_errs[:3]))

            # сформируем конкретный список индексов объектов, где null
            missing: List[Tuple[int, int]] = []
            rooms = candidate.get("rooms", [])
            if isinstance(rooms, list):
                for r_i, room in enumerate(rooms):
                    if not isinstance(room, dict):
                        continue
                    objs = room.get("objects", [])
                    if not isinstance(objs, list):
                        continue
                    for o_i, obj in enumerate(objs):
                        if not isinstance(obj, dict):
                            continue
                        pos = obj.get("pos")
                        if not isinstance(pos, dict):
                            missing.append((r_i, o_i))
                            continue
                        if pos.get("x") is None or pos.get("z") is None or (not _is_number(pos.get("x"))) or (not _is_number(pos.get("z"))):
                            missing.append((r_i, o_i))

            # repair prompt: требуем заполнить все missing и ничего больше
            prompt = prompt_base + (
                "\nREPAIR:\n"
                "Some pos.x/pos.z are missing or non-numeric.\n"
                "You MUST fill ALL of them with numbers (no null).\n"
                f"Missing indices: {missing}\n"
                "Return ONLY the JSON template. Modify ONLY pos.x and pos.z.\n"
                "Do NOT remove or reorder objects.\n"
            )
            continue

        # 3) validate геометрию
        vr = validate_scene(candidate)
        if vr.ok:
            return candidate

        logger.warning("validation fail")
        # repair подсказка с ошибками
        prompt = prompt_base + (
            "\nREPAIR:\n"
            "Your positions are geometrically invalid (outside polygon or other constraints).\n"
            "You MUST move objects strictly inside the floor polygon.\n"
            "Return ONLY the JSON template. Modify ONLY pos.x and pos.z.\n"
            f"Errors: {vr.errors[:10]}\n"
        )

    raise RuntimeError("No valid layout")


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fill missing furniture positions via LLM (OpenRouter).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe", default="data/output/model_probe_results_free.json")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=20)

    args = parser.parse_args()

    scene_raw = load_json_lenient(args.input)
    scene = sanitize_scene_in(scene_raw)

    models = load_models_from_probe(args.probe)
    if not models:
        logger.warning("Fallback model")
        models = [
            "google/gemma-3-12b-it:free",
        ]

    result = run_pipeline(
        scene=scene,
        models=models,
        timeout_s=float(args.timeout),
        attempts=int(args.attempts),
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
