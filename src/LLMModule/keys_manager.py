# Analysis/keys_manager.py
import os

class KeyManager:
    def __init__(self, keys: list[str]):
        self._keys = keys
        # все ключи изначально активны
        self._active = {key: True for key in keys}
        self._current_key: str | None = None

    @classmethod
    def from_env_prefix(cls, prefix: str = 'DEEPSEEK_KEY_') -> 'KeyManager':
        keys = [val for name, val in os.environ.items() if name.startswith(prefix) and val]
        if not keys:
            raise RuntimeError(f"No API keys found with prefix {prefix}")
        return cls(keys)

    def get_key(self) -> str:
        # если текущий ещё не исчерпан — возвращаем его
        if self._current_key and self._active.get(self._current_key, False):
            return self._current_key

        # иначе ищем в списке первый активный
        for key in self._keys:
            if self._active.get(key, False):
                self._current_key = key
                return key

        # ни одного не осталось
        raise RuntimeError("All API keys are exhausted")

    def mark_exhausted(self, key: str):
        if key in self._active:
            self._active[key] = False
            # если это был текущий — сбросим, чтобы следующая get_key взяла новый
            if self._current_key == key:
                self._current_key = None
