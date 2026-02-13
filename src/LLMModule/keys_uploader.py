import re
from collections import OrderedDict

def main():
    # 1. Пути к файлам с дефолтами
    default_keys = "src/LLMModule/file.keys"
    default_env  = ".env"

    raw_keys = input(f"Путь к файлу с ключами [по умолчанию: {default_keys}]: ").strip()
    keys_path = raw_keys if raw_keys else default_keys

    raw_env = input(f"Путь к существующему .env [по умолчанию: {default_env}]: ").strip()
    env_path = raw_env if raw_env else default_env

    # 2. Считать и отфильтровать ключи
    pattern = re.compile(r"^sk-or-v1-[0-9a-f]+$")
    unique_keys = OrderedDict()
    with open(keys_path, "r", encoding="utf-8") as fk:
        for line in fk:
            key = line.strip()
            if pattern.match(key):
                unique_keys.setdefault(key, None)

    # 3. Найти максимальный индекс уже в .env
    max_index = 0
    idx_pattern = re.compile(r"^ivangrigin_OPENROUTER_API_KEY_(\d+)\s*=")
    with open(env_path, "r", encoding="utf-8") as fe:
        for line in fe:
            m = idx_pattern.match(line)
            if m:
                i = int(m.group(1))
                if i > max_index:
                    max_index = i

    # 4. Дописать новые ключи
    with open(env_path, "a", encoding="utf-8") as fe:
        for key in unique_keys:
            max_index += 1
            fe.write(f"ivangrigin_OPENROUTER_API_KEY_{max_index} = {key}\n")

    print(f"Добавлено {max_index} новых ключей в {env_path}")

if __name__ == "__main__":
    main()
