## Использование вируального окружения
### Создать виртуальное окружение
python3 -m venv .venv
### Запустить виртуальное окружение
source .venv/bin/activate
### Загрузить рабочие библиотеки
pip install -r requirements.txt
### Добавить зависимости
pip install some_libs
pip freeze > requirements.txt
