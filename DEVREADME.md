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

a01@MacBook-Pro--Ivan ~ % ssh -p 32172 root@1.208.108.242 -L 8080:localhost:8080


python3 src/run_pipeline.py кровать шкаф кресло стол стул \
  --placer cube \
  --modes random \
  --room data/input/room_rec.json \
  --save-blend out/interior.blend \
  --render out/interior.png

python3 src/run_pipeline.py кровать шкаф кресло стол стул \
  --placer cube \
  --modes relaxed \
  --room data/input/room_rec.json \
  --save-blend out/interior.blend \
  --render out/interior.png

python3 src/run_pipeline.py \
  --placer cube \
  --prompt "Нужно разместить кровать, шкаф, кресло, стол и стул" \
  --room data/input/room_rec.json \
  --save-blend out/interior.blend \
  --render out/interior.png

python3 src/run_pipeline.py кровать шкаф кресло стол стул \
  --placer diffuscene_remote \
  --room data/input/room_rec.json \
  --remote-runner scripts/run_diffuscene_remote.sh \
  --save-blend out/diffuscene_remote.blend \
  --render out/diffuscene_remote.png

ssh -p 32172 root@1.208.108.242

ssh -p 32172 -N -L 11435:127.0.0.1:11434 root@1.208.108.242
curl http://127.0.0.1:11435/api/tags

python3 src/run_pipeline.py кровать шкаф кресло стол стул \
  --placer ollama_llm \
  --room data/input/room_rec.json \
  --ollama-url http://127.0.0.1:11435 \
  --ollama-model gpt-oss:20b \
  --ollama-timeout 300 \
  --ollama-temperature 0.1 \
  --ollama-max-attempts 8 \
  --save-blend out/ollama_llm.blend \
  --render out/ollama_llm.png