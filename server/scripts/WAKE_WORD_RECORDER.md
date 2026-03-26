# Wake-Word Recorder для VoiceMed

Скрипт для always-listening режима: врач говорит **"Джарвис"** — начинается запись, **"стоп Джарвис"** — запись завершается и отправляется на сервер.

## Установка

```bash
# 1. Зависимости Python
pip install -r requirements-wake-word.txt

# 2. Скачать Vosk модель для русского языка
#    Маленькая (~50 MB, быстрая, подходит для wake-word):
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip

#    Или большая (~1.5 GB, точнее):
# wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
```

## Запуск

```bash
# Минимальный запуск (без авторизации):
python wake_word_recorder.py \
  --model-path ./vosk-model-small-ru-0.22 \
  --api-url http://localhost:3000

# С авторизацией по паролю:
python wake_word_recorder.py \
  --model-path ./vosk-model-small-ru-0.22 \
  --api-url http://192.168.1.10:3000 \
  --auth-password mypassword

# С токеном:
python wake_word_recorder.py \
  --model-path ./vosk-model-small-ru-0.22 \
  --api-url http://localhost:3000 \
  --token "Bearer abc123"

# Кастомные фразы:
python wake_word_recorder.py \
  --model-path ./vosk-model-small-ru-0.22 \
  --api-url http://localhost:3000 \
  --wake-phrases "джарвис,начать запись" \
  --stop-phrases "стоп джарвис,конец записи"
```

## Параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--model-path` | (обязательный) | Путь к папке Vosk модели |
| `--api-url` | (обязательный) | URL сервера VoiceMed |
| `--token` | — | Bearer токен авторизации |
| `--auth-password` | — | Пароль для автоматического логина |
| `--wake-phrases` | `джарвис` | CSV фразы для начала записи |
| `--stop-phrases` | `стоп джарвис,конец записи` | CSV фразы для остановки |
| `--sample-rate` | 16000 | Частота дискретизации (Гц) |
| `--block-size` | 4000 | Размер блока (сэмплы) |
| `--max-duration` | 120 | Макс. длительность записи (сек) |

## Как работает

```
[IDLE] ──"Джарвис"──> [RECORDING] ──"стоп Джарвис"──> [UPLOADING] ──> [IDLE]
                           │                                │
                           │ (макс 120 сек)                 └─> /api/process
                           └──────────────────────────────────> /api/process
```

1. **IDLE** — микрофон слушает, аудио НЕ сохраняется. Последние 2 сек хранятся в кольцевом буфере (pre-buffer).
2. **RECORDING** — после wake-word аудио копится в памяти (включая pre-buffer, чтобы не срезать начало фразы).
3. **UPLOADING** — WAV сохраняется во временный файл, отправляется на `/api/process`, файл удаляется.
4. Возврат в **IDLE**.

## Smoke-test

1. Запустить VoiceMed сервер (`npm run dev` в `server/`)
2. Запустить скрипт (см. выше)
3. Убедиться, что в логах: `Listening for wake phrase: джарвис`
4. Сказать в микрофон: **"Джарвис"**
5. Убедиться: `State: idle -> recording`
6. Надиктовать текст (например: "Пациент Иванов, жалобы на головную боль")
7. Сказать: **"стоп Джарвис"**
8. Убедиться: `State: recording -> uploading`, затем `Processing OK!`
9. Убедиться: `State: uploading -> idle`
