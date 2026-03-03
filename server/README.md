# МедДок Backend Server

Node.js/Fastify сервер для обработки голосового ввода медицинских документов.

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
cd server
npm install
```

### 2. Установка Whisper (Speech-to-Text)
```bash
chmod +x scripts/install-whisper.sh
./scripts/install-whisper.sh
```

### 3. Установка LLM (Qwen)
```bash
chmod +x scripts/install-llm.sh
./scripts/install-llm.sh
```

### 4. Запуск LLM сервера
```bash
./start-llm-server.sh
```

### 5. Запуск API сервера
```bash
# Development
npm run dev

# Production
npm run build
npm start
```

## 📡 API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/health` | Проверка статуса сервера |
| GET | `/api/config` | Конфигурация сервера |
| POST | `/api/upload` | Загрузка аудиофайла |
| POST | `/api/transcribe` | Распознавание речи (STT) |
| POST | `/api/structure` | Структурирование текста (LLM) |
| POST | `/api/process` | Полный пайплайн (upload → STT → LLM) |
| POST | `/api/documents` | Сохранение документа |

### Примеры запросов

#### Полная обработка аудио
```bash
curl -X POST http://localhost:3001/api/process \
  -F "file=@recording.webm"
```

#### Только распознавание речи
```bash
# 1. Загрузить файл
curl -X POST http://localhost:3001/api/upload \
  -F "file=@recording.webm"

# Response: { "filename": "1234567890_recording.webm" }

# 2. Распознать
curl -X POST http://localhost:3001/api/transcribe \
  -H "Content-Type: application/json" \
  -d '{"filename": "1234567890_recording.webm"}'
```

#### Только структурирование
```bash
curl -X POST http://localhost:3001/api/structure \
  -H "Content-Type: application/json" \
  -d '{"text": "Пациент Иванов, 45 лет, жалобы на головную боль..."}'
```

## ⚙️ Конфигурация

Создайте файл `.env` на основе `.env.example`:

```env
# Server
PORT=3001
HOST=0.0.0.0

# Whisper
WHISPER_MODEL_PATH=./models/ggml-large-v3.bin
WHISPER_LANGUAGE=ru
WHISPER_DEVICE=cuda

# LLM
LLM_SERVER_URL=http://localhost:8080
LLM_MODEL=qwen2.5-14b
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.1
```

## 🏗️ Архитектура

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│  API Server │────▶│   Whisper   │
│   (React)   │     │  (Fastify)  │     │   (STT)     │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  LLM Server │
                   │  (llama.cpp)│
                   └─────────────┘
```

## 🔧 Требования

### Минимальные
- CPU: 8 cores
- RAM: 16 GB
- Storage: 50 GB SSD

### Рекомендуемые (для GPU)
- GPU: NVIDIA RTX 3080+ (12GB+ VRAM)
- RAM: 32 GB
- Storage: 100 GB NVMe SSD

## 🐳 Docker

```bash
# Из корня проекта
docker-compose up -d
```

## 📁 Структура

```
server/
├── src/
│   ├── index.ts          # Точка входа
│   ├── config.ts         # Конфигурация
│   ├── routes.ts         # API роуты
│   ├── types.ts          # TypeScript типы
│   └── services/
│       ├── whisper.ts    # Сервис STT
│       └── llm.ts        # Сервис LLM
├── scripts/
│   ├── install-whisper.sh
│   └── install-llm.sh
├── models/               # Модели AI
├── uploads/             # Временные файлы
└── Dockerfile
```

## 🔒 Безопасность

- Все данные обрабатываются локально
- Нет передачи в облачные сервисы
- Рекомендуется работа в изолированной сети
