# 🚀 Установка Whisper и Qwen на Windows (RTX 5090)

## 📋 Предварительные требования

1. **Windows 11** (рекомендуется)
2. **NVIDIA драйверы** — скачай последнюю версию: https://www.nvidia.com/drivers
3. **Git** — https://git-scm.com/download/win
4. **Python 3.11+** — https://www.python.org/downloads/
5. **Node.js 20+** — https://nodejs.org/
6. **Visual Studio Build Tools** — для компиляции llama.cpp

---

## 🔧 Шаг 1: Установка CUDA Toolkit

1. Скачай **CUDA Toolkit 12.6**: https://developer.nvidia.com/cuda-downloads
2. Выбери: Windows → x86_64 → 11 → exe (local)
3. Установи с настройками по умолчанию
4. Перезагрузи компьютер

**Проверка:**

```powershell
nvidia-smi
nvcc --version
```

---

## 🎙️ Шаг 2: Установка Whisper

Открой **PowerShell от администратора**:

```powershell
# Создать рабочую директорию
mkdir C:\MedDok
cd C:\MedDok

# Создать виртуальное окружение Python
python -m venv venv
.\venv\Scripts\Activate.ps1

# Установить PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Установить faster-whisper
pip install faster-whisper

# Проверить GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

**Тест Whisper:**

```powershell
python -c "from faster_whisper import WhisperModel; m = WhisperModel('large-v3', device='cuda', compute_type='float16'); print('Whisper OK!')"
```

> ⏳ При первом запуске скачает модель ~3GB

---

## 🤖 Шаг 3: Установка llama.cpp

### Вариант A: Скачать готовые бинарники (проще)

```powershell
cd C:\MedDok

# Скачать последний релиз с CUDA
# Перейди на https://github.com/ggerganov/llama.cpp/releases
# Скачай: llama-bXXXX-bin-win-cuda-cu12.x-x64.zip

# Или через PowerShell:
Invoke-WebRequest -Uri "https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-b5000-bin-win-cuda-cu12.2.0-x64.zip" -OutFile "llama-cpp.zip"

# Распаковать
Expand-Archive -Path "llama-cpp.zip" -DestinationPath "llama.cpp"
```

### Вариант B: Собрать самостоятельно (если нужна последняя версия)

```powershell
# Установить Visual Studio Build Tools
# Скачай: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# При установке выбери "Разработка классических приложений на C++"

# Установить CMake
winget install Kitware.CMake

# Клонировать и собрать
cd C:\MedDok
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Собрать с CUDA
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release

# Бинарники будут в build\bin\Release\
```

---

## 📥 Шаг 4: Скачать модель Qwen

```powershell
cd C:\MedDok
mkdir models
cd models

# Установить huggingface-cli
pip install huggingface-hub

# Скачать Qwen2.5-32B (лучший выбор для 32GB VRAM)
# Размер: ~19GB
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF qwen2.5-32b-instruct-q4_k_m.gguf --local-dir .

# ИЛИ напрямую через браузер:
# https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/qwen2.5-32b-instruct-q4_k_m.gguf
```

**Альтернативы:**

- **Qwen2.5-14B** (~8GB) — быстрее: `qwen2.5-14b-instruct-q4_k_m.gguf`
- **Qwen2.5-7B** (~4GB) — ещё быстрее: `qwen2.5-7b-instruct-q4_k_m.gguf`

---

## 🚀 Шаг 5: Запуск LLM сервера

Создай файл `C:\MedDok\start-llm.bat`:

```batch
@echo off
echo Starting LLM Server...
cd /d C:\MedDok\llama.cpp

llama-server.exe ^
    --model "C:\MedDok\models\qwen2.5-32b-instruct-q4_k_m.gguf" ^
    --port 8080 ^
    --host 0.0.0.0 ^
    --n-gpu-layers 99 ^
    --ctx-size 8192 ^
    --parallel 2 ^
    --flash-attn ^
    --cont-batching

pause
```

**Запуск:**

```powershell
C:\MedDok\start-llm.bat
```

**Проверка (в другом терминале):**

```powershell
curl http://localhost:8080/health
```

---

## 🏥 Шаг 6: Запуск МедДок

### 6.1 Распаковать приложение

```powershell
cd C:\MedDok
# Распакуй medical-voice-app-full.tar.gz сюда
tar -xzf medical-voice-app-full.tar.gz
```

### 6.2 Настроить Backend

Создай файл `C:\MedDok\medical-voice-app\server\.env`:

```env
PORT=3001
HOST=0.0.0.0
UPLOAD_DIR=./uploads

WHISPER_MODEL_PATH=large-v3
WHISPER_LANGUAGE=ru
WHISPER_DEVICE=cuda

LLM_SERVER_URL=http://localhost:8080
LLM_MODEL=qwen2.5-32b
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.1
LLM_PARALLEL_SLOTS=2
```

### 6.3 Запуск (3 терминала)

**Терминал 1 — LLM:**

```powershell
C:\MedDok\start-llm.bat
```

**Терминал 2 — Backend:**

```powershell
cd C:\MedDok\medical-voice-app\server
npm install
npm run dev
```

**Терминал 3 — Frontend:**

```powershell
cd C:\MedDok\medical-voice-app
npm install
npm run dev
```

### 6.4 Открыть приложение

Браузер: **http://localhost:5173**

---

## 📁 Итоговая структура

```
C:\MedDok\
├── venv\                    # Python окружение
├── llama.cpp\               # LLM сервер
│   └── llama-server.exe
├── models\
│   └── qwen2.5-32b-instruct-q4_k_m.gguf
├── medical-voice-app\
│   ├── server\              # Backend API
│   └── src\                 # Frontend
└── start-llm.bat            # Скрипт запуска
```

---

## 🔧 Скрипт для быстрого запуска всего

Создай `C:\MedDok\start-all.bat`:

```batch
@echo off
echo ========================================
echo    MedDok - Starting All Services
echo ========================================

:: Запуск LLM сервера в новом окне
start "LLM Server" cmd /k "cd /d C:\MedDok && start-llm.bat"

:: Подождать пока LLM загрузится
echo Waiting for LLM to load (30 sec)...
timeout /t 30 /nobreak

:: Активировать Python окружение и запустить Backend
start "Backend API" cmd /k "cd /d C:\MedDok\medical-voice-app\server && C:\MedDok\venv\Scripts\activate && npm run dev"

:: Запустить Frontend
start "Frontend" cmd /k "cd /d C:\MedDok\medical-voice-app && npm run dev"

echo ========================================
echo    All services starting...
echo    Open: http://localhost:5173
echo ========================================
pause
```

---

## ⚠️ Решение проблем

### "CUDA out of memory"

```batch
:: Уменьши слои на GPU
--n-gpu-layers 60
```

### "llama-server не найден"

```powershell
# Проверь путь к бинарнику
dir C:\MedDok\llama.cpp\*.exe
# Может быть в подпапке build\bin\Release\
```

### "Whisper не видит GPU"

```powershell
# Переустанови PyTorch с CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Python не активируется

```powershell
# Разреши выполнение скриптов
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## ✅ Чеклист

- [ ] NVIDIA драйверы установлены
- [ ] CUDA Toolkit 12.x установлен
- [ ] Python venv создан и активирован
- [ ] PyTorch с CUDA работает
- [ ] faster-whisper установлен
- [ ] llama.cpp скачан/собран
- [ ] Модель Qwen скачана (~19GB)
- [ ] LLM сервер запускается (порт 8080)
- [ ] Backend запускается (порт 3001)
- [ ] Frontend запускается (порт 5173)
- [ ] Приложение работает! 🎉

---

## 📊 Использование ресурсов

| Компонент        | VRAM       | RAM   |
| ---------------- | ---------- | ----- |
| Whisper large-v3 | ~4 GB      | ~2 GB |
| Qwen 32B Q4_K_M  | ~19 GB     | ~4 GB |
| **Итого**        | **~23 GB** | ~6 GB |
| **Свободно**     | **~9 GB**  | -     |

RTX 5090 справится отлично! 🚀
