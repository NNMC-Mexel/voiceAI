@echo off
setlocal

cd /d "%~dp0"

if not defined GIGAAM_SERVER_PORT set "GIGAAM_SERVER_PORT=9002"
if not defined GIGAAM_SERVER_HOST set "GIGAAM_SERVER_HOST=127.0.0.1"
if not defined GIGAAM_ALLOW_REMOTE_BIND set "GIGAAM_ALLOW_REMOTE_BIND=false"
if not defined GIGAAM_ALLOW_AUDIO_PATH set "GIGAAM_ALLOW_AUDIO_PATH=false"
if not defined GIGAAM_MODEL set "GIGAAM_MODEL=v3_ctc"
if not defined GIGAAM_DEVICE set "GIGAAM_DEVICE=cuda"
if not defined GIGAAM_LONGFORM_MODE set "GIGAAM_LONGFORM_MODE=auto"
if not defined GIGAAM_VAD_TARGET_SECONDS set "GIGAAM_VAD_TARGET_SECONDS=20"
if not defined GIGAAM_VAD_HARD_MAX_SECONDS set "GIGAAM_VAD_HARD_MAX_SECONDS=24"
if not defined GIGAAM_VAD_PADDING_SECONDS set "GIGAAM_VAD_PADDING_SECONDS=0.25"
if not defined GIGAAM_FALLBACK_CHUNK_SECONDS set "GIGAAM_FALLBACK_CHUNK_SECONDS=20"
if not defined GIGAAM_FALLBACK_OVERLAP_SECONDS set "GIGAAM_FALLBACK_OVERLAP_SECONDS=2"
if not defined GIGAAM_RUNTIME_LOCK set "GIGAAM_RUNTIME_LOCK=%~dp0server\scripts\gigaam-runtime.lock.json"
if not defined GIGAAM_HASH_CHECKPOINT set "GIGAAM_HASH_CHECKPOINT=true"
if not defined GIGAAM_STRICT_RUNTIME_LOCK set "GIGAAM_STRICT_RUNTIME_LOCK=false"

if not defined GIGAAM_PYTHON set "GIGAAM_PYTHON=python"

echo Checking GigaAM on http://127.0.0.1:%GIGAAM_SERVER_PORT%/health ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri ('http://127.0.0.1:' + $env:GIGAAM_SERVER_PORT + '/health') -TimeoutSec 2; Write-Host ('GigaAM is already running: ' + ($r | ConvertTo-Json -Compress)); exit 10 } catch { exit 0 }"
if "%ERRORLEVEL%"=="10" goto already_running

echo.
echo Starting GigaAM:
echo   model: %GIGAAM_MODEL%
echo   device: %GIGAAM_DEVICE%
echo   url:    http://127.0.0.1:%GIGAAM_SERVER_PORT%
echo   python: %GIGAAM_PYTHON%
echo.

"%GIGAAM_PYTHON%" server\scripts\gigaam_server.py

echo.
echo GigaAM stopped or failed.
pause
exit /b %ERRORLEVEL%

:already_running
echo.
echo Backend can use it via WHISPER_SERVER_URL=http://127.0.0.1:%GIGAAM_SERVER_PORT%
pause
exit /b 0
