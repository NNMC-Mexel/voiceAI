@echo off
setlocal

cd /d "%~dp0"

set "GIGAAM_SERVER_PORT=9002"
set "GIGAAM_SERVER_HOST=0.0.0.0"
set "GIGAAM_MODEL=v3_ctc"
set "GIGAAM_DEVICE=cuda"
set "GIGAAM_FALLBACK_CHUNK_SECONDS=24"

set "PYTHON_EXE=C:\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Checking GigaAM on http://127.0.0.1:9002/health ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:9002/health' -TimeoutSec 2; Write-Host ('GigaAM is already running: ' + ($r | ConvertTo-Json -Compress)); exit 10 } catch { exit 0 }"
if "%ERRORLEVEL%"=="10" goto already_running

echo.
echo Starting GigaAM:
echo   model: %GIGAAM_MODEL%
echo   device: %GIGAAM_DEVICE%
echo   url:    http://127.0.0.1:9002
echo.

"%PYTHON_EXE%" server\scripts\gigaam_server.py

echo.
echo GigaAM stopped or failed.
pause
exit /b %ERRORLEVEL%

:already_running
echo.
echo Backend can use it via WHISPER_SERVER_URL=http://127.0.0.1:9002
pause
exit /b 0
