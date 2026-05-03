@echo off
cd /d %~dp0
echo Iniciando backend de reconocimiento facial...
uv run uvicorn main:app --host 0.0.0.0 --port 8000
pause
