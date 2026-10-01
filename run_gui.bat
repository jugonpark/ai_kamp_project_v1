@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo 가상환경이 없습니다. 먼저 setup_environment.bat를 실행해주세요.
    pause
    exit /b 1
)
start "KAMP Predictive Maintenance" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0gui_app.py"
