@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo 가상환경이 없습니다. 먼저 setup_environment.bat를 실행해주세요.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0gui_app.py"
if errorlevel 1 echo GUI 실행 중 오류가 발생했습니다. 위 traceback을 확인하세요.
pause
