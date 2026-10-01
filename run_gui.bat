@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Python virtual environment not found: .venv\Scripts\pythonw.exe
    pause
    exit /b 1
)
start "KAMP Predictive Maintenance" ".venv\Scripts\pythonw.exe" "gui_app.py"
