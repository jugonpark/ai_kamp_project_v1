@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if errorlevel 1 goto directory_error

set "PY_CMD="
py -3.12 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.12"
    goto python_found
)
python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto python_found
)
py -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py"
    goto python_found
)
echo Python 3.12 64-bit 설치가 필요합니다. Python 3.13 64-bit도 지원합니다.
goto failed

:python_found
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo 프로젝트 가상환경을 생성합니다...
    %PY_CMD% -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo 가상환경 생성에 실패했습니다.
        goto failed
    )
) else (
    echo 기존 가상환경을 재사용합니다.
)

"%VENV_PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) and sys.maxsize > 2**32 else 1)"
if errorlevel 1 (
    echo 기존 .venv의 Python 버전 또는 아키텍처가 맞지 않습니다.
    echo Python 3.12/3.13 64-bit로 새 가상환경을 만들어주세요.
    goto failed
)

echo pip을 업데이트합니다...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo pip 업데이트에 실패했습니다.
    goto failed
)

echo 프로젝트 패키지를 설치합니다...
"%VENV_PY%" -m pip install --only-binary=:all: -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo 패키지 설치에 실패했습니다. 인터넷 연결과 Python 버전을 확인하세요.
    goto failed
)

echo 설치된 환경을 검증합니다...
"%VENV_PY%" "%~dp0verify_environment.py"
if errorlevel 1 (
    echo 환경 검증에 실패했습니다. 위 오류를 확인하세요.
    goto failed
)

echo ========================================
echo KAMP AI 환경 설정 완료
echo ========================================
echo GUI 실행: run_gui.bat
echo 디버그 실행: run_gui_debug.bat
pause
exit /b 0

:directory_error
echo 프로젝트 폴더로 이동하지 못했습니다.
:failed
echo 설정이 완료되지 않았습니다. 오류를 확인한 뒤 다시 실행하세요.
pause
exit /b 1
