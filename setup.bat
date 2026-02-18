@echo off
setlocal
echo clarogent Setup Wizard

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

:: 2. Create Virtual Environment
if not exist "venv" (
    echo [1/3] Creating Virtual Environment...
    python -m venv venv
) else (
    echo [1/3] Virtual Environment found.
)

:: 3. Activate and Install
echo [2/3] Installing Dependencies...
call venv\Scripts\activate
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

:: 4. Launch
echo [3/3] Starting Dashboard...
echo.
streamlit run app.py
pause
