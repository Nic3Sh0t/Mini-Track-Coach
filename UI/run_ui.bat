@echo off
chcp 65001 >nul
echo ========================================
echo TrackCoach Web UI Launch Script
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found, please install Python 3.7+
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Detected streamlit is not installed, installing...
    pip install streamlit>=1.28.0
    if errorlevel 1 (
        echo Error: streamlit installation failed
        pause
        exit /b 1
    )
)

REM Run test script to check environment
echo.
echo Checking application environment...
python test_app.py
if errorlevel 1 (
    echo.
    echo ========================================
    echo Environment check failed, please review error messages above
    echo ========================================
    pause
    exit /b 1
)
echo.

echo Starting Streamlit application...
echo.
echo Browser will open automatically, if not, please visit: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM First test if app.py can be imported normally (use os.getcwd() to avoid path issues)
echo Checking app.py...
python -c "import sys; import os; sys.path.insert(0, os.getcwd()); import app" 2>nul
if errorlevel 1 (
    echo.
    echo ========================================
    echo Warning: app.py import test failed
    echo Displaying detailed error information...
    echo ========================================
    python -c "import sys; import os; sys.path.insert(0, os.getcwd()); import app" 2>&1
    echo.
    echo Please check the error messages above
    pause
    exit /b 1
)

echo app.py check passed, starting Streamlit...
echo.

REM Launch Streamlit with detailed output
python -m streamlit run app.py --server.headless false --server.port 8501

pause
