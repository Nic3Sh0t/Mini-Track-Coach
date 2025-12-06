@echo off
chcp 65001 >nul
echo ========================================
echo TrackCoach Web UI Launch Script (Debug Mode)
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version
if errorlevel 1 (
    echo Error: Python not found, please install Python 3.7+
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 1: Check Dependencies
echo ========================================
echo.

REM Check if streamlit is installed
echo Checking streamlit...
python -c "import streamlit; print('streamlit version:', streamlit.__version__)" 2>&1
if errorlevel 1 (
    echo streamlit not installed, installing...
    pip install streamlit>=1.28.0
    if errorlevel 1 (
        echo Error: streamlit installation failed
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Step 2: Check Paths
echo ========================================
echo.

REM Use os.getcwd() to avoid backslash issues in paths
python -c "import os; from pathlib import Path; ui_dir = Path(os.getcwd()); new_test_dir = ui_dir.parent; proj_root = new_test_dir.parent; print('UI Directory:', ui_dir); print('Parent Directory:', new_test_dir); print('Project Root Directory:', proj_root); print('trackcoach directory exists:', (new_test_dir / 'trackcoach').exists()); print('pipeline.py exists:', (new_test_dir / 'trackcoach' / 'pipeline.py').exists())"

echo.
echo ========================================
echo Step 3: Test app.py Import
echo ========================================
echo.

REM Use environment variables to avoid backslash issues in paths
python -c "import sys; from pathlib import Path; import os; ui_dir = Path(os.getcwd()); sys.path.insert(0, str(ui_dir.parent)); import app; print('app.py import successful')" 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo app.py import failed! Detailed error information:
    echo ========================================
    python -c "import sys; from pathlib import Path; import os; ui_dir = Path(os.getcwd()); sys.path.insert(0, str(ui_dir.parent)); import app" 2>&1
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 4: Launch Streamlit
echo ========================================
echo.
echo Starting Streamlit application...
echo Browser will open automatically, if not, please visit: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

python -m streamlit run app.py --server.headless false --server.port 8501 --logger.level=debug

pause
