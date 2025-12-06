@echo off
chcp 65001 >nul
echo ========================================
echo Streamlit Connection Test
echo ========================================
echo.
echo This script will test if Streamlit can start normally
echo.

cd /d "%~dp0"

echo Step 1: Testing simple Streamlit application...
echo.
python -m streamlit run test_streamlit_simple.py --server.headless false --server.port 8501
echo.
echo If the test above can open the browser normally, Streamlit itself is working fine
echo The problem may be in the app.py code
pause
