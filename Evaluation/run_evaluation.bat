@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo Evaluating all model-generated reports
echo ========================================
echo.

python "evaluate_all_models.py"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Evaluation completed!
    echo ========================================
    echo Results saved in: Eval\
) else (
    echo.
    echo ========================================
    echo Evaluation failed!
    echo ========================================
)

pause
