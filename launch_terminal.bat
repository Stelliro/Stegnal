@echo off
cd /d "%~dp0"
title Project Umbra // Terminal
color 0b

echo ========================================================
echo        PROJECT UMBRA // TERMINAL (app.py)
echo ========================================================
echo.

call "%~dp0_bootstrap_env.bat"
if errorlevel 1 (
    echo.
    echo [CRITICAL] Environment setup failed.
    pause
    exit /b 1
)

echo [SYSTEM] Launching Terminal...
echo ========================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"

if errorlevel 1 (
    echo.
    echo [CRASH] The application exited with error code %errorlevel%.
    pause
)
