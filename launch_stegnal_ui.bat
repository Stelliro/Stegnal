@echo off
cd /d "%~dp0"
title Stegnal // Desktop Explorer
color 0b

echo ========================================================
echo        STEGNAL // DESKTOP EXPLORER (stegnal ui)
echo ========================================================
echo.

call "%~dp0_bootstrap_env.bat"
if errorlevel 1 (
    echo.
    echo [CRITICAL] Environment setup failed.
    pause
    exit /b 1
)

echo [SYSTEM] Launching Desktop Explorer...
echo ========================================================
"%~dp0.venv\Scripts\python.exe" -m stegnal ui

if errorlevel 1 (
    echo.
    echo [CRASH] The application exited with error code %errorlevel%.
    pause
)
