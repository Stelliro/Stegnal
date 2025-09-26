@echo off
setlocal
cd /d "%~dp0"

rem Resolve Python command (prefer python, fallback to py)
set "PY_CMD="
where python >nul 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
    where py >nul 2>&1 && set "PY_CMD=py"
)
if not defined PY_CMD (
    echo Could not find Python on PATH. Please install Python 3.9+ and try again.
    exit /b 1
)

rem Ensure required dependencies are available
%PY_CMD% -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo Failed to upgrade pip. Continuing with existing version.
)
%PY_CMD% -m pip install --upgrade --quiet --editable "%cd%"
if errorlevel 1 (
    echo Failed to install Project Umbra dependencies.
    exit /b 1
)

rem Launch the Project Umbra UI via the CLI wrapper
%PY_CMD% -m umbra ui %*
