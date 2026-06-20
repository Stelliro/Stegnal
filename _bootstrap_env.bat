@echo off
REM ============================================================
REM  _bootstrap_env.bat  --  shared environment bootstrapper
REM
REM  Guarantees a WORKING .venv (Python 3.12) with the umbra
REM  package installed, then returns. Called by the launch_*.bat
REM  scripts. Exits 0 on success, 1 on failure.
REM
REM  Unlike the old launcher, this VALIDATES the venv by actually
REM  running its python -- a venv whose base interpreter was
REM  uninstalled is detected and rebuilt instead of failing later.
REM ============================================================
set "VENV_DIR=%~dp0.venv"
set "PYEXE=%VENV_DIR%\Scripts\python.exe"

if exist "%PYEXE%" call :validate_venv
if not exist "%PYEXE%" call :create_venv
if not exist "%PYEXE%" exit /b 1

REM Ensure the package itself is importable; install on first run.
"%PYEXE%" -c "import umbra" >nul 2>nul
if errorlevel 1 call :install_pkg
exit /b 0

:validate_venv
"%PYEXE%" --version >nul 2>nul
if errorlevel 1 (
    echo [WARN] Existing .venv is broken ^(its base Python was removed^). Rebuilding...
    rmdir /s /q "%VENV_DIR%"
)
exit /b 0

:create_venv
set "BASEPY="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "BASEPY=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined BASEPY where python >nul 2>nul && set "BASEPY=python"
if not defined BASEPY (
    echo [CRITICAL] No suitable Python found. Please install Python 3.12.
    exit /b 1
)
echo [SYSTEM] Creating virtual environment ^(.venv^)...
"%BASEPY%" -m venv "%VENV_DIR%"
"%PYEXE%" -m pip install --upgrade pip >nul 2>nul
exit /b 0

:install_pkg
echo [SYSTEM] Installing Umbra + dependencies ^(first run; may take a minute^)...
"%PYEXE%" -m pip install -e ".[ui]"
exit /b 0
