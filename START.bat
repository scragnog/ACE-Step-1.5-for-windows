@echo off
REM ============================================================
REM  ACE-Step — One-Click Launcher
REM  Starts: Python API + Express Backend + Next.js Frontend
REM ============================================================
setlocal

REM Read frontend port from .env
set "VITE_PORT=3000"
if exist "ace-step-ui\.env" (
    for /f "tokens=2 delims==" %%a in ('findstr /b "VITE_PORT" "ace-step-ui\.env"') do set "VITE_PORT=%%a"
)

cd /d "%~dp0"

echo.
echo =============================================
echo   ACE-Step One-Click Launcher
echo =============================================
echo.

REM ---- Step 1: Check UI dependencies ----
echo [1/4] Checking UI dependencies...
if not exist "ace-step-ui\node_modules" (
    echo   Installing frontend dependencies...
    cd ace-step-ui
    call npm install
    cd ..
)
if not exist "ace-step-ui\server\node_modules" (
    echo   Installing server dependencies...
    cd ace-step-ui\server
    call npm install
    cd ..\..
)
echo   Done.
echo.

REM ---- Step 2: Set environment variables ----
REM IMPORTANT: No trailing spaces after values!
echo [2/4] Setting environment...
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "HF_HOME=huggingface"
set "XFORMERS_FORCE_DISABLE_TRITON=1"
set "PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG=1"
set "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130"
set "UV_CACHE_DIR=%LOCALAPPDATA%\uv\cache"
set "UV_NO_BUILD_ISOLATION=1"
set "UV_NO_CACHE=0"
set "UV_LINK_MODE=symlink"
set "UV_INDEX_STRATEGY=unsafe-best-match"
echo   Done.
echo.

REM ---- Step 3: Start Python API server (new window) ----
echo [3/4] Starting Python API server on port 8001...
start /min "ACE-Step Python API" cmd /k "cd /d "%~dp0" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && set "PYTHONPATH=%~dp0" && set "HF_HOME=huggingface" && set "XFORMERS_FORCE_DISABLE_TRITON=1" && set "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130" && set "UV_CACHE_DIR=%LOCALAPPDATA%\uv\cache" && set "UV_NO_BUILD_ISOLATION=1" && set "UV_LINK_MODE=symlink" && set "UV_INDEX_STRATEGY=unsafe-best-match" && uv run acestep-api --port 8001"
echo   Waiting for API to initialize...
timeout /t 10 /nobreak >nul
echo   Done.
echo.

REM ---- Step 4: Start UI (Express backend + Next.js frontend) ----
echo [4/4] Starting UI servers...

REM Get local IP for LAN access
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        set "LOCAL_IP=%%b"
    )
)

REM Set ACE-Step paths for the UI
set "ACESTEP_PATH=%~dp0"
set "PYTHON_PATH=%~dp0.venv\Scripts\python.exe"

REM Start Express backend
start /min "ACE-Step UI Backend" cmd /k "cd /d "%~dp0ace-step-ui\server" && npm run dev"
timeout /t 3 /nobreak >nul

REM Start Next.js frontend
start /min "ACE-Step UI Frontend" cmd /k "cd /d "%~dp0ace-step-ui" && npm run dev"
timeout /t 3 /nobreak >nul

echo.
echo =============================================
echo   ACE-Step is running!
echo =============================================
echo.
echo   Python API:  http://localhost:8001
echo   Backend:     http://localhost:3001
echo   Frontend:    http://localhost:%VITE_PORT%
echo.
if defined LOCAL_IP (
    echo   LAN Access:  http://%LOCAL_IP%:%VITE_PORT%
    echo.
)
echo   Three windows opened:
echo     - Python API (acestep-api)
echo     - Express Backend (server)
echo     - Next.js Frontend
echo.
echo   Close those windows to stop.
echo =============================================
echo.

REM Open browser
echo Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:%VITE_PORT%

pause
