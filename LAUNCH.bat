@echo off
REM ============================================================
REM  ACE-Step — Single-Click Launcher with Loading Screen
REM  Opens browser immediately, then launches the existing
REM  PowerShell scripts that are known to work.
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"

REM Tell start.bat not to open a browser — our loading page handles that
set "ACESTEP_NO_BROWSER=1"

echo.
REM Read frontend port from .env
set "VITE_PORT=3000"
if exist "ace-step-ui\.env" (
    for /f "tokens=2 delims==" %%a in ('findstr /b "VITE_PORT" "ace-step-ui\.env"') do set "VITE_PORT=%%a"
)

REM Read current model selections from .env
set "CURRENT_MODEL=acestep-v15-base"
set "CURRENT_LM_MODEL=acestep-5Hz-lm-0.6B"
if exist ".env" (
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_CONFIG_PATH=" ".env"') do set "CURRENT_MODEL=%%b"
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_LM_MODEL_PATH=" ".env"') do set "CURRENT_LM_MODEL=%%b"
)

REM Scan checkpoints/ for available ACE-Step models (acestep-v15-*)
set "MODEL_LIST="
for /d %%d in (checkpoints\acestep-v15-*) do (
    set "DIRNAME=%%~nxd"
    if defined MODEL_LIST (
        set "MODEL_LIST=!MODEL_LIST!,'!DIRNAME!'"
    ) else (
        set "MODEL_LIST='!DIRNAME!'"
    )
)
if not defined MODEL_LIST set "MODEL_LIST='acestep-v15-base'"

REM Scan checkpoints/ for available LM models (acestep-5Hz-lm-*)
set "LM_MODEL_LIST="
for /d %%d in (checkpoints\acestep-5Hz-lm-*) do (
    set "DIRNAME=%%~nxd"
    if defined LM_MODEL_LIST (
        set "LM_MODEL_LIST=!LM_MODEL_LIST!,'!DIRNAME!'"
    ) else (
        set "LM_MODEL_LIST='!DIRNAME!'"
    )
)
if not defined LM_MODEL_LIST set "LM_MODEL_LIST='acestep-5Hz-lm-0.6B'"

echo =============================================
echo   ACE-Step One-Click Launcher
echo =============================================
echo.

REM ---- Step 1: Write config and open loading page ----
echo [1/4] Opening loading screen...
(
echo var VITE_PORT = '%VITE_PORT%';
echo var AVAILABLE_MODELS = [%MODEL_LIST%];
echo var AVAILABLE_LM_MODELS = [%LM_MODEL_LIST%];
echo var CURRENT_MODEL = '%CURRENT_MODEL%';
echo var CURRENT_LM_MODEL = '%CURRENT_LM_MODEL%';
) > "%~dp0loading-config.js"
start "" "%~dp0loading.html"
echo   Done.
echo.

REM ---- Step 2: Check UI dependencies ----
echo [2/4] Checking UI dependencies...
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

REM ---- Step 2b: Rebuild server TypeScript ----
echo [2b/4] Building server...
cd ace-step-ui\server
call npx tsc 2>nul
cd ..\..
echo   Done.
echo.

REM ---- Step 2c: Clear Python bytecode cache for fresh code ----
echo [2c/4] Clearing Python bytecode cache...
for /d /r "acestep" %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d"
)
echo   Done.
echo.

REM ---- Step 2d: Patch checkpoints for solver/guidance support ----
echo [2d/4] Patching checkpoints for solver/guidance support...
.venv\Scripts\python.exe patch_checkpoints.py
echo   Done.
echo.

REM ---- Step 3: Start UI servers FIRST (Express + Vite) ----
REM  Express starts fast (~2s), giving the loading screen time to
REM  call /api/models/update-env before Python reads .env.
echo [3/4] Starting UI servers...
start /min "ACE-Step UI" powershell -ExecutionPolicy Bypass -Command "Set-Location '%~dp0'; & '.\4、run_npmgui.ps1'"
echo   Started (minimized window).
echo.

REM ---- Brief pause to let Express come up before Python reads .env ----
timeout /t 3 /nobreak >nul

REM ---- Step 4: Start Python API server ----
echo [4/4] Starting Python API server...
start /min "ACE-Step Python API" powershell -ExecutionPolicy Bypass -Command "Set-Location '%~dp0'; & '.\3、run_server.ps1'"
echo   Started.
echo.

REM ---- Done ----
echo =============================================
echo   All services starting up!
echo =============================================
echo.
echo   The loading screen is open in your browser.
echo   It will auto-redirect to ACE-Step once
echo   all services are ready.
echo.
echo   Python API:  http://localhost:8001
echo   Backend:     http://localhost:3001
echo   Frontend:    http://localhost:%VITE_PORT%
echo.
echo   Two minimized windows are running:
echo     - Python API (run_server.ps1)
echo     - UI servers  (run_npmgui.ps1)
echo.
echo   Close those windows to stop the services.
echo =============================================
echo.
echo   This window will close automatically.
echo   (Services will keep running in the background)
timeout /t 5 /nobreak >nul
