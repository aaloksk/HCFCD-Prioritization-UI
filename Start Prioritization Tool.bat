@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "SUPPORT_DIR=%ROOT_DIR%Support Files"
set "APP_PATH=%SUPPORT_DIR%\app.py"
set "REQ_PATH=%SUPPORT_DIR%\requirements.txt"
set "VENV_DIR=%SUPPORT_DIR%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "STAMP_FILE=%VENV_DIR%\requirements.sha256"

if not exist "%APP_PATH%" (
    echo Missing application file:
    echo %APP_PATH%
    pause
    exit /b 1
)

call :find_python
if errorlevel 1 (
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo First-time setup: creating local Python environment...
    if /I "%BOOTSTRAP_MODE%"=="EXE" (
        "%BOOTSTRAP_PYTHON%" -m venv "%VENV_DIR%"
    ) else (
        call %BOOTSTRAP_PYTHON% -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo Failed to create local virtual environment.
        pause
        exit /b 1
    )
)

set "REQ_HASH="
for /f %%H in ('powershell -NoProfile -Command "$p=$env:REQ_PATH; if(-not $p){exit 1}; (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash"') do set "REQ_HASH=%%H"
if not defined REQ_HASH (
    echo Failed to calculate requirements hash.
    pause
    exit /b 1
)
set "NEEDS_INSTALL=1"
if exist "%STAMP_FILE%" (
    set /p INSTALLED_HASH=<"%STAMP_FILE%"
    if /I "%INSTALLED_HASH%"=="%REQ_HASH%" set "NEEDS_INSTALL=0"
)

if "%NEEDS_INSTALL%"=="1" (
    echo Installing or updating support packages...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to update pip.
        pause
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install -r "%REQ_PATH%"
    if errorlevel 1 (
        echo Failed to install required packages.
        pause
        exit /b 1
    )
    > "%STAMP_FILE%" echo %REQ_HASH%
)

echo Launching Prioritization Tool...
"%PYTHON_EXE%" -m streamlit run "%APP_PATH%" --browser.gatherUsageStats false
exit /b %errorlevel%

:find_python
set "BOOTSTRAP_PYTHON="
set "BOOTSTRAP_MODE="

call :try_command py -3
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_exe "%LOCALAPPDATA%\anaconda3\envs\PrioritznFrmwrkHCFCD\python.exe"
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_exe "%LOCALAPPDATA%\anaconda3\python.exe"
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_exe "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_exe "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_exe "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_exe "C:\ProgramData\anaconda3\python.exe"
if defined BOOTSTRAP_PYTHON exit /b 0

call :try_command python
if defined BOOTSTRAP_PYTHON exit /b 0

echo Python 3 is required the first time this tool runs.
echo Please install Python 3.11 or newer, then run this launcher again.
exit /b 1

:try_command
call %* --version >nul 2>&1
if not errorlevel 1 (
    set "BOOTSTRAP_PYTHON=%*"
    set "BOOTSTRAP_MODE=CMD"
)
exit /b 0

:try_exe
if exist %1 (
    %1 --version >nul 2>&1
    if not errorlevel 1 (
        set "BOOTSTRAP_PYTHON=%~1"
        set "BOOTSTRAP_MODE=EXE"
    )
)
exit /b 0
