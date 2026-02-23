@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Kokoro TTS Environment Installer
:: https://github.com/thewh1teagle/kokoro-onnx
:: ===============================

set "KOKORO_TTS_ENV_NAME=kokoro_tts"
set "REQ_FILE=%~dp0requirements_kokoro-tts.txt"

echo.
echo =============================================
echo Kokoro TTS Environment Installer
echo =============================================
echo.

:: Find conda
call :FIND_CONDA
if not defined CONDA_EXE (
    echo [ERROR] Conda not found. Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Run installation
call :DO_INSTALL
set "INSTALL_RESULT=%ERRORLEVEL%"

if %INSTALL_RESULT% equ 0 (
    echo.
    echo [INFO] Kokoro TTS environment setup complete!
    echo [INFO] Environment name: %KOKORO_TTS_ENV_NAME%
    echo.
) else (
    echo.
    echo [ERROR] Kokoro TTS environment setup failed!
    echo.
)

pause
endlocal
exit /b %INSTALL_RESULT%

:: ===============================
:: Installation Logic
:: ===============================
:DO_INSTALL
echo.
echo [INFO] Setting up Kokoro TTS environment...

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    exit /b 1
)

:: Remove defaults channel to avoid anaconda.com rate limits
echo [INFO] Configuring conda to use conda-forge only...
"%CONDA_EXE%" config --remove channels defaults 2>nul
"%CONDA_EXE%" config --add channels conda-forge

:: Create environment if needed (Python 3.11)
"%CONDA_EXE%" env list | findstr /C:"%KOKORO_TTS_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%KOKORO_TTS_ENV_NAME%" with Python 3.11...
    "%CONDA_EXE%" create -n "%KOKORO_TTS_ENV_NAME%" python=3.11 -c conda-forge --override-channels -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%KOKORO_TTS_ENV_NAME%"
if errorlevel 1 exit /b 1

:: Install build dependencies first
echo [INFO] Installing build dependencies...
python -m pip install --upgrade pip wheel setuptools >nul 2>&1

:: Install requirements from file
echo [INFO] Installing requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

:: Verify Kokoro TTS installation
echo [INFO] Verifying Kokoro TTS installation...
python -c "from kokoro_onnx import Kokoro; print('Kokoro TTS OK')"
if errorlevel 1 (
    echo [ERROR] Kokoro TTS import failed!
    exit /b 1
)

echo.
echo [INFO] Kokoro TTS environment setup complete!
echo [INFO] Environment name: %KOKORO_TTS_ENV_NAME%
echo [INFO] Model will be downloaded on first use
echo [INFO] Available voices: af, am, bf, bm
exit /b 0

:: ===============================
:: Helper: Find Conda
:: ===============================
:FIND_CONDA
set "CONDA_EXE="
set "CONDA_BASE="
for %%D in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if exist "%%~D\Scripts\conda.exe" (
        set "CONDA_EXE=%%~D\Scripts\conda.exe"
        set "CONDA_BASE=%%~D"
        goto :EOF
    )
)
goto :EOF

:: ===============================
:: Helper: Activate Environment
:: ===============================
:ACTIVATE_ENV
set "TARGET_ENV=%~1"
if not defined CONDA_BASE (
    echo [ERROR] Conda not found.
    exit /b 1
)

set "ENV_DIR=%CONDA_BASE%\envs\%TARGET_ENV%"
if not exist "%ENV_DIR%" (
    echo [INFO] Environment "%TARGET_ENV%" does not exist yet - will be created.
    exit /b 0
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%PATH%"
set "CONDA_DEFAULT_ENV=%TARGET_ENV%"
set "CONDA_PREFIX=%ENV_DIR%"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found after activation.
    exit /b 1
)
exit /b 0
