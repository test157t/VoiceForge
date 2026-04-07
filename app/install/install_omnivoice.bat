@echo off
cd /d "%~dp0..\.."
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: OmniVoice Environment Installer
:: https://huggingface.co/k2-fsa/OmniVoice
:: ===============================

set "OMNIVOICE_ENV_NAME=omnivoice_tts"
set "REQ_FILE=%~dp0requirements_omnivoice.txt"

echo.
echo =============================================
echo   OmniVoice Environment Installer
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
    echo [INFO] OmniVoice environment setup complete!
    echo [INFO] Environment name: %OMNIVOICE_ENV_NAME%
    echo.
) else (
    echo.
    echo [ERROR] OmniVoice environment setup failed!
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
echo [INFO] Setting up OmniVoice environment...

if not exist "%REQ_FILE%" (
    echo [ERROR] Missing requirements file: "%REQ_FILE%"
    exit /b 1
)

:: Create environment if needed (Python 3.11)
"%CONDA_EXE%" env list | findstr /C:"%OMNIVOICE_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating conda environment "%OMNIVOICE_ENV_NAME%" with Python 3.11...
    echo [INFO] Using conda-forge only to avoid Anaconda channel rate limits...
    "%CONDA_EXE%" create -n "%OMNIVOICE_ENV_NAME%" python=3.11 -c conda-forge --override-channels -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        echo [HINT] This usually means channel access is blocked/rate-limited. Try again later or use Miniforge.
        exit /b 1
    )
)

call :ACTIVATE_ENV "%OMNIVOICE_ENV_NAME%"
if errorlevel 1 exit /b 1

:: Install build dependencies first
echo [INFO] Installing build dependencies...
python -m pip install --upgrade pip wheel setuptools >nul 2>&1

:: Install PyTorch and torchaudio first (OmniVoice upstream recommendation)
echo [INFO] Installing PyTorch 2.8.0 and torchaudio 2.8.0 (CUDA 12.8 wheels)...
python -m pip install torch==2.8.0 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128

:: Verify CUDA is available (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo [WARN] Could not verify CUDA availability
)

:: Install remaining requirements from file
echo [INFO] Installing requirements from %REQ_FILE%...
python -m pip install -r "%REQ_FILE%"

:: transformers audio stack expects soxr for some tokenizer/audio modules
echo [INFO] Installing missing audio dependency (soxr)...
python -m pip install soxr

:: Final verification
echo [INFO] Verifying OmniVoice installation...
python -c "from omnivoice import OmniVoice; print('OmniVoice OK')"
if errorlevel 1 (
    echo [ERROR] OmniVoice import failed!
    exit /b 1
)

python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo [INFO] OmniVoice environment setup complete!
echo [INFO] Environment name: %OMNIVOICE_ENV_NAME%
echo [INFO] Model will be downloaded on first use
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
