@echo off
:: OmniVoice ONNX CPU server launcher
cd /d "%~dp0.."

:: Find conda base directory
set "CONDA_BASE="
for %%D in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if exist "%%~D\Scripts\conda.exe" (
        set "CONDA_BASE=%%~D"
        goto :found_conda
    )
)

:found_conda
if not defined CONDA_BASE (
    echo [ERROR] Conda not found. Please install Miniconda or ensure it's in the standard location.
    pause
    exit /b 1
)

call "%CONDA_BASE%\Scripts\activate.bat" "omnivoice_tts"
if errorlevel 1 (
    echo [ERROR] Failed to activate OmniVoice environment
    pause
    exit /b 1
)

set "OMNIVOICE_RUNTIME=onnx-cpu"
if not defined OMNIVOICE_MODEL_ID set "OMNIVOICE_MODEL_ID=gluschenko/omnivoice-onnx"
if not defined OMNIVOICE_ONNX_MODEL_FILE set "OMNIVOICE_ONNX_MODEL_FILE=onnx/omnivoice.qint8.onnx"

echo [INFO] Starting OmniVoice ONNX server...
echo [INFO] Port: 8899
echo [INFO] Runtime: %OMNIVOICE_RUNTIME%
echo [INFO] Model: %OMNIVOICE_MODEL_ID%
echo [INFO] ONNX file: %OMNIVOICE_ONNX_MODEL_FILE%
echo.

:: Run from the servers folder
cd /d "%~dp0..\servers"
python -X faulthandler -u omnivoice_server.py --port 8899
if errorlevel 1 (
    echo [ERROR] OmniVoice ONNX server crashed!
    pause
)
pause
