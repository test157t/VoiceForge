@echo off
:: OmniVoice server launcher
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

set "OMNIVOICE_RUNTIME=torch"
if /I "%~1"=="onnx" set "OMNIVOICE_RUNTIME=onnx-cpu"
if /I "%~1"=="onnx-gpu" set "OMNIVOICE_RUNTIME=onnx-gpu"

if /I "%OMNIVOICE_RUNTIME%"=="onnx-cpu" (
    if not defined OMNIVOICE_MODEL_ID set "OMNIVOICE_MODEL_ID=gluschenko/omnivoice-onnx"
    if not defined OMNIVOICE_ONNX_MODEL_FILE set "OMNIVOICE_ONNX_MODEL_FILE=onnx/omnivoice.qint8.onnx"
    if not defined OMNIVOICE_ONNX_NUM_STEP set "OMNIVOICE_ONNX_NUM_STEP=24"
    if not defined OMNIVOICE_ONNX_GRAPH_OPT set "OMNIVOICE_ONNX_GRAPH_OPT=all"
    if not defined OMNIVOICE_ONNX_INTER_OP_THREADS set "OMNIVOICE_ONNX_INTER_OP_THREADS=1"
    if not defined OMNIVOICE_ONNX_INTRA_OP_THREADS set "OMNIVOICE_ONNX_INTRA_OP_THREADS=%NUMBER_OF_PROCESSORS%"
)
if /I "%OMNIVOICE_RUNTIME%"=="onnx-gpu" (
    if not defined OMNIVOICE_MODEL_ID set "OMNIVOICE_MODEL_ID=gluschenko/omnivoice-onnx"
    if not defined OMNIVOICE_ONNX_MODEL_FILE set "OMNIVOICE_ONNX_MODEL_FILE=onnx/omnivoice.qint8.onnx"
    if not defined OMNIVOICE_ONNX_NUM_STEP set "OMNIVOICE_ONNX_NUM_STEP=24"
    if not defined OMNIVOICE_ONNX_GRAPH_OPT set "OMNIVOICE_ONNX_GRAPH_OPT=all"
    if not defined OMNIVOICE_ONNX_GPU_PROVIDERS set "OMNIVOICE_ONNX_GPU_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider"
)

echo [INFO] Starting OmniVoice server...
echo [INFO] Port: 8898
echo [INFO] Runtime: %OMNIVOICE_RUNTIME%
if /I "%OMNIVOICE_RUNTIME%"=="onnx-cpu" echo [INFO] ONNX file: %OMNIVOICE_ONNX_MODEL_FILE%
if /I "%OMNIVOICE_RUNTIME%"=="onnx-gpu" echo [INFO] ONNX file: %OMNIVOICE_ONNX_MODEL_FILE%
if /I "%OMNIVOICE_RUNTIME%"=="onnx-cpu" echo [INFO] ONNX num_step: %OMNIVOICE_ONNX_NUM_STEP%
if /I "%OMNIVOICE_RUNTIME%"=="onnx-gpu" echo [INFO] ONNX num_step: %OMNIVOICE_ONNX_NUM_STEP%
if /I "%OMNIVOICE_RUNTIME%"=="onnx-cpu" echo [INFO] ONNX intra/inter threads: %OMNIVOICE_ONNX_INTRA_OP_THREADS%/%OMNIVOICE_ONNX_INTER_OP_THREADS%
if /I "%OMNIVOICE_RUNTIME%"=="onnx-gpu" echo [INFO] ONNX GPU providers: %OMNIVOICE_ONNX_GPU_PROVIDERS%
echo.

:: Run from the servers folder
cd /d "%~dp0..\servers"
python -X faulthandler -u omnivoice_server.py --port 8898
if errorlevel 1 (
    echo [ERROR] OmniVoice server crashed!
    pause
)
pause
