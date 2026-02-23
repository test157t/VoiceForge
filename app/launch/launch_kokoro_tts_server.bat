@echo off
:: Kokoro TTS server launcher
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

call "%CONDA_BASE%\Scripts\activate.bat" "kokoro_tts"
if errorlevel 1 (
    echo [ERROR] Failed to activate Kokoro TTS environment
    pause
    exit /b 1
)

echo [INFO] Starting Kokoro TTS server...
echo [INFO] Port: 8897
echo [INFO] Available voices: af, am, bf, bm
echo.

:: Run from the servers folder
cd /d "%~dp0..\servers"
python -X faulthandler -u kokoro_tts_server.py --port 8897
if errorlevel 1 (
    echo [ERROR] Kokoro TTS server crashed!
    pause
)
pause
