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

echo [INFO] Starting OmniVoice server...
echo [INFO] Port: 8898
echo.

:: Run from the servers folder
cd /d "%~dp0..\servers"
python -X faulthandler -u omnivoice_server.py --port 8898
if errorlevel 1 (
    echo [ERROR] OmniVoice server crashed!
    pause
)
pause
