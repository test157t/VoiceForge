@echo off
cd /d "%~dp0"
setlocal EnableExtensions EnableDelayedExpansion

:: ===============================
:: Configuration
:: ===============================
set "CONDA_ENV_NAME=voiceforge"
set "ASR_ENV_NAME=asr"
set "RVC_ENV_NAME=rvc"
set "AUDIO_SERVICES_ENV_NAME=audio_services"
set "CHATTERBOX_ENV_NAME=chatterbox"
set "CHATTERBOX_TRAIN_ENV_NAME=chatterbox_train"
set "POCKET_TTS_ENV_NAME=pocket_tts"
set "KOKORO_TTS_ENV_NAME=kokoro_tts"
set "OMNIVOICE_TTS_ENV_NAME=omnivoice_tts"
set "CONFIG_FILE=%~dp0voiceforge_config.bat"
set "ENV_CONFIG_FILE=%~dp0voiceforge_env_config.bat"

:: Server Selection Defaults (1=enabled, 0=disabled)
set "LAUNCH_ASR=1"
set "LAUNCH_AUDIO_SERVICES=1"
set "LAUNCH_RVC=1"
set "LAUNCH_CHATTERBOX=1"
set "LAUNCH_POCKET_TTS=1"
set "LAUNCH_KOKORO_TTS=1"
set "LAUNCH_OMNIVOICE_TTS=1"
set "LAUNCH_TRAINING=1"

:: Environment Install Defaults (1=install, 0=skip)
set "INSTALL_MAIN=1"
set "INSTALL_ASR=1"
set "INSTALL_AUDIO=1"
set "INSTALL_RVC=1"
set "INSTALL_CHATTERBOX=1"
set "INSTALL_POCKET_TTS=1"
set "INSTALL_KOKORO_TTS=1"
set "INSTALL_OMNIVOICE_TTS=1"
set "INSTALL_TRAINING=0"

:: Load saved preferences if config files exist
if exist "%CONFIG_FILE%" (
    call "%CONFIG_FILE%"
)
if exist "%ENV_CONFIG_FILE%" (
    call "%ENV_CONFIG_FILE%"
)
set "REQ_FILE=%~dp0app\install\requirements_main.txt"
set "CUSTOM_DEPS=%~dp0app\assets\custom_dependencies"
set "PYTHONFAULTHANDLER=1"
set "ASR_SERVER_PORT=8889"
set "RVC_SERVER_PORT=8891"
set "AUDIO_SERVICES_SERVER_PORT=8892"
set "CHATTERBOX_SERVER_PORT=8893"
set "POCKET_TTS_SERVER_PORT=8894"
set "KOKORO_TTS_SERVER_PORT=8896"
set "OMNIVOICE_TTS_SERVER_PORT=8898"
set "TRAINING_SERVER_PORT=8895"

:: Find conda
call :FIND_CONDA
if not defined CONDA_EXE (
    echo [ERROR] Conda not found. Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: ===============================
:: Main Menu
:: ===============================
:MENU
cls
echo.
echo =============================================
echo VoiceForge Launcher
echo =============================================
echo [1] Start - Launch App
echo [2] Configure - Select Servers to Launch
echo [3] Setup - Install / Manage Environments
echo [4] Exit
echo =============================================
call :SHOW_SAVED_STATUS
echo.

choice /C 1234 /N /M "Choose [1-4]: "
if errorlevel 4 goto END
if errorlevel 3 goto UTILITIES_MENU
if errorlevel 2 goto CONFIGURE_SERVERS
if errorlevel 1 goto RUN
goto MENU

:: ===============================
:: Configure Servers Menu
:: ===============================
:CONFIGURE_SERVERS
cls
echo.
echo =============================================
echo Configure Servers to Launch
echo =============================================
echo Toggle servers ON/OFF to save preferences:
echo.
call :SHOW_SERVER_STATUS
echo.
echo [1] Toggle ASR Server         - Whisper + GLM-ASR + Parakeet
echo [2] Toggle Audio Services     - Pre/Post processing
echo [3] Toggle RVC Server         - Voice conversion
echo [4] Toggle Chatterbox         - TTS
echo [5] Toggle Pocket TTS         - Lightweight TTS
echo [6] Toggle Kokoro TTS         - ONNX TTS
echo [7] Toggle OmniVoice TTS      - Multilingual TTS
echo [8] Toggle Training Environment - Model training
echo.
echo [9] Save Configuration
echo [0] Enable All
echo [D] Disable All
echo [R] Reset to Defaults
echo [B] Back to Setup Menu
echo =============================================
echo.

choice /C 1234567890DRB /N /M "Choose option: "
if errorlevel 13 goto UTILITIES_MENU
if errorlevel 12 goto RESET_SERVER_DEFAULTS
if errorlevel 11 goto DISABLE_ALL_SERVERS
if errorlevel 10 goto ENABLE_ALL_SERVERS
if errorlevel 9 goto SAVE_SERVER_CONFIG
if errorlevel 8 call :TOGGLE_ENV LAUNCH_TRAINING
if errorlevel 7 call :TOGGLE_ENV LAUNCH_OMNIVOICE_TTS
if errorlevel 6 call :TOGGLE_ENV LAUNCH_KOKORO_TTS
if errorlevel 5 call :TOGGLE_ENV LAUNCH_POCKET_TTS
if errorlevel 4 call :TOGGLE_ENV LAUNCH_CHATTERBOX
if errorlevel 3 call :TOGGLE_ENV LAUNCH_RVC
if errorlevel 2 call :TOGGLE_ENV LAUNCH_AUDIO_SERVICES
if errorlevel 1 call :TOGGLE_ENV LAUNCH_ASR
goto CONFIGURE_SERVERS

:: ===============================
:: Configure Environments Menu
:: ===============================
:CONFIGURE_ENVS
cls
echo.
echo =============================================
echo Configure Environments to Install
echo =============================================
echo Toggle environments ON/OFF for setup:
echo.
echo Current Selection:
if "%INSTALL_MAIN%"=="1" (echo   [X] Main Environment   - Required) else (echo   [ ] Main Environment   - Required)
if "%INSTALL_ASR%"=="1" (echo   [X] ASR Environment    - Whisper + GLM-ASR + Parakeet) else (echo   [ ] ASR Environment    - Whisper + GLM-ASR + Parakeet)
if "%INSTALL_AUDIO%"=="1" (echo   [X] Audio Services     - Pre/Post processing) else (echo   [ ] Audio Services     - Pre/Post processing)
if "%INSTALL_RVC%"=="1" (echo   [X] RVC Environment    - Voice conversion) else (echo   [ ] RVC Environment    - Voice conversion)
if "%INSTALL_CHATTERBOX%"=="1" (echo   [X] Chatterbox         - TTS) else (echo   [ ] Chatterbox         - TTS)
if "%INSTALL_POCKET_TTS%"=="1" (echo   [X] Pocket TTS         - Lightweight TTS) else (echo   [ ] Pocket TTS         - Lightweight TTS)
if "%INSTALL_KOKORO_TTS%"=="1" (echo   [X] Kokoro TTS         - ONNX TTS) else (echo   [ ] Kokoro TTS         - ONNX TTS)
if "%INSTALL_OMNIVOICE_TTS%"=="1" (echo   [X] OmniVoice TTS      - Multilingual TTS) else (echo   [ ] OmniVoice TTS      - Multilingual TTS)
if "%INSTALL_TRAINING%"=="1" (echo   [X] Training            - Model training) else (echo   [ ] Training            - Model training)
echo.
echo [1] Toggle Main Environment   - Required
echo [2] Toggle ASR Environment    - Whisper + GLM-ASR + Parakeet
echo [3] Toggle Audio Services     - Pre/Post processing
echo [4] Toggle RVC Environment    - Voice conversion
echo [5] Toggle Chatterbox         - TTS
echo [6] Toggle Pocket TTS         - Lightweight TTS
echo [7] Toggle Kokoro TTS         - ONNX TTS
echo [8] Toggle OmniVoice TTS      - Multilingual TTS
echo [9] Toggle Training           - Model training
echo.
echo [A] Save Configuration
echo [0] Enable All
echo [R] Reset to Defaults
echo [B] Back to Setup Menu
echo =============================================
echo.

choice /C 123456789A0RB /N /M "Choose option: "
if errorlevel 13 goto UTILITIES_MENU
if errorlevel 12 goto RESET_ENV_DEFAULTS
if errorlevel 11 goto ENABLE_ALL_ENVS
if errorlevel 10 goto SAVE_ENV_CONFIG
if errorlevel 9 call :TOGGLE_ENV INSTALL_TRAINING
if errorlevel 8 call :TOGGLE_ENV INSTALL_OMNIVOICE_TTS
if errorlevel 7 call :TOGGLE_ENV INSTALL_KOKORO_TTS
if errorlevel 6 call :TOGGLE_ENV INSTALL_POCKET_TTS
if errorlevel 5 call :TOGGLE_ENV INSTALL_CHATTERBOX
if errorlevel 4 call :TOGGLE_ENV INSTALL_RVC
if errorlevel 3 call :TOGGLE_ENV INSTALL_AUDIO
if errorlevel 2 call :TOGGLE_ENV INSTALL_ASR
if errorlevel 1 call :TOGGLE_ENV INSTALL_MAIN
goto CONFIGURE_ENVS

:SAVE_SERVER_CONFIG
echo.
echo [INFO] Saving server launch configuration to %CONFIG_FILE%...
(
echo @echo off
echo :: VoiceForge Server Configuration
echo :: Generated automatically - do not edit manually
echo.
echo set "LAUNCH_ASR=%LAUNCH_ASR%"
echo set "LAUNCH_AUDIO_SERVICES=%LAUNCH_AUDIO_SERVICES%"
echo set "LAUNCH_RVC=%LAUNCH_RVC%"
echo set "LAUNCH_CHATTERBOX=%LAUNCH_CHATTERBOX%"
echo set "LAUNCH_POCKET_TTS=%LAUNCH_POCKET_TTS%"
echo set "LAUNCH_KOKORO_TTS=%LAUNCH_KOKORO_TTS%"
echo set "LAUNCH_OMNIVOICE_TTS=%LAUNCH_OMNIVOICE_TTS%"
echo set "LAUNCH_TRAINING=%LAUNCH_TRAINING%"
) > "%CONFIG_FILE%"
echo [INFO] Server launch configuration saved successfully!
echo.
pause
goto CONFIGURE_SERVERS

:ENABLE_ALL_SERVERS
set "LAUNCH_ASR=1"
set "LAUNCH_AUDIO_SERVICES=1"
set "LAUNCH_RVC=1"
set "LAUNCH_CHATTERBOX=1"
set "LAUNCH_POCKET_TTS=1"
set "LAUNCH_KOKORO_TTS=1"
set "LAUNCH_OMNIVOICE_TTS=1"
set "LAUNCH_TRAINING=1"
goto CONFIGURE_SERVERS

:DISABLE_ALL_SERVERS
set "LAUNCH_ASR=0"
set "LAUNCH_AUDIO_SERVICES=0"
set "LAUNCH_RVC=0"
set "LAUNCH_CHATTERBOX=0"
set "LAUNCH_POCKET_TTS=0"
set "LAUNCH_KOKORO_TTS=0"
set "LAUNCH_OMNIVOICE_TTS=0"
set "LAUNCH_TRAINING=0"
goto CONFIGURE_SERVERS

:RESET_SERVER_DEFAULTS
set "LAUNCH_ASR=1"
set "LAUNCH_AUDIO_SERVICES=1"
set "LAUNCH_RVC=1"
set "LAUNCH_CHATTERBOX=1"
set "LAUNCH_POCKET_TTS=1"
set "LAUNCH_KOKORO_TTS=1"
set "LAUNCH_OMNIVOICE_TTS=1"
set "LAUNCH_TRAINING=1"
if exist "%CONFIG_FILE%" del "%CONFIG_FILE%"
echo [INFO] Reset server launch defaults and removed saved config.
timeout /t 2 /nobreak >nul
goto CONFIGURE_SERVERS

:SHOW_SAVED_STATUS
set "SERVER_COUNT=0"
if "%LAUNCH_ASR%"=="1" set /a SERVER_COUNT+=1
if "%LAUNCH_AUDIO_SERVICES%"=="1" set /a SERVER_COUNT+=1
if "%LAUNCH_RVC%"=="1" set /a SERVER_COUNT+=1
if "%LAUNCH_CHATTERBOX%"=="1" set /a SERVER_COUNT+=1
if "%LAUNCH_POCKET_TTS%"=="1" set /a SERVER_COUNT+=1
if "%LAUNCH_KOKORO_TTS%"=="1" set /a SERVER_COUNT+=1
if "%LAUNCH_OMNIVOICE_TTS%"=="1" set /a SERVER_COUNT+=1
if "%LAUNCH_TRAINING%"=="1" set /a SERVER_COUNT+=1
echo Servers enabled: %SERVER_COUNT% of 8
call :SHOW_ENV_STATUS
exit /b 0

:SHOW_SERVER_STATUS
echo Current Selection:
if "%LAUNCH_ASR%"=="1" (echo   [X] ASR Server         - Whisper + GLM-ASR + Parakeet) else (echo   [ ] ASR Server         - Whisper + GLM-ASR + Parakeet)
if "%LAUNCH_AUDIO_SERVICES%"=="1" (echo   [X] Audio Services     - Pre/Post processing) else (echo   [ ] Audio Services     - Pre/Post processing)
if "%LAUNCH_RVC%"=="1" (echo   [X] RVC Server         - Voice conversion) else (echo   [ ] RVC Server         - Voice conversion)
if "%LAUNCH_CHATTERBOX%"=="1" (echo   [X] Chatterbox         - TTS) else (echo   [ ] Chatterbox         - TTS)
if "%LAUNCH_POCKET_TTS%"=="1" (echo   [X] Pocket TTS         - Lightweight TTS) else (echo   [ ] Pocket TTS         - Lightweight TTS)
if "%LAUNCH_KOKORO_TTS%"=="1" (echo   [X] Kokoro TTS         - ONNX TTS) else (echo   [ ] Kokoro TTS         - ONNX TTS)
if "%LAUNCH_OMNIVOICE_TTS%"=="1" (echo   [X] OmniVoice TTS      - Multilingual TTS) else (echo   [ ] OmniVoice TTS      - Multilingual TTS)
if "%LAUNCH_TRAINING%"=="1" (echo   [X] Training Server    - Model training) else (echo   [ ] Training Server    - Model training)
exit /b 0

:SHOW_ENV_CONFIG_STATUS
echo Current Selection:
if "%INSTALL_MAIN%"=="1" (echo   [X] Main Environment   - Required) else (echo   [ ] Main Environment   - Required)
if "%INSTALL_ASR%"=="1" (echo   [X] ASR Environment    - Whisper + GLM-ASR + Parakeet) else (echo   [ ] ASR Environment    - Whisper + GLM-ASR + Parakeet)
if "%INSTALL_AUDIO%"=="1" (echo   [X] Audio Services     - Pre/Post processing) else (echo   [ ] Audio Services     - Pre/Post processing)
if "%INSTALL_RVC%"=="1" (echo   [X] RVC Environment    - Voice conversion) else (echo   [ ] RVC Environment    - Voice conversion)
if "%INSTALL_CHATTERBOX%"=="1" (echo   [X] Chatterbox         - TTS) else (echo   [ ] Chatterbox         - TTS)
if "%INSTALL_POCKET_TTS%"=="1" (echo   [X] Pocket TTS         - Lightweight TTS) else (echo   [ ] Pocket TTS         - Lightweight TTS)
if "%INSTALL_KOKORO_TTS%"=="1" (echo   [X] Kokoro TTS         - ONNX TTS) else (echo   [ ] Kokoro TTS         - ONNX TTS)
if "%INSTALL_OMNIVOICE_TTS%"=="1" (echo   [X] OmniVoice TTS      - Multilingual TTS) else (echo   [ ] OmniVoice TTS      - Multilingual TTS)
if "%INSTALL_TRAINING%"=="1" (echo   [X] Training            - Model training) else (echo   [ ] Training            - Model training)
exit /b 0

:SHOW_ENV_STATUS
set "ENV_COUNT=0"
if "%INSTALL_MAIN%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_ASR%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_AUDIO%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_RVC%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_CHATTERBOX%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_POCKET_TTS%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_KOKORO_TTS%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_OMNIVOICE_TTS%"=="1" set /a ENV_COUNT+=1
if "%INSTALL_TRAINING%"=="1" set /a ENV_COUNT+=1
echo Environments: %ENV_COUNT% of 9 selected
echo (Press 2 to configure)
exit /b 0

:TOGGLE_ENV
set "VAR_NAME=%~1"
if "!%VAR_NAME%!"=="1" (
    set "%VAR_NAME%=0"
) else (
    set "%VAR_NAME%=1"
)
exit /b 0

:SAVE_ENV_CONFIG
echo.
echo [INFO] Saving environment configuration to %ENV_CONFIG_FILE%...
(
echo @echo off
echo :: VoiceForge Environment Configuration
echo :: Generated automatically - do not edit manually
echo.
echo set "INSTALL_MAIN=%INSTALL_MAIN%"
echo set "INSTALL_ASR=%INSTALL_ASR%"
echo set "INSTALL_AUDIO=%INSTALL_AUDIO%"
echo set "INSTALL_RVC=%INSTALL_RVC%"
echo set "INSTALL_CHATTERBOX=%INSTALL_CHATTERBOX%"
echo set "INSTALL_POCKET_TTS=%INSTALL_POCKET_TTS%"
echo set "INSTALL_KOKORO_TTS=%INSTALL_KOKORO_TTS%"
echo set "INSTALL_OMNIVOICE_TTS=%INSTALL_OMNIVOICE_TTS%"
echo set "INSTALL_TRAINING=%INSTALL_TRAINING%"
) > "%ENV_CONFIG_FILE%"
echo [INFO] Environment configuration saved successfully!
echo.
pause
goto UTILITIES_MENU

:ENABLE_ALL_ENVS
set "INSTALL_MAIN=1"
set "INSTALL_ASR=1"
set "INSTALL_AUDIO=1"
set "INSTALL_RVC=1"
set "INSTALL_CHATTERBOX=1"
set "INSTALL_POCKET_TTS=1"
set "INSTALL_KOKORO_TTS=1"
set "INSTALL_OMNIVOICE_TTS=1"
set "INSTALL_TRAINING=1"
goto CONFIGURE_ENVS

:DISABLE_ALL_ENVS_EXCEPT_MAIN
set "INSTALL_MAIN=1"
set "INSTALL_ASR=0"
set "INSTALL_AUDIO=0"
set "INSTALL_RVC=0"
set "INSTALL_CHATTERBOX=0"
set "INSTALL_POCKET_TTS=0"
set "INSTALL_KOKORO_TTS=0"
set "INSTALL_OMNIVOICE_TTS=0"
set "INSTALL_TRAINING=0"
goto CONFIGURE_ENVS

:RESET_ENV_DEFAULTS
set "INSTALL_MAIN=1"
set "INSTALL_ASR=1"
set "INSTALL_AUDIO=1"
set "INSTALL_RVC=1"
set "INSTALL_CHATTERBOX=1"
set "INSTALL_POCKET_TTS=1"
set "INSTALL_KOKORO_TTS=1"
set "INSTALL_OMNIVOICE_TTS=1"
set "INSTALL_TRAINING=0"
if exist "%ENV_CONFIG_FILE%" del "%ENV_CONFIG_FILE%"
echo [INFO] Reset to defaults and removed saved config.
timeout /t 2 /nobreak >nul
goto CONFIGURE_ENVS

:: ===============================
:: Setup Menu
:: ===============================
:UTILITIES_MENU
cls
echo.
echo =============================================
echo VoiceForge Setup
echo =============================================
echo [1] Install/Update Selected
echo [2] Configure Environments
echo [3] Delete All Environments
echo [4] Back to Menu
echo =============================================
call :SHOW_ENV_STATUS
echo.

choice /C 1234 /N /M "Choose [1-4]: "
if errorlevel 4 goto MENU
if errorlevel 3 goto DELETE_ALL_ENVS
if errorlevel 2 goto CONFIGURE_ENVS
if errorlevel 1 goto INSTALL_UPDATE_SELECTED
goto UTILITIES_MENU

:: ===============================
:: Install/Update Selected Environments
:: ===============================
:INSTALL_UPDATE_SELECTED
echo.
echo [INFO] Installing/Updating selected environments...
echo.

if "%INSTALL_MAIN%"=="1" (
    echo [INFO] Installing/Updating Main environment...
    call "%~dp0app\install\install_main.bat"
    if errorlevel 1 echo [WARN] Main environment setup had issues
)

if "%INSTALL_ASR%"=="1" (
    echo [INFO] Installing/Updating ASR environment...
    call "%~dp0app\install\install_asr.bat"
    if errorlevel 1 echo [WARN] ASR environment setup had issues
)

if "%INSTALL_AUDIO%"=="1" (
    echo [INFO] Installing/Updating Audio Services environment...
    call "%~dp0app\install\install_audio_services.bat"
    if errorlevel 1 echo [WARN] Audio Services environment setup had issues
)

if "%INSTALL_RVC%"=="1" (
    echo [INFO] Installing/Updating RVC environment...
    call "%~dp0app\install\install_rvc.bat"
    if errorlevel 1 echo [WARN] RVC environment setup had issues
)

if "%INSTALL_CHATTERBOX%"=="1" (
    echo [INFO] Installing/Updating Chatterbox environment...
    call "%~dp0app\install\install_chatterbox.bat"
    if errorlevel 1 echo [WARN] Chatterbox environment setup had issues
)

if "%INSTALL_POCKET_TTS%"=="1" (
    echo [INFO] Installing/Updating Pocket TTS environment...
    call "%~dp0app\install\install_pocket_tts.bat"
    if errorlevel 1 echo [WARN] Pocket TTS environment setup had issues
)

if "%INSTALL_KOKORO_TTS%"=="1" (
    echo [INFO] Installing/Updating Kokoro TTS environment...
    call "%~dp0app\install\install_kokoro_tts.bat"
    if errorlevel 1 echo [WARN] Kokoro TTS environment setup had issues
)

if "%INSTALL_OMNIVOICE_TTS%"=="1" (
    echo [INFO] Installing/Updating OmniVoice TTS environment...
    call "%~dp0app\install\install_omnivoice.bat"
    if errorlevel 1 echo [WARN] OmniVoice TTS environment setup had issues
)

if "%INSTALL_TRAINING%"=="1" (
    echo [INFO] Installing/Updating Training environment...
    call "%~dp0app\install\install_chatterbox_train.bat"
    if errorlevel 1 echo [WARN] Training environment setup had issues
)

echo.
echo [INFO] Install/Update complete!
pause
goto UTILITIES_MENU

:: ===============================
:: Training Menu
:: ===============================
:TRAINING_MENU
cls
echo.
echo =============================================
echo TTS Training Setup
echo =============================================
echo [1] Install Chatterbox Training (Fine-tuning)
echo [2] Launch Training Server
echo [3] Back to Setup
echo =============================================
echo.

choice /C 123 /N /M "Choose [1-3]: "
if errorlevel 3 goto UTILITIES_MENU
if errorlevel 2 goto LAUNCH_TRAINING_SERVER
if errorlevel 1 goto INSTALL_CHATTERBOX_TRAINING
goto TRAINING_MENU

:INSTALL_CHATTERBOX_TRAINING
echo.
echo [INFO] Installing Chatterbox fine-tuning environment...
call "%~dp0app\install\install_chatterbox_train.bat"
if errorlevel 1 (
    echo [ERROR] Chatterbox training environment setup failed.
) else (
    echo [INFO] Chatterbox training environment installed successfully!
)
pause
goto TRAINING_MENU

:LAUNCH_TRAINING_SERVER
echo.
echo [INFO] Launching Training Server...
if not exist "%~dp0app\launch\launch_training_server.bat" (
    echo [ERROR] Training server launcher not found!
    pause
    goto TRAINING_MENU
)
start "VoiceForge Training Server" cmd /k "%~dp0app\launch\launch_training_server.bat"
echo [INFO] Training server starting on port %TRAINING_SERVER_PORT%...
timeout /t 2 /nobreak >nul
pause
goto TRAINING_MENU

:: ===============================
:: Helper: Launch Background Services
:: ===============================
:LAUNCH_SERVICES
echo.
echo [INFO] Starting background services...

if not defined CONDA_BASE (
echo [WARN] CONDA_BASE not set - skipping optional services
exit /b 0
)

set "SERVICES_LAUNCHED=0"

:: Launch services with small delays to avoid conda temp file race condition
if "%LAUNCH_ASR%"=="1" (
    if exist "%CONDA_BASE%\envs\%ASR_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_asr_server.bat" (
        start "VoiceForge ASR" cmd /k "%~dp0app\launch\launch_asr_server.bat"
        echo [INFO] ASR Server starting on port %ASR_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [WARN] ASR Server not available - environment or launcher missing
    )
) else (
    echo [SKIP] ASR Server - disabled by user
)

if "%LAUNCH_AUDIO_SERVICES%"=="1" (
    if exist "%CONDA_BASE%\envs\%AUDIO_SERVICES_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_audio_services_server.bat" (
        start "VoiceForge Audio Services" cmd /k "%~dp0app\launch\launch_audio_services_server.bat"
        echo [INFO] Audio Services starting on port %AUDIO_SERVICES_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [WARN] Audio Services not available - environment or launcher missing
    )
) else (
    echo [SKIP] Audio Services - disabled by user
)

if "%LAUNCH_RVC%"=="1" (
    if exist "%CONDA_BASE%\envs\%RVC_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_rvc_server.bat" (
        start "VoiceForge RVC" cmd /k "%~dp0app\launch\launch_rvc_server.bat"
        echo [INFO] RVC Server starting on port %RVC_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [WARN] RVC Server not available - environment or launcher missing
    )
) else (
    echo [SKIP] RVC Server - disabled by user
)

if "%LAUNCH_CHATTERBOX%"=="1" (
    if exist "%CONDA_BASE%\envs\%CHATTERBOX_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_chatterbox_server.bat" (
        start "VoiceForge Chatterbox" cmd /k "%~dp0app\launch\launch_chatterbox_server.bat"
        echo [INFO] Chatterbox Server starting on port %CHATTERBOX_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [WARN] Chatterbox Server not available - environment or launcher missing
    )
) else (
    echo [SKIP] Chatterbox Server - disabled by user
)

if "%LAUNCH_POCKET_TTS%"=="1" (
    if exist "%CONDA_BASE%\envs\%POCKET_TTS_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_pocket_tts_server.bat" (
        start "VoiceForge Pocket TTS" cmd /k "%~dp0app\launch\launch_pocket_tts_server.bat"
        echo [INFO] Pocket TTS Server starting on port %POCKET_TTS_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [WARN] Pocket TTS Server not available - environment or launcher missing
    )
) else (
    echo [SKIP] Pocket TTS Server - disabled by user
)

if "%LAUNCH_KOKORO_TTS%"=="1" (
    if exist "%CONDA_BASE%\envs\%KOKORO_TTS_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_kokoro_tts_server.bat" (
        start "VoiceForge Kokoro TTS" cmd /k "%~dp0app\launch\launch_kokoro_tts_server.bat"
        echo [INFO] Kokoro TTS Server starting on port %KOKORO_TTS_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [WARN] Kokoro TTS Server not available - environment or launcher missing
    )
) else (
    echo [SKIP] Kokoro TTS Server - disabled by user
)

if "%LAUNCH_OMNIVOICE_TTS%"=="1" (
    if exist "%CONDA_BASE%\envs\%OMNIVOICE_TTS_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_omnivoice_server.bat" (
        start "VoiceForge OmniVoice TTS" cmd /k "%~dp0app\launch\launch_omnivoice_server.bat"
        echo [INFO] OmniVoice TTS Server starting on port %OMNIVOICE_TTS_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [WARN] OmniVoice TTS Server not available - environment or launcher missing
    )
) else (
    echo [SKIP] OmniVoice TTS Server - disabled by user
)

if "%LAUNCH_TRAINING%"=="1" (
    if exist "%CONDA_BASE%\envs\%CHATTERBOX_TRAIN_ENV_NAME%\python.exe" if exist "%~dp0app\launch\launch_training_server.bat" (
        start "VoiceForge Training" cmd /k "%~dp0app\launch\launch_training_server.bat"
        echo [INFO] Training Server starting on port %TRAINING_SERVER_PORT%...
        set /a SERVICES_LAUNCHED+=1
    ) else (
        echo [WARN] Training Server not available - environment or launcher missing
    )
) else (
    echo [SKIP] Training Server - disabled by user
)

echo [INFO] Services launching... (%SERVICES_LAUNCHED% services started)
exit /b 0

:: ===============================
:: UPDATE ALL ENVIRONMENTS
:: ===============================
:UPDATE_ALL
echo.
echo [INFO] Updating all environments...
echo.

:: Update main env if exists
"%CONDA_EXE%" env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %CONDA_ENV_NAME%...
    call "%~dp0app\install\install_main.bat"
    if errorlevel 1 echo [WARN] Main environment update had issues
) else (
    echo [SKIP] %CONDA_ENV_NAME% not installed
)

:: Update ASR env if exists (unified Whisper + GLM-ASR)
"%CONDA_EXE%" env list | findstr /C:"%ASR_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %ASR_ENV_NAME%...
    call "%~dp0app\install\install_asr.bat"
    if errorlevel 1 echo [WARN] ASR environment update had issues
) else (
    echo [SKIP] %ASR_ENV_NAME% not installed
)

:: Update Audio Services env if exists
"%CONDA_EXE%" env list | findstr /C:"%AUDIO_SERVICES_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %AUDIO_SERVICES_ENV_NAME%...
    call "%~dp0app\install\install_audio_services.bat"
    if errorlevel 1 echo [WARN] Audio Services environment update had issues
) else (
    echo [SKIP] %AUDIO_SERVICES_ENV_NAME% not installed
)

:: Update RVC env if exists
"%CONDA_EXE%" env list | findstr /C:"%RVC_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %RVC_ENV_NAME%...
    call "%~dp0app\install\install_rvc.bat"
    if errorlevel 1 echo [WARN] RVC environment update had issues
) else (
    echo [SKIP] %RVC_ENV_NAME% not installed
)

:: Update Chatterbox env if exists
"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %CHATTERBOX_ENV_NAME%...
    call "%~dp0app\install\install_chatterbox.bat"
    if errorlevel 1 echo [WARN] Chatterbox environment update had issues
) else (
    echo [SKIP] %CHATTERBOX_ENV_NAME% not installed
)

:: Update Pocket TTS env if exists
"%CONDA_EXE%" env list | findstr /C:"%POCKET_TTS_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %POCKET_TTS_ENV_NAME%...
    call "%~dp0app\install\install_pocket_tts.bat"
    if errorlevel 1 echo [WARN] Pocket TTS environment update had issues
) else (
    echo [SKIP] %POCKET_TTS_ENV_NAME% not installed
)

:: Update Kokoro TTS env if exists
"%CONDA_EXE%" env list | findstr /C:"%KOKORO_TTS_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %KOKORO_TTS_ENV_NAME%...
    call "%~dp0app\install\install_kokoro_tts.bat"
    if errorlevel 1 echo [WARN] Kokoro TTS environment update had issues
) else (
    echo [SKIP] %KOKORO_TTS_ENV_NAME% not installed
)

:: Update OmniVoice TTS env if exists
"%CONDA_EXE%" env list | findstr /C:"%OMNIVOICE_TTS_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Updating %OMNIVOICE_TTS_ENV_NAME%...
    call "%~dp0app\install\install_omnivoice.bat"
    if errorlevel 1 echo [WARN] OmniVoice TTS environment update had issues
) else (
    echo [SKIP] %OMNIVOICE_TTS_ENV_NAME% not installed
)

echo.
echo [INFO] Update complete!
pause
goto UTILITIES_MENU

:: ===============================
:: INSTALL ALL ENVIRONMENTS
:: ===============================
:INSTALL_ALL_ENVS
echo.
echo [INFO] Installing all environments...
echo [INFO] This will setup: %CONDA_ENV_NAME%, %ASR_ENV_NAME%, %RVC_ENV_NAME%, %AUDIO_SERVICES_ENV_NAME%, %CHATTERBOX_ENV_NAME%, %POCKET_TTS_ENV_NAME%, %KOKORO_TTS_ENV_NAME%, %OMNIVOICE_TTS_ENV_NAME%
echo.
pause

:: Install main env
call "%~dp0app\install\install_main.bat"
if errorlevel 1 (
    echo [ERROR] Main environment setup failed.
    pause
    goto UTILITIES_MENU
)

:: Install Audio Services env (preprocess + postprocess + background audio)
call "%~dp0app\install\install_audio_services.bat"
if errorlevel 1 (
    echo [WARN] Audio Services dependency install had issues.
)

:: Install unified ASR env (Whisper + GLM-ASR in one environment)
call "%~dp0app\install\install_asr.bat"
if errorlevel 1 (
    echo [WARN] ASR environment setup had issues (optional component).
)

:: Install RVC env
call "%~dp0app\install\install_rvc.bat"
if errorlevel 1 (
    echo [ERROR] RVC environment setup failed.
    pause
    goto UTILITIES_MENU
)

:: Install Chatterbox-Turbo env
call "%~dp0app\install\install_chatterbox.bat"
if errorlevel 1 (
    echo [ERROR] Chatterbox-Turbo environment setup failed.
    pause
    goto UTILITIES_MENU
)

:: Install Pocket TTS env
call "%~dp0app\install\install_pocket_tts.bat"
if errorlevel 1 (
    echo [WARN] Pocket TTS environment setup had issues (optional component).
)

:: Install Kokoro TTS env
call "%~dp0app\install\install_kokoro_tts.bat"
if errorlevel 1 (
    echo [WARN] Kokoro TTS environment setup had issues (optional component).
)

:: Install OmniVoice TTS env
call "%~dp0app\install\install_omnivoice.bat"
if errorlevel 1 (
    echo [WARN] OmniVoice TTS environment setup had issues (optional component).
)

echo.
echo [INFO] All environments installed successfully!
pause
goto UTILITIES_MENU

:: ===============================
:: DELETE ALL ENVIRONMENTS
:: ===============================
:DELETE_ALL_ENVS
echo.
echo =============================================
echo   WARNING: This will delete ALL environments!
echo =============================================
echo.
echo   Environments to be deleted:
echo     - %CONDA_ENV_NAME%
echo     - %ASR_ENV_NAME%
echo     - %AUDIO_SERVICES_ENV_NAME%
echo     - %RVC_ENV_NAME%
echo     - %CHATTERBOX_ENV_NAME%
echo     - %POCKET_TTS_ENV_NAME%
echo     - %KOKORO_TTS_ENV_NAME%
echo     - %OMNIVOICE_TTS_ENV_NAME%
echo     - %CHATTERBOX_TRAIN_ENV_NAME% (if exists)
echo.
echo   Press Y to confirm, N to cancel.
echo.

choice /C YN /N /M "Delete all environments? [Y/N]: "
if errorlevel 2 goto UTILITIES_MENU

echo.
echo [INFO] Deleting environments...

:: Delete main env
"%CONDA_EXE%" env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %CONDA_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%CONDA_ENV_NAME%" -y
)

:: Delete ASR env (unified Whisper + GLM-ASR)
"%CONDA_EXE%" env list | findstr /C:"%ASR_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %ASR_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%ASR_ENV_NAME%" -y
)

:: Delete Audio Services env
"%CONDA_EXE%" env list | findstr /C:"%AUDIO_SERVICES_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %AUDIO_SERVICES_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%AUDIO_SERVICES_ENV_NAME%" -y
)

:: Delete RVC env
"%CONDA_EXE%" env list | findstr /C:"%RVC_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %RVC_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%RVC_ENV_NAME%" -y
)

:: Delete Chatterbox env
"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %CHATTERBOX_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%CHATTERBOX_ENV_NAME%" -y
)

:: Delete Pocket TTS env
"%CONDA_EXE%" env list | findstr /C:"%POCKET_TTS_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %POCKET_TTS_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%POCKET_TTS_ENV_NAME%" -y
)

:: Delete Kokoro TTS env
"%CONDA_EXE%" env list | findstr /C:"%KOKORO_TTS_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %KOKORO_TTS_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%KOKORO_TTS_ENV_NAME%" -y
)

:: Delete OmniVoice TTS env
"%CONDA_EXE%" env list | findstr /C:"%OMNIVOICE_TTS_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %OMNIVOICE_TTS_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%OMNIVOICE_TTS_ENV_NAME%" -y
)

:: Delete Training envs (if they exist)
"%CONDA_EXE%" env list | findstr /C:"%CHATTERBOX_TRAIN_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing %CHATTERBOX_TRAIN_ENV_NAME%...
    "%CONDA_EXE%" env remove -n "%CHATTERBOX_TRAIN_ENV_NAME%" -y
)

echo.
echo [INFO] All environments deleted!
pause
goto UTILITIES_MENU

:: ===============================
:: RUN - Launch App (uses current LAUNCH_* settings)
:: ===============================
:RUN
echo [DEBUG] Starting RUN section...
call :ACTIVATE_ENV "%CONDA_ENV_NAME%"
if errorlevel 1 (
echo [ERROR] Failed to activate environment "%CONDA_ENV_NAME%"
echo [INFO] Please run Setup ^> Install/Update Selected to setup the environment
    pause
    goto MENU
)

echo [DEBUG] Environment activated successfully
echo [DEBUG] Verifying Python is accessible...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found or not working after activation
    pause
    goto MENU
)

set "PYTHONPATH=%CUSTOM_DEPS%;%PYTHONPATH%"

:: Set server URLs for main app
set "ASR_SERVER_URL=http://127.0.0.1:%ASR_SERVER_PORT%"
set "RVC_SERVER_URL=http://127.0.0.1:%RVC_SERVER_PORT%"
set "CHATTERBOX_SERVER_URL=http://127.0.0.1:%CHATTERBOX_SERVER_PORT%"
set "POCKET_TTS_SERVER_URL=http://127.0.0.1:%POCKET_TTS_SERVER_PORT%"
set "OMNIVOICE_TTS_SERVER_URL=http://127.0.0.1:%OMNIVOICE_TTS_SERVER_PORT%"
set "TRAINING_SERVER_URL=http://127.0.0.1:%TRAINING_SERVER_PORT%"

:: Launch ASR server in background (if env exists)
echo [DEBUG] About to launch background services...
echo [DEBUG] Press Ctrl+C now if you want to skip background services
timeout /t 1 /nobreak >nul 2>&1

echo [DEBUG] Calling LAUNCH_SERVICES function...
call :LAUNCH_SERVICES 2>&1
set "SERVICES_EXIT=%ERRORLEVEL%"
echo [DEBUG] LAUNCH_SERVICES returned with exit code: %SERVICES_EXIT%

if %SERVICES_EXIT% neq 0 (
    echo.
    echo =============================================
    echo [WARN] Background services launch had issues
    echo [WARN] Exit code: %SERVICES_EXIT%
    echo =============================================
    echo [WARN] This is usually OK - services are optional
    echo [INFO] Continuing with main application...
    echo [INFO] You can start services manually if needed
    echo.
    echo [DEBUG] Press any key to continue to main app...
    pause >nul
) else (
    echo [DEBUG] Background services launch completed successfully
    echo [DEBUG] Services should be running in background windows
)

echo [DEBUG] Checking if main.py exists...
cd /d "%~dp0"
if not exist "app\util\main.py" (
    echo [ERROR] main.py not found in directory: %CD%
    echo [ERROR] Expected location: %~dp0app\util\main.py
    pause
    goto MENU
)
echo [DEBUG] Found main.py in: %CD%\app\util

echo [INFO] Launching VoiceForge...
echo [INFO] ASR Server URL: %ASR_SERVER_URL% (Whisper + GLM-ASR)
echo [INFO] RVC Server URL: %RVC_SERVER_URL%
echo [INFO] Chatterbox Server URL: %CHATTERBOX_SERVER_URL%
echo [INFO] Pocket TTS Server URL: %POCKET_TTS_SERVER_URL%
echo [INFO] OmniVoice TTS Server URL: %OMNIVOICE_TTS_SERVER_URL%
echo [INFO] Training Server URL: %TRAINING_SERVER_URL%
echo [DEBUG] Python path: 
where python
echo [DEBUG] Current directory: %CD%
echo [DEBUG] Running: python -X faulthandler -u app\util\main.py
echo.

:: Set window title for easier identification
title VoiceForge - Running...

:: Create error log file path for reference
set "ERROR_LOG=%~dp0voiceforge_error.log"
echo [INFO] Starting VoiceForge...
echo [INFO] If errors occur, they will be displayed below.
echo [INFO] Error log backup: %ERROR_LOG%
echo.

:: Run Python with both stdout and stderr visible
:: Capture stderr explicitly to ensure errors are shown
python -X faulthandler -u app\util\main.py 2>&1
set "PYTHON_EXIT=%ERRORLEVEL%"

if %PYTHON_EXIT% neq 0 (
    echo.
    echo =============================================
    echo [ERROR] VoiceForge crashed with exit code %PYTHON_EXIT%!
    echo =============================================
    echo [INFO] Check the error messages above for details.
echo [INFO] Common issues:
echo - Missing Python packages (run Setup ^> Install/Update Selected)
    echo   - Missing or corrupted environment
    echo   - Port already in use (check if another instance is running)
    echo.
    echo [INFO] Press any key to return to menu...
    pause >nul
) else (
    echo.
    echo [INFO] VoiceForge exited normally.
    pause
)

goto MENU

:: ===============================
:: RUN_SERVER - Launch API (uses current LAUNCH_* settings)
:: ===============================
:RUN_SERVER
call :ACTIVATE_ENV "%CONDA_ENV_NAME%"
if errorlevel 1 goto MENU

set "PYTHONPATH=%CUSTOM_DEPS%;%PYTHONPATH%"

:: Set server URLs
set "ASR_SERVER_URL=http://127.0.0.1:%ASR_SERVER_PORT%"
set "RVC_SERVER_URL=http://127.0.0.1:%RVC_SERVER_PORT%"
set "CHATTERBOX_SERVER_URL=http://127.0.0.1:%CHATTERBOX_SERVER_PORT%"
set "POCKET_TTS_SERVER_URL=http://127.0.0.1:%POCKET_TTS_SERVER_PORT%"
set "OMNIVOICE_TTS_SERVER_URL=http://127.0.0.1:%OMNIVOICE_TTS_SERVER_PORT%"
set "TRAINING_SERVER_URL=http://127.0.0.1:%TRAINING_SERVER_PORT%"

:: Launch selected servers in background
call :LAUNCH_SERVICES

echo [INFO] Launching API server on port 8888...
echo [INFO] ASR Server URL: %ASR_SERVER_URL% (Whisper + GLM-ASR)
echo [INFO] RVC Server URL: %RVC_SERVER_URL%
echo [INFO] Chatterbox Server URL: %CHATTERBOX_SERVER_URL%
echo [INFO] Pocket TTS Server URL: %POCKET_TTS_SERVER_URL%
echo [INFO] OmniVoice TTS Server URL: %OMNIVOICE_TTS_SERVER_URL%
echo [INFO] Training Server URL: %TRAINING_SERVER_URL%
python -X faulthandler -u "app\servers\main_server.py" --port 8888

echo.
pause
goto MENU

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
    pause
    exit /b 1
)

set "ENV_DIR=%CONDA_BASE%\envs\%TARGET_ENV%"
if not exist "%ENV_DIR%" (
    echo [ERROR] Environment "%TARGET_ENV%" not found. Please run setup first.
    pause
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%PATH%"
set "CONDA_DEFAULT_ENV=%TARGET_ENV%"
set "CONDA_PREFIX=%ENV_DIR%"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found after activation.
    pause
    exit /b 1
)
exit /b 0

:: ===============================
:: END
:: ===============================
:END
echo [INFO] Goodbye!
endlocal
exit /b 0
