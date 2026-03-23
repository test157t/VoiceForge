# VoiceForge

## Servers

VoiceForge is split into multiple microservices. Each service has its own install and launch script:

- Main API/UI: `app/install/install_main.bat`, `Voice_Forge.bat`
- Chatterbox TTS: `app/install/install_chatterbox.bat`, `app/launch/launch_chatterbox_server.bat`
- RVC: `app/install/install_rvc.bat`, `app/launch/launch_rvc_server.bat`
- Audio Services: `app/install/install_audio_services.bat`, `app/launch/launch_audio_services_server.bat`
- ASR (Whisper + GLM-ASR + Parakeet): `app/install/install_asr.bat`, `app/launch/launch_asr_server.bat`
