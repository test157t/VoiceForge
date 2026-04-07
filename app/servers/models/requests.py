"""
Unified request/response models for VoiceForge TTS API.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from .params import (
    RVCParams,
    PostProcessParams,
    BackgroundParams,
)


class TTSRequest(BaseModel):
    """
    Unified TTS request model for Chatterbox TTS with optional RVC and post-processing.
    """
    # Required
    input: str = Field(..., description="Text to convert to speech")
    
    # Request tracking (for cancellation support)
    request_id: Optional[str] = Field(default=None, description="Client-provided request ID for cancellation")
    
    # TTS settings
    tts_mode: Literal["chunked", "streaming"] = Field(
        default="chunked",
        description="Generation mode: 'chunked' (wait for complete) or 'streaming' (progressive)"
    )
    tts_backend: Literal["chatterbox", "pocket_tts", "kokoro", "omnivoice"] = Field(
        default="chatterbox",
        description="TTS backend: chatterbox, pocket_tts, kokoro, or omnivoice"
    )
    tts_batch_tokens: int = Field(default=100, description="Max tokens per TTS batch")
    tts_token_method: str = Field(default="tiktoken", description="Token counting method")
    chatterbox_prompt_audio: Optional[str] = Field(default=None, description="Reference audio path for voice cloning")
    chatterbox_seed: int = Field(default=0, description="Random seed (0 = random)")
    
    # Pocket TTS settings
    pocket_tts_voice: str = Field(
        default="alba",
        description="Voice name or path to audio prompt for Pocket TTS cloning"
    )

    # Kokoro TTS settings
    kokoro_voice: str = Field(
        default="af_sarah",
        description="Voice preset for Kokoro TTS v1.0 (e.g., af_sarah, am_michael, bf_emma, bm_george)"
    )

    # OmniVoice settings
    omnivoice_voice: str = Field(
        default="auto",
        description="OmniVoice voice mode: auto, instruct:<attributes>, or path to reference audio"
    )
    omnivoice_ref_text: Optional[str] = Field(
        default=None,
        description="Optional transcript for OmniVoice reference audio"
    )
    omnivoice_ref_asr_model: Optional[str] = Field(
        default=None,
        description="ASR model to use when transcribing OmniVoice reference audio (e.g. glm-asr-nano, parakeet-tdt-0.6b-v3)"
    )

    # Output format
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3", 
        description="Audio format for response"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed multiplier")
    
    # Pipeline control
    enable_rvc: Optional[bool] = Field(default=None, description="Enable RVC (defaults to config)")
    enable_post: Optional[bool] = Field(default=None, description="Enable post-processing (defaults to config)")
    enable_background: Optional[bool] = Field(default=None, description="Enable background blending")
    
    # RVC parameters
    rvc_model: Optional[str] = Field(default=None, description="RVC model name")
    pitch_algo: str = Field(default="rmvpe+", description="Pitch algorithm")
    pitch_level: int = Field(default=0, description="Pitch adjustment level")
    index_influence: float = Field(default=0.75, ge=0.0, le=1.0, description="Index influence")
    respiration_median_filtering: int = Field(default=3, description="Respiration median filtering")
    envelope_ratio: float = Field(default=0.25, description="Envelope ratio")
    consonant_breath_protection: float = Field(default=0.33, description="Consonant breath protection")
    
    # Post-processing parameters (all effects OFF by default - 0 = disabled)
    # NOTE: These must match exactly what the VoiceForge UI exposes
    highpass: float = Field(default=0, description="Highpass filter frequency (0=off)")
    lowpass: float = Field(default=0, description="Lowpass filter frequency (0=off)")
    bass_freq: float = Field(default=100, description="Bass frequency")
    bass_gain: float = Field(default=0, description="Bass gain (0=off)")
    treble_freq: float = Field(default=8000, description="Treble frequency")
    treble_gain: float = Field(default=0, description="Treble gain (0=off)")
    reverb_delay: float = Field(default=0, description="Reverb delay ms (0=off)")
    reverb_decay: float = Field(default=0, description="Reverb decay (0=off)")
    crystalizer: float = Field(default=0, description="Crystalizer amount (0=off)")
    deesser: float = Field(default=0, description="De-esser amount (0=off)")
    
    # Spatial Audio (8D)
    audio_8d_enabled: bool = Field(default=False, description="Enable spatial audio effect")
    audio_8d_mode: str = Field(default="rotate", description="Position mode: center, extreme, sweep, rotate, static, static_right")
    audio_8d_speed: float = Field(default=0.1, description="Movement speed (Hz)")
    audio_8d_depth: float = Field(default=180.0, description="Arc in degrees (180=L↔R, 360=full)")
    audio_8d_distance: float = Field(default=0.3, description="Distance (0=in ear, 1=far)")
    audio_8d_quality: str = Field(default="balanced", description="Quality preset: fast, balanced, ultra")
    audio_8d_itd: bool = Field(default=True, description="Interaural Time Difference")
    audio_8d_proximity: bool = Field(default=True, description="Bass boost when close")
    audio_8d_crossfeed: bool = Field(default=True, description="Natural bleed between ears")
    audio_8d_micro_movements: bool = Field(default=True, description="Subtle organic variation")
    audio_8d_speech_aware: bool = Field(default=True, description="Transitions at speech pauses")
    
    # Pitch Shift
    pitch_shift_enabled: bool = Field(default=False, description="Enable pitch shift")
    pitch_shift_semitones: int = Field(default=0, description="Pitch shift in semitones")
    
    # ASMR Enhancement
    asmr_enabled: bool = Field(default=False, description="Enable ASMR whisper enhancement")
    asmr_tingles: int = Field(default=60, ge=0, le=100, description="2-8kHz tingle zone enhancement")
    asmr_breathiness: int = Field(default=65, ge=0, le=100, description="High frequency air and breath sounds")
    asmr_crispness: int = Field(default=55, ge=0, le=100, description="Mouth sounds, consonants, crisp detail")
    
    # Background audio
    bg_files: List[str] = Field(default_factory=list, description="Background audio files")
    bg_volumes: List[float] = Field(default_factory=list, description="Background volumes")
    bg_delays: List[float] = Field(default_factory=list, description="Background delays (seconds)")
    bg_fade_ins: List[float] = Field(default_factory=list, description="Background fade in durations (seconds)")
    bg_fade_outs: List[float] = Field(default_factory=list, description="Background fade out durations (seconds)")
    background_volume: float = Field(default=0.3, ge=0.0, le=1.0, description="Background volume")
    background_delay: float = Field(default=0.0, ge=0.0, description="Background delay (seconds)")
    server_mix_background: bool = Field(default=True, description="Mix background on server (vs client-side)")
    main_audio_volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Main audio volume for blending (legacy)")
    output_volume: float = Field(default=1.0, ge=0.0, le=3.0, description="Final output volume (applied to saved files)")
    use_config_bg_tracks: bool = Field(default=False, description="Use bg_tracks from config")
    save_output: bool = Field(default=False, description="Save final output to server output directory")
    
    def get_rvc_params(self) -> RVCParams:
        """Extract RVC parameters as dataclass."""
        return RVCParams(
            pitch_algo=self.pitch_algo,
            pitch_lvl=self.pitch_level,
            index_influence=self.index_influence,
            respiration_median_filtering=self.respiration_median_filtering,
            envelope_ratio=self.envelope_ratio,
            consonant_breath_protection=self.consonant_breath_protection,
        )
    
    def get_post_params(self) -> PostProcessParams:
        """Extract post-processing parameters as dataclass."""
        return PostProcessParams(
            highpass=self.highpass,
            lowpass=self.lowpass,
            bass_freq=self.bass_freq,
            bass_gain=self.bass_gain,
            treble_freq=self.treble_freq,
            treble_gain=self.treble_gain,
            reverb_delay=self.reverb_delay,
            reverb_decay=self.reverb_decay,
            crystalizer=self.crystalizer,
            deesser=self.deesser,
            audio_8d_enabled=self.audio_8d_enabled,
            audio_8d_mode=self.audio_8d_mode,
            audio_8d_speed=self.audio_8d_speed,
            audio_8d_depth=self.audio_8d_depth,
            audio_8d_distance=self.audio_8d_distance,
            audio_8d_quality=self.audio_8d_quality,
            audio_8d_itd=self.audio_8d_itd,
            audio_8d_proximity=self.audio_8d_proximity,
            audio_8d_crossfeed=self.audio_8d_crossfeed,
            audio_8d_micro_movements=self.audio_8d_micro_movements,
            audio_8d_speech_aware=self.audio_8d_speech_aware,
            pitch_shift_enabled=self.pitch_shift_enabled,
            pitch_shift_semitones=self.pitch_shift_semitones,
            asmr_enabled=self.asmr_enabled,
            asmr_tingles=self.asmr_tingles,
            asmr_breathiness=self.asmr_breathiness,
            asmr_crispness=self.asmr_crispness,
        )
    
    def get_background_params(self) -> BackgroundParams:
        """Extract background parameters as dataclass."""
        files = self.bg_files
        volumes = self.bg_volumes if self.bg_volumes else [self.background_volume] * len(files)
        # Use bg_delays if provided, otherwise use legacy single background_delay
        delays = self.bg_delays if self.bg_delays else [self.background_delay] * len(files) if files else []
        fade_ins = self.bg_fade_ins if self.bg_fade_ins else [0.0] * len(files)
        fade_outs = self.bg_fade_outs if self.bg_fade_outs else [0.0] * len(files)
        
        return BackgroundParams(
            enabled=self.enable_background or False,
            files=files,
            volumes=volumes,
            delays=delays,
            fade_ins=fade_ins,
            fade_outs=fade_outs,
            main_volume=self.main_audio_volume,
            use_config_tracks=self.use_config_bg_tracks,
        )
    
    class Config:
        extra = "ignore"


class TTSResponse(BaseModel):
    """Response model for TTS generation (for JSON responses)."""
    success: bool
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    format: Optional[str] = None
    error: Optional[str] = None
    
    class Config:
        extra = "ignore"
