"""
Kokoro TTS ONNX Server
FastAPI-based TTS server using kokoro-onnx for text-to-speech generation.

Supports:
- Multiple voices (af, am, bf, bm, etc.)
- Speed control
- OpenAI-compatible /v1/audio/speech API
- Streaming with SSE
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict
import io
import numpy as np
import logging
import os
import tempfile
import hashlib
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
kokoro_pipeline = None

# Available voices in kokoro v1.0
# See: https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
AVAILABLE_VOICES = [
    # American English (en-us)
    "af_sarah", "af_bella", "af_heart", "af_nicole", "af_sky",
    "am_michael", "am_echo", "am_onyx", "am_fable", "am_puck", "am_sage",
    # British English (en-gb)
    "bf_emma", "bf_isabella", "bf_alice",
    "bm_george", "bm_lewis", "bm_daniel",
]

DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.0


def download_kokoro_models():
    """Download Kokoro model files if they don't exist."""
    import urllib.request
    import os
    
    model_dir = os.environ.get("KOKORO_MODEL_DIR", os.path.join(os.path.expanduser("~"), ".kokoro"))
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "kokoro-v1.0.onnx")
    voices_path = os.path.join(model_dir, "voices-v1.0.bin")
    
    # URLs for model files
    model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    
    # Download model if needed
    if not os.path.exists(model_path):
        logger.info(f"Downloading Kokoro model to {model_path}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            logger.info("Model downloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None, None
    
    # Download voices if needed
    if not os.path.exists(voices_path):
        logger.info(f"Downloading Kokoro voices to {voices_path}...")
        try:
            urllib.request.urlretrieve(voices_url, voices_path)
            logger.info("Voices downloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to download voices: {e}")
            return None, None
    
    return model_path, voices_path


def load_kokoro_model():
    """Load the Kokoro TTS model."""
    global kokoro_pipeline
    
    try:
        from kokoro_onnx import Kokoro
        
        logger.info("Loading Kokoro TTS model...")
        
        # Download models if needed
        model_path, voices_path = download_kokoro_models()
        
        if model_path is None or voices_path is None:
            logger.error("Failed to download or locate model files")
            return False
        
        # Load the model
        kokoro_pipeline = Kokoro(model_path, voices_path)
        logger.info("Kokoro TTS model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load Kokoro TTS model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Kokoro TTS Server...")
    load_kokoro_model()
    yield
    # Shutdown
    logger.info("Shutting down Kokoro TTS Server...")


app = FastAPI(
    title="Kokoro TTS Server",
    description="ONNX-based TTS server using kokoro-onnx",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    """TTS request model (OpenAI-compatible)."""
    model_config = ConfigDict(populate_by_name=True)
    
    model: str = Field(default="kokoro-tts", description="Model name")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice to use")
    response_format: str = Field(default="wav", description="Audio format: wav, mp3, opus")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed (0.25-4.0)")


class TTSStreamRequest(BaseModel):
    """TTS streaming request model."""
    model_config = ConfigDict(populate_by_name=True)
    
    model: str = Field(default="kokoro-tts")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice to use")
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": kokoro_pipeline is not None,
        "model": "kokoro-onnx",
        "available_voices": AVAILABLE_VOICES
    }


@app.get("/v1/voices")
async def get_voices():
    """Get list of available voices."""
    return {
        "voices": [
            {"id": v, "name": v.upper(), "description": f"{v.upper()} voice"} 
            for v in AVAILABLE_VOICES
        ]
    }


def synthesize_speech(text: str, voice: str, speed: float = 1.0) -> np.ndarray:
    """
    Synthesize speech using Kokoro.
    
    Args:
        text: Text to synthesize
        voice: Voice ID (af, am, bf, bm)
        speed: Playback speed multiplier
        
    Returns:
        Audio samples as numpy array
    """
    global kokoro_pipeline
    
    if kokoro_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Validate voice
    if voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not available. Choose from: {', '.join(AVAILABLE_VOICES)}"
        )
    
    try:
        logger.info(f"Synthesizing with voice={voice}, speed={speed}, text_length={len(text)}")
        
        # Generate audio using kokoro-onnx
        samples, sample_rate = kokoro_pipeline.create(text, voice=voice, speed=speed, lang="en-us")
        
        logger.info(f"Generated {len(samples)} samples at {sample_rate}Hz")
        return samples, sample_rate
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


def encode_audio(samples: np.ndarray, sample_rate: int, format: str = "wav") -> bytes:
    """Encode audio samples to the requested format."""
    import scipy.io.wavfile
    import io
    
    format = format.lower()
    
    if format == "wav":
        # Convert to int16
        samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, sample_rate, samples_int16)
        buffer.seek(0)
        return buffer.read()
    
    elif format == "mp3":
        # For MP3, we'd need pydub or similar. For now, return WAV with warning
        logger.warning("MP3 format requested but not implemented, returning WAV")
        return encode_audio(samples, sample_rate, "wav")
    
    elif format == "opus":
        # For Opus, return WAV as fallback
        logger.warning("Opus format requested but not implemented, returning WAV")
        return encode_audio(samples, sample_rate, "wav")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Create speech from text (OpenAI-compatible endpoint).
    
    Returns audio as binary response.
    """
    try:
        samples, sample_rate = synthesize_speech(
            text=request.input,
            voice=request.voice,
            speed=request.speed
        )
        
        audio_bytes = encode_audio(samples, sample_rate, request.response_format)
        
        content_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus"
        }.get(request.response_format, "audio/wav")
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Voice": request.voice
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech creation failed: {str(e)}")


@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: TTSStreamRequest):
    """
    Create speech with streaming (SSE-based).
    
    Returns Server-Sent Events with audio chunks as base64.
    """
    async def stream_generator():
        try:
            # Start event
            yield f"data: {json.dumps({'type': 'start', 'voice': request.voice})}\n\n"
            
            # Generate full audio (kokoro doesn't support true streaming)
            samples, sample_rate = synthesize_speech(
                text=request.input,
                voice=request.voice,
                speed=request.speed
            )
            
            # Encode to WAV
            import base64
            audio_bytes = encode_audio(samples, sample_rate, "wav")
            
            # Send as single chunk
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            yield f"data: {json.dumps({'type': 'chunk', 'audio': audio_b64, 'sample_rate': sample_rate})}\n\n"
            
            # Complete event
            yield f"data: {json.dumps({'type': 'complete', 'sample_rate': sample_rate, 'total_samples': len(samples)})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Voice": request.voice
        }
    )


@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "service": "Kokoro TTS Server",
        "version": "1.0.0",
        "model": "kokoro-onnx",
        "endpoints": {
            "health": "/health",
            "voices": "/v1/voices",
            "speech": "/v1/audio/speech",
            "speech_stream": "/v1/audio/speech/stream"
        },
        "available_voices": AVAILABLE_VOICES
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Kokoro TTS Server")
    parser.add_argument("--port", type=int, default=8896, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Kokoro TTS Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
