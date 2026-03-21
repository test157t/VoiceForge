"""
ASR Router - Speech Recognition endpoints.

Handles:
- /v1/audio/transcriptions - OpenAI-compatible transcription
- /api/transcribe - Simple transcription
- /api/save-transcription - Save transcription text to file
- /v1/asr/health - ASR health check
"""

import json
import os
import tempfile
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

# Import common (sets up sys.path)
from .common import verify_auth

from util.clients import (
    get_whisperasr_client,
    is_whisperasr_available,
    get_glmasr_client,
    is_glmasr_available,
    transcribe_audio,
    get_shared_session,
    WHISPERASR_SERVER_URL,
    GLMASR_SERVER_URL,
)


router = APIRouter(tags=["ASR"])

# ASR Configuration
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "whisper-large-v3-turbo")
ASR_DEFAULT_LANGUAGE = os.getenv("ASR_DEFAULT_LANGUAGE", "en")


@router.get("/v1/asr/health")
async def asr_health_check(backend: str = None):
    """Check ASR server health.
    
    Args:
        backend: Optional backend to check ("whisper" or "glm"). 
                 If not specified, checks both and returns combined status.
    """
    result = {
        "status": "healthy",
        "backends": {}
    }
    
    # Check Whisper ASR
    if backend is None or backend == "whisper":
        try:
            resp = get_shared_session().get(f"{WHISPERASR_SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                result["backends"]["whisper"] = resp.json()
                result["backends"]["whisper"]["url"] = WHISPERASR_SERVER_URL
            else:
                result["backends"]["whisper"] = {"status": "error", "detail": resp.text}
        except Exception as e:
            result["backends"]["whisper"] = {"status": "unavailable", "detail": str(e)}
    
    # Check GLM-ASR
    if backend is None or backend == "glm":
        try:
            resp = get_shared_session().get(f"{GLMASR_SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                result["backends"]["glm"] = resp.json()
                result["backends"]["glm"]["url"] = GLMASR_SERVER_URL
            else:
                result["backends"]["glm"] = {"status": "error", "detail": resp.text}
        except Exception as e:
            result["backends"]["glm"] = {"status": "unavailable", "detail": str(e)}
    
    # Set overall status
    whisper_ok = result["backends"].get("whisper", {}).get("status") == "healthy"
    glm_ok = result["backends"].get("glm", {}).get("status") == "healthy"
    
    if not whisper_ok and not glm_ok:
        result["status"] = "unhealthy"
    elif not whisper_ok or not glm_ok:
        result["status"] = "partial"
    
    return result


@router.post("/api/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...),
    clean_vocals: bool = Form(default=False),
    skip_existing_vocals: bool = Form(default=True),
    postprocess_audio: bool = Form(default=False),
    device: str = Form(default="gpu"),
    model: str = Form(default=None),
    language: str = Form(default=None),
    _: bool = Depends(verify_auth)
):
    """
    Transcribe audio to text.
    
    Args:
        audio: Audio file to transcribe
        clean_vocals: Whether to clean vocals first with UVR5
        model: ASR model to use. Use "glm-asr-nano" (transformers)
               or "whisper-large-v3-turbo" etc. for Whisper
        language: Language code (e.g., 'en', 'es', 'zh')
    """
    effective_model = model or ASR_MODEL_NAME
    
    # Check appropriate server availability based on model
    if effective_model.lower().startswith("glm"):
        if not is_glmasr_available():
            raise HTTPException(
                status_code=503,
                detail="GLM-ASR server is not available. Run launch_glmasr_server.bat to start it."
            )
    else:
        if not is_whisperasr_available():
            raise HTTPException(
                status_code=503,
                detail="Whisper ASR server is not available. Start it with option [4] in the launcher."
            )
    
    # Save uploaded file
    file_ext = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=file_ext)
    os.close(fd)
    
    try:
        content = await audio.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        
        # Transcribe (routes to appropriate backend automatically)
        result = transcribe_audio(
            tmp_path,
            language=language or ASR_DEFAULT_LANGUAGE,
            model=effective_model
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    http_request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3-turbo"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form(default="json"),
    timestamp_granularities: Optional[str] = Form(default=None),
    _: bool = Depends(verify_auth)
):
    """
    OpenAI-compatible transcription endpoint.
    
    Creates a transcription from audio using Whisper or GLM-ASR.
    
    Supported models:
    - "glm-asr-nano": GLM-ASR-Nano-2512 via transformers
    - "whisper-large-v3-turbo": Whisper large-v3-turbo (default)
    - "whisper-large-v3": Whisper large-v3
    - "whisper-medium", "whisper-small", "whisper-base", "whisper-tiny": Other Whisper sizes
    """
    # Check appropriate server availability based on model
    if model.lower().startswith("glm"):
        if not is_glmasr_available():
            raise HTTPException(
                status_code=503,
                detail="GLM-ASR server is not available. Run launch_glmasr_server.bat to start it."
            )
    else:
        if not is_whisperasr_available():
            raise HTTPException(
                status_code=503,
                detail="Whisper ASR server is not available"
            )
    
    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=file_ext)
    os.close(fd)
    
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        
        # Transcribe
        result = transcribe_audio(
            tmp_path,
            language=language or ASR_DEFAULT_LANGUAGE,
            model=model
        )
        
        # Format response based on response_format
        if response_format == "text":
            return result.get("text", "")
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": language or ASR_DEFAULT_LANGUAGE,
                "duration": result.get("duration"),
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
            }
        elif response_format == "srt":
            # Generate SRT format
            segments = result.get("segments", [])
            srt_lines = []
            for i, seg in enumerate(segments, 1):
                start = _format_srt_time(seg.get("start", 0))
                end = _format_srt_time(seg.get("end", 0))
                text = seg.get("text", "").strip()
                srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
            return "\n".join(srt_lines)
        elif response_format == "vtt":
            # Generate VTT format
            segments = result.get("segments", [])
            vtt_lines = ["WEBVTT\n"]
            for seg in segments:
                start = _format_vtt_time(seg.get("start", 0))
                end = _format_vtt_time(seg.get("end", 0))
                text = seg.get("text", "").strip()
                vtt_lines.append(f"\n{start} --> {end}\n{text}")
            return "\n".join(vtt_lines)
        else:
            # Default JSON
            return {"text": result.get("text", "")}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


def _format_srt_time(seconds: float) -> str:
    """Format time for SRT subtitles."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format time for VTT subtitles."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@router.post("/api/transcribe/onnx")
async def transcribe_onnx(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    model: str = Form(default=None),
    _: bool = Depends(verify_auth)
):
    """
    Transcribe audio using ONNX Whisper (if available).
    Falls back to standard ASR if ONNX not available.
    
    Also supports GLM-ASR via model="glm-asr-nano".
    """
    effective_model = model or ASR_MODEL_NAME
    
    if effective_model.lower().startswith("glm"):
        if not is_glmasr_available():
            raise HTTPException(
                status_code=503,
                detail="GLM-ASR server is not available"
            )
    else:
        if not is_whisperasr_available():
            raise HTTPException(
                status_code=503,
                detail="Whisper ASR server is not available"
            )
    
    file_ext = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=file_ext)
    os.close(fd)
    
    try:
        content = await audio.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        
        result = transcribe_audio(tmp_path, language=language, model=effective_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@router.post("/api/transcribe/stream")
async def transcribe_stream(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    model: str = Form(default=None),
    _: bool = Depends(verify_auth)
):
    """
    Transcribe audio with streaming response.
    Returns results as they become available.
    
    Supports both Whisper and GLM-ASR backends.
    """
    effective_model = model or ASR_MODEL_NAME
    
    if effective_model.lower().startswith("glm"):
        if not is_glmasr_available():
            raise HTTPException(
                status_code=503,
                detail="GLM-ASR server is not available"
            )
    else:
        if not is_whisperasr_available():
            raise HTTPException(
                status_code=503,
                detail="Whisper ASR server is not available"
            )
    
    file_ext = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=file_ext)
    os.close(fd)
    
    try:
        content = await audio.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        
        # For now, just return normal transcription
        # Streaming would require websocket or SSE
        result = transcribe_audio(tmp_path, language=language, model=effective_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@router.post("/api/save-transcription")
async def save_transcription(
    request: Request,
    _: bool = Depends(verify_auth)
):
    """
    Save transcription text to a file.
    
    Args:
        request: JSON body with { "text": "transcription content" }
        
    Returns:
        { "filename": "transcription_YYYYMMDD_HHMMSS.txt", "path": "/path/to/file" }
    """
    from datetime import datetime
    
    try:
        body = await request.json()
        text = body.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Create transcriptions directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "outputs", "transcriptions")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Save the transcription
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        
        return {"filename": filename, "path": filepath}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
