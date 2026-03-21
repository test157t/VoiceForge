"""
TTS Router - Text-to-Speech endpoints.

Handles:
- /v1/audio/speech (OpenAI-compatible)
- /api/generate (UI endpoint)
- /api/generate/stream (Streaming endpoint)
- /api/tts/warmup (Model warmup)
"""

import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# Import common (sets up sys.path)
from .common import verify_auth, DEFAULT_MODEL, MODEL_DIR

# Now imports from app directory work
from servers.models.requests import TTSRequest
from servers.services.pipeline import (
    generate_audio,
    generate_audio_streaming,
    cancel_generation,
    cancel_all_generations,
    get_active_requests,
)
from util.file_utils import validate_model_exists
from util.clients import get_chatterbox_client
from config import get_config_value


router = APIRouter(tags=["TTS"])


class WarmupRequest(BaseModel):
    prompt_audio_path: Optional[str] = None


@router.post("/api/tts/warmup")
async def warmup_tts_model(
    request: WarmupRequest = None,
    _: bool = Depends(verify_auth)
):
    """
    Warmup the Chatterbox TTS model.
    
    Runs a test inference to warm up CUDA kernels and model internal states.
    Uses the provided prompt or falls back to config.
    """
    try:
        chatterbox = get_chatterbox_client()
        
        if not chatterbox.is_available():
            raise HTTPException(
                status_code=503,
                detail="Chatterbox TTS server is not available"
            )
        
        # Get prompt path from request or config
        prompt_path = None
        if request and request.prompt_audio_path:
            prompt_path = request.prompt_audio_path
        else:
            # Fall back to config
            prompt_path = get_config_value("chatterbox_prompt_path")
        
        result = chatterbox.warmup(prompt_path)
        
        if result and result.get("status") == "ok":
            return {
                "status": "ok", 
                "message": result.get("message", "TTS model warmed up successfully")
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Warmup failed - check Chatterbox server logs")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/generate/cancel")
async def cancel_tts_generation(
    request_id: Optional[str] = Query(default=None, description="Request ID to cancel"),
    _: bool = Depends(verify_auth)
):
    """
    Cancel a TTS generation request.
    
    If request_id is provided, cancels that specific request.
    If not provided, cancels all active requests.
    """
    print(f"[Cancel] Request to cancel: {request_id or 'ALL'}")
    if request_id:
        success = cancel_generation(request_id)
        print(f"[Cancel] cancel_generation({request_id}) returned: {success}")
        if success:
            # Flag is set - workers will stop after current chunk finishes
            # Note: PyTorch model.generate() cannot be interrupted mid-execution
            return {
                "status": "pending",
                "message": f"Request {request_id} marked for cancellation (will stop after current chunk)",
                "note": "Generation will stop after the current audio chunk finishes processing"
            }
        else:
            return {"status": "ok", "message": f"Request {request_id} not found or already completed"}
    else:
        count = cancel_all_generations()
        print(f"[Cancel] cancel_all_generations() returned: {count}")
        return {"status": "ok", "message": f"Cancelled {count} request(s)", "cancelled": count}


@router.get("/api/generate/active")
async def get_active_tts_requests(_: bool = Depends(verify_auth)):
    """Get list of active TTS generation request IDs."""
    return {"requests": get_active_requests()}


@router.post("/v1/audio/speech")
async def create_speech(
    request: TTSRequest,
    _: bool = Depends(verify_auth)
):
    """
    OpenAI-compatible speech synthesis endpoint.
    
    Generates audio from text using Chatterbox TTS,
    with optional RVC voice conversion, post-processing, and background blending.
    """
    # Use client-provided request_id if available, otherwise generate one
    request_id = request.request_id or str(uuid.uuid4())[:8]
    
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    # Resolve RVC model
    rvc_model = _resolve_rvc_model(request)
    if rvc_model:
        request.rvc_model = rvc_model
    
    print(f"[{request_id}] TTS request: mode={request.tts_mode}, backend={request.tts_backend}, rvc={request.rvc_model}")
    
    # Run pipeline with request_id for cancellation support
    result = await generate_audio(request, request_id=request_id)
    
    try:
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or "TTS generation failed")
        
        return Response(
            content=result.audio_data,
            media_type=result.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            }
        )
    finally:
        result.cleanup()


@router.post("/api/generate")
async def generate_speech_ui(
    request: TTSRequest,
    _: bool = Depends(verify_auth)
):
    """
    UI endpoint for speech generation.
    
    Same as /v1/audio/speech but with full parameter support for the UI.
    """
    return await create_speech(request, _)


@router.post("/api/generate/stream")
async def generate_speech_stream(
    request: TTSRequest,
    _: bool = Depends(verify_auth)
):
    """
    Streaming TTS endpoint - returns SSE events with audio chunks.
    
    Each text chunk is processed through TTS -> RVC -> PostProcess
    and streamed to the client as it completes.
    
    Events:
    - start: {type: "start", request_id: str, chunks: N, rvc_enabled: bool, post_enabled: bool}
    - chunk: {type: "chunk", index: N, total: N, audio: base64, duration: float, text: str}
    - complete: {type: "complete", chunks_sent: N}
    - error: {type: "error", message: str}
    - cancelled: {type: "cancelled", message: str}
    """
    import time
    arrival_time = time.time()
    
    # Use client-provided request_id if available
    request_id = request.request_id or str(uuid.uuid4())[:8]
    
    # Log exact input text (repr) so debugging is unambiguous.
    # Do not append artificial ellipsis here.
    print(f"[{request_id}] === REQUEST ARRIVED at {arrival_time:.3f} === input_len={len(request.input)} input={request.input!r}")
    
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    # Resolve RVC model
    rvc_model = _resolve_rvc_model(request)
    if rvc_model:
        request.rvc_model = rvc_model
    
    print(f"[{request_id}] TTS stream request: backend={request.tts_backend}, rvc={request.rvc_model}")
    
    async def event_generator():
        """Generate SSE events from the streaming pipeline."""
        async for event in generate_audio_streaming(request, request_id=request_id):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


def _resolve_rvc_model(request: TTSRequest) -> Optional[str]:
    """
    Resolve RVC model from request.
    
    Returns:
        RVC model name or None
    """
    # Use explicitly specified model
    if request.rvc_model:
        model_name = request.rvc_model
    else:
        model_name = DEFAULT_MODEL
    
    # Verify model exists
    if model_name and not validate_model_exists(model_name):
        if model_name != DEFAULT_MODEL:
            model_name = DEFAULT_MODEL
        if not validate_model_exists(model_name):
            model_name = None
    
    return model_name
