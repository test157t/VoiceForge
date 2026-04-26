from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import io
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

omnivoice_models: Dict[str, Any] = {}
_model_details: Dict[str, str] = {}
_model_load_lock = threading.Lock()
_model_load_failures: Dict[str, Dict[str, Any]] = {}
DEFAULT_OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
OMNIVOICE_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 48000
ASR_SERVER_URL = os.getenv("ASR_SERVER_URL", "http://127.0.0.1:8889").rstrip("/")
OMNIVOICE_USE_EXTERNAL_ASR = os.getenv("OMNIVOICE_USE_EXTERNAL_ASR", "1").strip().lower() in {"1", "true", "yes", "on"}
OMNIVOICE_EXTERNAL_ASR_MODEL = os.getenv("OMNIVOICE_EXTERNAL_ASR_MODEL", "glm-asr-nano")


def _normalize_runtime(value: str) -> str:
    runtime = (value or "torch").strip().lower()
    if runtime in {"", "auto", "default", "torch"}:
        return "torch"
    raise ValueError(f"Unsupported OmniVoice runtime: {value}")


def _resolve_default_runtime() -> str:
    try:
        return _normalize_runtime(os.getenv("OMNIVOICE_RUNTIME", "torch"))
    except ValueError:
        logger.warning("Ignoring unsupported OMNIVOICE_RUNTIME value; using torch")
    return "torch"


OMNIVOICE_DEFAULT_RUNTIME = _resolve_default_runtime()


_model_from_env = os.getenv("OMNIVOICE_MODEL_ID", "").strip()
OMNIVOICE_TORCH_MODEL_ID = os.getenv("OMNIVOICE_TORCH_MODEL_ID", "").strip()
if not OMNIVOICE_TORCH_MODEL_ID:
    if _model_from_env:
        OMNIVOICE_TORCH_MODEL_ID = _model_from_env
    else:
        OMNIVOICE_TORCH_MODEL_ID = DEFAULT_OMNIVOICE_MODEL_ID
OMNIVOICE_PRELOAD_DEFAULT_RUNTIME = os.getenv("OMNIVOICE_PRELOAD_DEFAULT_RUNTIME", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OMNIVOICE_MODEL_LOAD_RETRY_SECONDS = int(os.getenv("OMNIVOICE_MODEL_LOAD_RETRY_SECONDS", "30"))

VOICE_GUIDE = [
    "auto",
    "instruct:female, low pitch, british accent",
    r"C:\path\to\reference.wav",
]


class AudioSpeechRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "model": "omnivoice-tts",
                "input": "Hello, world!",
                "voice": "auto",
                "response_format": "wav",
                "speed": 1.0,
                "max_tokens": 50,
            }
        },
    )

    model: str = Field(default="omnivoice-tts", description="Model to use")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(
        default="auto",
        description="auto, instruct:<attributes>, or path to reference audio for cloning",
    )
    ref_text: Optional[str] = Field(default=None, description="Optional transcript for reference audio")
    response_format: str = Field(default="wav", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    num_step: Optional[int] = Field(default=None, ge=4, le=64, description="Decoding steps (lower = faster, lower quality)")
    max_tokens: int = Field(default=50, ge=5, le=200, description="Max tokens per chunk")
    token_method: str = Field(default="tiktoken", description="Token counting method: tiktoken or words")
    prechunked: bool = Field(default=False, description="If true, skip server-side text splitting")
    runtime: Optional[str] = Field(
        default=None,
        description="Runtime override: torch or auto (default server runtime)",
    )


def _resolve_runtime_override(runtime: Optional[str]) -> str:
    requested = (runtime or "").strip().lower()
    if requested in {"", "auto", "default"}:
        return OMNIVOICE_DEFAULT_RUNTIME
    try:
        return _normalize_runtime(requested)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _model_id_for_runtime(runtime: str) -> str:
    return OMNIVOICE_TORCH_MODEL_ID


def _resolve_audio_path(path_or_empty: str) -> Optional[str]:
    candidate = (path_or_empty or "").strip().strip('"')
    if not candidate:
        return None
    if candidate.startswith("file://"):
        candidate = candidate[len("file://") :]
    if os.path.isfile(candidate):
        return candidate
    return None


def _split_text_into_chunks(text: str, max_tokens: int = 50, token_method: str = "tiktoken") -> List[str]:
    import sys

    util_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "util")
    if util_path not in sys.path:
        sys.path.insert(0, util_path)

    from text_utils import split_text

    method = (token_method or "tiktoken").strip().lower()
    if method not in {"tiktoken", "words"}:
        method = "tiktoken"

    return [chunk for chunk in split_text(text, max_tokens=max_tokens, token_method=method) if chunk.strip()]


def _extract_waveform(result: Any) -> np.ndarray:
    """Normalize OmniVoice output into a mono float32 numpy array."""
    tensor = None

    if isinstance(result, list) and result:
        tensor = result[0]
    elif isinstance(result, torch.Tensor):
        tensor = result

    if tensor is None:
        raise RuntimeError("OmniVoice returned empty audio")

    if isinstance(tensor, torch.Tensor):
        wav = tensor.detach().cpu().float().numpy()
    else:
        wav = np.asarray(tensor, dtype=np.float32)

    if wav.ndim == 2:
        wav = wav[0]
    if wav.ndim != 1:
        raise RuntimeError(f"Unexpected OmniVoice output shape: {wav.shape}")

    return wav.astype(np.float32)


def _voice_kwargs(voice: str, ref_text: Optional[str]) -> Dict[str, Any]:
    voice = (voice or "auto").strip()
    lower = voice.lower()
    kwargs: Dict[str, Any] = {}

    looks_like_path = ("/" in voice) or ("\\" in voice) or lower.endswith(".wav")

    ref_audio = _resolve_audio_path(voice)
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
        if ref_text and ref_text.strip():
            kwargs["ref_text"] = ref_text.strip()
        return kwargs
    if looks_like_path:
        raise HTTPException(status_code=400, detail=f"Reference audio file not found: {voice}")

    if lower.startswith("instruct:"):
        instruct = voice.split(":", 1)[1].strip()
        if instruct:
            kwargs["instruct"] = instruct
        return kwargs

    if lower not in {"", "auto", "default", "random"}:
        kwargs["instruct"] = voice

    return kwargs


def _transcribe_reference_with_asr_server(ref_audio_path: str) -> Optional[str]:
    """Use VoiceForge ASR server once per request to avoid per-chunk re-transcription."""
    if not OMNIVOICE_USE_EXTERNAL_ASR:
        return None

    try:
        t0 = time.perf_counter()
        with open(ref_audio_path, "rb") as f:
            files = {"file": (os.path.basename(ref_audio_path), f, "audio/wav")}
            data = {
                "language": "auto",
                "response_format": "json",
                "model": OMNIVOICE_EXTERNAL_ASR_MODEL,
                "clean_vocals": "false",
                "skip_existing_vocals": "true",
                "postprocess_audio": "false",
                "device": "gpu",
            }
            with httpx.Client(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
                response = client.post(f"{ASR_SERVER_URL}/v1/audio/transcriptions", files=files, data=data)

        if response.status_code != 200:
            logger.warning(
                f"External ASR transcription failed ({response.status_code}). "
                "Falling back to OmniVoice internal transcription."
            )
            return None

        payload = response.json()
        text = (payload.get("text") or "").strip()
        if not text:
            logger.warning("External ASR returned empty text; falling back to OmniVoice internal transcription.")
            return None

        asr_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Reference transcript obtained from ASR server ({len(text)} chars, {asr_ms:.0f} ms)")
        return text
    except Exception as e:
        logger.warning(f"External ASR unavailable; falling back to OmniVoice internal transcription: {e}")
        return None


def _prepare_reference_text_once(voice: str, ref_text: Optional[str]) -> Optional[str]:
    """Prepare ref_text once per request so chunked generation does not retranscribe each chunk."""
    if ref_text and ref_text.strip():
        logger.info(f"Using caller-provided ref_text ({len(ref_text.strip())} chars)")
        return ref_text.strip()

    ref_audio = _resolve_audio_path(voice)
    if not ref_audio:
        return None

    logger.info(f"Preparing reference transcript once for: {os.path.basename(ref_audio)}")
    return _transcribe_reference_with_asr_server(ref_audio)


def _generate_chunk_audio(
    model: Any,
    text: str,
    voice: str,
    ref_text: Optional[str],
    speed: float,
    num_step: Optional[int] = None,
) -> np.ndarray:
    kwargs = _voice_kwargs(voice, ref_text)
    if num_step is not None:
        kwargs["num_step"] = int(num_step)
    result = model.generate(text=text, speed=speed, **kwargs)
    audio = _extract_waveform(result)

    if OMNIVOICE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
        audio = scipy.signal.resample_poly(audio, TARGET_SAMPLE_RATE, OMNIVOICE_SAMPLE_RATE)

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _load_torch_runtime_model(model_id: str):
    from omnivoice import OmniVoice

    if torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device_map = "mps"
        dtype = torch.float32
    else:
        device_map = "cpu"
        dtype = torch.float32

    model = OmniVoice.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype,
    )
    return model, f"torch runtime on {device_map} (dtype={dtype}, model={model_id})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if OMNIVOICE_PRELOAD_DEFAULT_RUNTIME:
        logger.info("Preloading OmniVoice default runtime model...")
        try:
            model, detail = _load_torch_runtime_model(_model_id_for_runtime("torch"))
            omnivoice_models[OMNIVOICE_DEFAULT_RUNTIME] = model
            _model_details[OMNIVOICE_DEFAULT_RUNTIME] = detail
            logger.info(f"OmniVoice loaded successfully with {detail}")
        except Exception as e:
            logger.error(f"Failed to preload OmniVoice model: {e}")
            raise
    else:
        logger.info(
            "OmniVoice runtime preload disabled; models will lazy-load on first request. "
            f"Default runtime={OMNIVOICE_DEFAULT_RUNTIME}"
        )

    yield
    logger.info("Shutting down OmniVoice server...")


app = FastAPI(title="OmniVoice Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_or_load_model_for_runtime(runtime: str):
    model = omnivoice_models.get(runtime)
    if model is not None:
        return model

    failure = _model_load_failures.get(runtime)
    if failure:
        age = time.time() - float(failure.get("ts", 0.0))
        if age < OMNIVOICE_MODEL_LOAD_RETRY_SECONDS:
            wait_s = int(max(1, OMNIVOICE_MODEL_LOAD_RETRY_SECONDS - age))
            raise RuntimeError(
                f"Previous runtime load failed recently; retry in {wait_s}s. "
                f"Last error: {failure.get('error', 'unknown')}"
            )

    with _model_load_lock:
        model = omnivoice_models.get(runtime)
        if model is not None:
            return model

        failure = _model_load_failures.get(runtime)
        if failure:
            age = time.time() - float(failure.get("ts", 0.0))
            if age < OMNIVOICE_MODEL_LOAD_RETRY_SECONDS:
                wait_s = int(max(1, OMNIVOICE_MODEL_LOAD_RETRY_SECONDS - age))
                raise RuntimeError(
                    f"Previous runtime load failed recently; retry in {wait_s}s. "
                    f"Last error: {failure.get('error', 'unknown')}"
                )

        model_id = _model_id_for_runtime(runtime)
        logger.info(f"Lazy-loading OmniVoice runtime={runtime} model={model_id}...")
        try:
            model, detail = _load_torch_runtime_model(model_id)
        except Exception as e:
            _model_load_failures[runtime] = {
                "ts": time.time(),
                "error": str(e),
            }
            logger.error(f"OmniVoice runtime={runtime} failed to load: {e}")
            raise

        _model_load_failures.pop(runtime, None)
        omnivoice_models[runtime] = model
        _model_details[runtime] = detail
        logger.info(f"OmniVoice runtime={runtime} ready ({detail})")
        return model


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}: {json.dumps(exc.errors(), indent=2)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            },
            "detail": exc.errors(),
        },
    )


@app.get("/")
async def root():
    return {
        "service": "OmniVoice Server",
        "default_runtime": OMNIVOICE_DEFAULT_RUNTIME,
        "loaded_runtimes": sorted(list(omnivoice_models.keys())),
        "runtime_details": _model_details,
        "failed_runtimes": sorted(list(_model_load_failures.keys())),
        "models": {"torch": OMNIVOICE_TORCH_MODEL_ID},
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "voices": "/v1/voices",
            "speech": "/v1/audio/speech",
            "speech_stream": "/v1/audio/speech/stream",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": bool(omnivoice_models),
        "default_runtime": OMNIVOICE_DEFAULT_RUNTIME,
        "loaded_runtimes": sorted(list(omnivoice_models.keys())),
        "runtime_details": _model_details,
        "failed_runtimes": sorted(list(_model_load_failures.keys())),
        "models": {"torch": OMNIVOICE_TORCH_MODEL_ID},
        "output_sample_rate": TARGET_SAMPLE_RATE,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "omnivoice-tts",
                "object": "model",
                "created": 0,
                "owned_by": "k2-fsa",
                "source": {"torch": OMNIVOICE_TORCH_MODEL_ID},
            }
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    return {
        "voices": VOICE_GUIDE,
        "notes": {
            "auto": "No voice prompt, model chooses voice automatically",
            "instruct": "Prefix with 'instruct:' for voice design attributes",
            "path": "Provide a local reference WAV path for voice cloning",
        },
    }


@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    runtime = _resolve_runtime_override(request.runtime)
    try:
        model = _get_or_load_model_for_runtime(runtime)
    except Exception as e:
        logger.error(f"Runtime load failed for runtime={runtime}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load OmniVoice runtime '{runtime}': {e}")

    effective_num_step = request.num_step

    if request.prechunked:
        chunks = [request.input.strip()]
    else:
        chunks = _split_text_into_chunks(
            request.input,
            max_tokens=request.max_tokens,
            token_method=request.token_method,
        )
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text to synthesize")

    prepared_ref_text = _prepare_reference_text_once(request.voice, request.ref_text)

    mode = "prechunked" if request.prechunked else "chunked"
    logger.info(f"Generating {len(chunks)} OmniVoice chunks (mode={mode}, runtime={runtime})...")

    all_audio = []
    start = time.perf_counter()
    total_audio_sec = 0.0
    for idx, chunk in enumerate(chunks, start=1):
        try:
            chunk_start = time.perf_counter()
            chunk_audio = _generate_chunk_audio(
                model,
                chunk,
                request.voice,
                prepared_ref_text,
                request.speed,
                effective_num_step,
            )
            if len(chunk_audio) > 0:
                all_audio.append(chunk_audio)
            chunk_audio_sec = len(chunk_audio) / TARGET_SAMPLE_RATE if len(chunk_audio) > 0 else 0.0
            total_audio_sec += chunk_audio_sec
            chunk_wall = time.perf_counter() - chunk_start
            chunk_xrt = (chunk_audio_sec / chunk_wall) if chunk_wall > 0 else 0.0
            logger.info(f"[{idx}/{len(chunks)}] generated {len(chunk_audio)} samples ({chunk_audio_sec:.2f}s audio in {chunk_wall:.2f}s, {chunk_xrt:.2f}x RT)")
        except Exception as e:
            logger.error(f"[{idx}/{len(chunks)}] chunk failed: {e}")

    if not all_audio:
        raise HTTPException(status_code=500, detail="No audio chunks generated")

    audio_np = np.concatenate(all_audio)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, TARGET_SAMPLE_RATE, audio_np)
    elapsed = time.perf_counter() - start
    xrt = (total_audio_sec / elapsed) if elapsed > 0 else 0.0
    rtf = (elapsed / total_audio_sec) if total_audio_sec > 0 else 0.0
    logger.info(f"Done: {len(audio_np)} samples ({total_audio_sec:.2f}s audio) in {elapsed:.2f}s | speed={xrt:.2f}x RT | RTF={rtf:.3f}")

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@app.post("/v1/audio/speech/stream")
async def create_speech_streaming(request: AudioSpeechRequest):
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    runtime = _resolve_runtime_override(request.runtime)
    try:
        model = _get_or_load_model_for_runtime(runtime)
    except Exception as e:
        logger.error(f"Runtime load failed for runtime={runtime}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load OmniVoice runtime '{runtime}': {e}")

    effective_num_step = request.num_step

    async def event_generator():
        import base64

        try:
            chunks = _split_text_into_chunks(
                request.input,
                max_tokens=request.max_tokens,
                token_method=request.token_method,
            )
            if not chunks:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No valid text to synthesize'})}\n\n"
                return

            prepared_ref_text = _prepare_reference_text_once(request.voice, request.ref_text)

            yield f"data: {json.dumps({'type': 'start', 'chunks': len(chunks), 'sample_rate': TARGET_SAMPLE_RATE, 'runtime': runtime})}\n\n"

            total_duration = 0.0
            start = time.perf_counter()
            sent = 0

            for idx, chunk in enumerate(chunks):
                try:
                    chunk_start = time.perf_counter()
                    audio_np = _generate_chunk_audio(
                        model,
                        chunk,
                        request.voice,
                        prepared_ref_text,
                        request.speed,
                        effective_num_step,
                    )
                    if len(audio_np) == 0:
                        continue

                    audio_buffer = io.BytesIO()
                    scipy.io.wavfile.write(audio_buffer, TARGET_SAMPLE_RATE, audio_np)
                    audio_bytes = audio_buffer.getvalue()
                    duration = len(audio_np) / TARGET_SAMPLE_RATE
                    total_duration += duration
                    gen_time = time.perf_counter() - chunk_start
                    sent += 1
                    chunk_xrt = (duration / gen_time) if gen_time > 0 else 0.0
                    logger.info(f"[stream {idx+1}/{len(chunks)}] {duration:.2f}s audio in {gen_time:.2f}s ({chunk_xrt:.2f}x RT)")

                    event = {
                        "type": "chunk",
                        "index": idx,
                        "audio_bytes_b64": base64.b64encode(audio_bytes).decode("utf-8"),
                        "duration": round(duration, 2),
                        "generation_time": round(gen_time, 2),
                        "text_preview": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception as e:
                    logger.error(f"Chunk {idx + 1} failed: {e}")
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Chunk {idx + 1} failed: {str(e)}'})}\n\n"

            if sent == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No audio generated'})}\n\n"
                return

            total_time = time.perf_counter() - start
            xrt = (total_duration / total_time) if total_time > 0 else 0.0
            rtf = (total_time / total_duration) if total_duration > 0 else 0.0
            logger.info(f"[stream complete] chunks={sent}, audio={total_duration:.2f}s, wall={total_time:.2f}s, speed={xrt:.2f}x RT, RTF={rtf:.3f}")
            yield f"data: {json.dumps({'type': 'complete', 'chunks_sent': sent, 'audio_duration': round(total_duration, 2), 'total_time': round(total_time, 2), 'xrt': round(xrt, 2), 'rtf': round(rtf, 3)})}\n\n"
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(description="OmniVoice Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8898, help="Port to bind to")
    parser.add_argument(
        "--proxy-headers",
        action="store_true",
        default=True,
        help="Trust X-Forwarded-* headers from reverse proxy (Tailscale, nginx, etc.)",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default="*",
        help="IPs allowed to send forwarded headers",
    )
    args = parser.parse_args()

    logger.info(f"Starting OmniVoice Server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level=os.getenv("VF_UVICORN_LOG_LEVEL", "warning").lower(),
        access_log=_env_flag("VF_ACCESS_LOGS", "0"),
    )
