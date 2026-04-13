# asr_server.py
"""
Unified ASR microservice supporting multiple backends:
- Whisper (Faster Whisper / CTranslate2)
- GLM-ASR (zai-org/GLM-ASR-Nano-2512)
- Parakeet (NVIDIA Parakeet models via transformers)

Runs in the 'asr' conda environment.
The main VoiceForge server calls this via HTTP.
"""
import os
import sys
from pathlib import Path

# This file is in app/servers/, set up paths
SCRIPT_DIR = Path(__file__).parent  # app/servers
APP_DIR = SCRIPT_DIR.parent  # app
UTIL_DIR = APP_DIR / "util"  # app/util
CONFIG_DIR = APP_DIR / "config"  # app/config
MODELS_DIR = APP_DIR / "models"  # app/models

# Add paths for imports
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(UTIL_DIR))
sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(MODELS_DIR))

# Set up logging BEFORE any other imports
from logging_utils import create_server_logger, suppress_all_logging

# Create server-specific logging functions
log_info, log_warn, log_error = create_server_logger("ASR")

log_info("Starting unified ASR server...")

import warnings
warnings.filterwarnings("ignore")

import logging
import tempfile
import traceback
import time
import platform
from contextlib import contextmanager

# Suppress all library logging
suppress_all_logging()

# Windows-specific fix for temp file cleanup issues
if platform.system() == "Windows":
    _original_temp_dir_cleanup = tempfile.TemporaryDirectory.cleanup
    
    def _safe_temp_cleanup(self):
        """Wrapper that ignores Windows file lock errors during cleanup."""
        try:
            _original_temp_dir_cleanup(self)
        except (PermissionError, NotADirectoryError, OSError):
            pass
    
    tempfile.TemporaryDirectory.cleanup = _safe_temp_cleanup
    log_info("Applied Windows temp file cleanup fix")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import soundfile as sf
import numpy as np
from typing import Optional, Literal, AsyncGenerator
import threading
import json
import asyncio
import queue
import io
import base64


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Check PyTorch/CUDA
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    log_info(f"CUDA available: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
else:
    log_warn("=" * 60)
    log_warn("CUDA NOT AVAILABLE - ASR will run on CPU (slow!)")
    log_warn("To fix: Install CUDA-enabled PyTorch in asr env:")
    log_warn("  conda activate asr")
    log_warn("  pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    log_warn("=" * 60)

# UVR5 vocal cleaning (preprocess microservice)
try:
    from clients import clean_vocals_uvr5, is_preprocess_server_available
    PREPROCESS_CLIENT_AVAILABLE = True
except Exception as e:
    PREPROCESS_CLIENT_AVAILABLE = False

# Postprocess client for audio enhancement
try:
    from clients import run_postprocess, is_postprocess_server_available
    POSTPROCESS_CLIENT_AVAILABLE = True
except Exception as e:
    POSTPROCESS_CLIENT_AVAILABLE = False
    log_warn(f"Postprocess client not available: {e}")

# Thread pool for GPU operations
# With sufficient VRAM, 2 workers allows one to prepare while another transcribes
from concurrent.futures import ThreadPoolExecutor
ASR_WORKERS = int(os.getenv("ASR_WORKERS", "2"))
_cuda_executor = ThreadPoolExecutor(max_workers=ASR_WORKERS, thread_name_prefix="cuda_worker")

# Live call mode safety guard: force transcription if silence is never detected.
LIVE_CALL_MAX_BUFFER_SECONDS = float(os.getenv("ASR_LIVE_CALL_MAX_BUFFER_SECONDS", "20"))
LIVE_CALL_MIN_AUDIO_SECONDS = float(os.getenv("ASR_LIVE_CALL_MIN_AUDIO_SECONDS", "0.3"))


# ==========================================
# WHISPER BACKEND
# ==========================================

FASTER_WHISPER_AVAILABLE = False
FASTER_WHISPER_ERROR = None
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    log_info("Faster Whisper backend available")
except ImportError as e:
    FASTER_WHISPER_ERROR = str(e)
    log_warn(f"Faster Whisper not installed: {e}")

# Whisper model management
_whisper_model = None
_whisper_model_name = None
_whisper_model_lock = threading.Lock()

# Available Faster Whisper models
FASTER_WHISPER_MODELS = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "large-v3",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}
DEFAULT_WHISPER_MODEL = "large-v3-turbo"


def load_whisper_model(model_name: str = None):
    """Load Faster Whisper model (thread-safe, lazy). Can switch models."""
    global _whisper_model, _whisper_model_name
    
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(f"Faster Whisper not available: {FASTER_WHISPER_ERROR}")
    
    if model_name is None:
        model_name = DEFAULT_WHISPER_MODEL
    
    # Map friendly names to model identifiers
    model_id = FASTER_WHISPER_MODELS.get(model_name, model_name)
    
    with _whisper_model_lock:
        # Check if we need to switch models
        if _whisper_model is not None and _whisper_model_name != model_name:
            log_info(f"Switching Whisper model from {_whisper_model_name} to {model_name}...")
            del _whisper_model
            _whisper_model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if _whisper_model is None:
            log_info(f"Loading Faster Whisper model: {model_name} ({model_id})")
            t_start = time.perf_counter()
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            _whisper_model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                download_root=str(APP_DIR / "models" / "whisper"),
            )
            
            _whisper_model_name = model_name
            t_load = time.perf_counter() - t_start
            log_info(f"Whisper model loaded in {t_load:.1f}s on {device.upper()} ({compute_type})")
        
        return _whisper_model


def unload_whisper_model():
    """Unload Whisper model and free GPU memory."""
    global _whisper_model, _whisper_model_name
    
    freed_gb = 0.0
    with _whisper_model_lock:
        if _whisper_model is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated() / 1e9
            
            del _whisper_model
            _whisper_model = None
            _whisper_model_name = None
            
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated() / 1e9
                freed_gb = before - after
            
            log_info(f"Whisper model unloaded, freed {freed_gb:.2f}GB")
    
    return freed_gb


def transcribe_with_whisper(audio_path: str, model_name: str = None, language: str = "en") -> dict:
    """Transcribe audio using Faster Whisper."""
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(f"Faster Whisper not available: {FASTER_WHISPER_ERROR}")
    
    model = load_whisper_model(model_name)
    
    audio_info = sf.info(audio_path)
    total_duration = audio_info.duration
    
    log_info(f"[WHISPER] Transcribing: {audio_path} ({total_duration:.1f}s)")
    t_start = time.perf_counter()
    
    segments_iter, info = model.transcribe(
        audio_path,
        language=language if language != "auto" else None,
        beam_size=5,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )
    
    segments = []
    full_text_parts = []
    last_end_time = 0
    last_text = ""
    repeat_count = 0
    max_repeats = 3
    max_expected_segments = int(total_duration / 0.5) + 50
    
    for segment in segments_iter:
        # Safety checks for loops
        if segment.end <= last_end_time and len(segments) > 5:
            continue
        
        if segment.text.strip() == last_text and last_text:
            repeat_count += 1
            if repeat_count >= max_repeats:
                continue
        else:
            repeat_count = 0
        
        last_end_time = segment.end
        last_text = segment.text.strip()
        
        segments.append({
            "id": len(segments),
            "seek": int(segment.seek),
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        })
        full_text_parts.append(segment.text.strip())
        
        if len(segments) > max_expected_segments:
            log_warn(f"[WHISPER] Breaking: too many segments ({len(segments)})")
            break
    
    full_text = " ".join(full_text_parts)
    
    t_elapsed = time.perf_counter() - t_start
    rtf = t_elapsed / total_duration if total_duration > 0 else 0
    log_info(f"[WHISPER] Done: {len(full_text)} chars, RTF={rtf:.2f}")
    
    return {
        "text": full_text,
        "segments": segments,
        "language": info.language,
        "duration": total_duration,
    }


# ==========================================
# GLM-ASR BACKEND
# ==========================================

GLM_ASR_AVAILABLE = False
GLM_ASR_ERROR = None
try:
    from transformers import AutoProcessor, AutoModelForSeq2SeqLM
    GLM_ASR_AVAILABLE = True
    log_info("GLM-ASR backend available")
except ImportError as e:
    GLM_ASR_ERROR = str(e)
    log_warn(f"GLM-ASR (transformers) not installed: {e}")

PARAKEET_AVAILABLE = False
PARAKEET_ERROR = None
try:
    import nemo.collections.asr as nemo_asr
    PARAKEET_AVAILABLE = True
    log_info("Parakeet backend available")
except Exception as e:
    PARAKEET_ERROR = str(e)
    log_warn(f"Parakeet backend not available (NeMo init failed): {e}")

# GLM model management
_glm_model = None
_glm_processor = None
_glm_model_name = None
_glm_model_lock = threading.Lock()

_parakeet_model = None
_parakeet_model_name = None
_parakeet_model_lock = threading.Lock()

# Model configuration
GLM_ASR_MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
DEFAULT_GLM_MODEL = "glm-asr-nano"

# GLM-ASR memory/perf tuning knobs
# - none: full precision weights on GPU
# - 8bit: ~40-50% less VRAM, requires bitsandbytes
# - 4bit: ~60-75% less VRAM, requires bitsandbytes (best-effort quality)
GLM_ASR_QUANTIZATION = os.getenv("GLM_ASR_QUANTIZATION", "none").strip().lower()
GLM_ASR_DTYPE = os.getenv("GLM_ASR_DTYPE", "auto").strip().lower()
GLM_ASR_USE_CACHE = _env_flag("GLM_ASR_USE_CACHE", "0")
GLM_ASR_GPU_MEMORY_GB = float(os.getenv("GLM_ASR_GPU_MEMORY_GB", "0") or 0)
GLM_ASR_MAX_NEW_TOKENS = int(os.getenv("GLM_ASR_MAX_NEW_TOKENS", "500") or 500)

PARAKEET_MODELS = {
    "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "parakeet-rnnt-1.1b": "nvidia/parakeet-rnnt-1.1b",
}
DEFAULT_PARAKEET_MODEL = "parakeet-tdt-0.6b-v2"


def load_glm_model(model_name: str = None):
    """Load GLM-ASR model (thread-safe, lazy). Can switch models."""
    global _glm_model, _glm_processor, _glm_model_name
    
    if not GLM_ASR_AVAILABLE:
        raise RuntimeError(f"GLM-ASR not available: {GLM_ASR_ERROR}")
    
    if model_name is None:
        model_name = DEFAULT_GLM_MODEL
    
    with _glm_model_lock:
        # Check if we need to switch models
        if _glm_model is not None and _glm_model_name != model_name:
            log_info(f"Switching GLM-ASR model from {_glm_model_name} to {model_name}...")
            del _glm_model
            del _glm_processor
            _glm_model = None
            _glm_processor = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if _glm_model is None:
            log_info(f"Loading GLM-ASR model: {GLM_ASR_MODEL_ID}")
            t_start = time.perf_counter()
            
            # Load processor
            _glm_processor = AutoProcessor.from_pretrained(
                GLM_ASR_MODEL_ID,
                cache_dir=str(APP_DIR / "models" / "glm_asr"),
                trust_remote_code=True
            )
            
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "auto": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            }
            dtype = dtype_map.get(GLM_ASR_DTYPE, torch.float16) if device == "cuda" else torch.float32

            model_kwargs = {
                "cache_dir": str(APP_DIR / "models" / "glm_asr"),
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if device == "cuda":
                model_kwargs["device_map"] = "auto"

                if GLM_ASR_GPU_MEMORY_GB > 0:
                    model_kwargs["max_memory"] = {0: f"{GLM_ASR_GPU_MEMORY_GB:.0f}GiB", "cpu": "48GiB"}
                    model_kwargs["offload_folder"] = str(APP_DIR / "models" / "glm_asr" / "offload")
                    log_info(f"GLM-ASR GPU memory cap enabled: ~{GLM_ASR_GPU_MEMORY_GB:.0f}GiB (CPU offload active)")

                quant_mode = GLM_ASR_QUANTIZATION
                if quant_mode in {"8bit", "int8"}:
                    try:
                        from transformers import BitsAndBytesConfig
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                        model_kwargs["torch_dtype"] = torch.float16
                        log_info("GLM-ASR quantization enabled: 8-bit")
                    except Exception as e:
                        log_warn(f"8-bit quantization requested but unavailable ({e}); falling back to {GLM_ASR_DTYPE}")
                        model_kwargs["torch_dtype"] = dtype
                elif quant_mode in {"4bit", "int4"}:
                    try:
                        from transformers import BitsAndBytesConfig
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                        model_kwargs["torch_dtype"] = torch.float16
                        log_info("GLM-ASR quantization enabled: 4-bit (NF4)")
                    except Exception as e:
                        log_warn(f"4-bit quantization requested but unavailable ({e}); falling back to {GLM_ASR_DTYPE}")
                        model_kwargs["torch_dtype"] = dtype
                else:
                    model_kwargs["torch_dtype"] = dtype
            else:
                model_kwargs["torch_dtype"] = torch.float32

            _glm_model = AutoModelForSeq2SeqLM.from_pretrained(GLM_ASR_MODEL_ID, **model_kwargs)

            if device == "cpu":
                _glm_model = _glm_model.to(device)

            try:
                _glm_model.generation_config.use_cache = GLM_ASR_USE_CACHE
            except Exception:
                pass
            
            _glm_model_name = model_name
            t_load = time.perf_counter() - t_start
            log_info(
                f"GLM-ASR model loaded in {t_load:.1f}s on {device.upper()} "
                f"(quant={GLM_ASR_QUANTIZATION}, dtype={GLM_ASR_DTYPE}, use_cache={GLM_ASR_USE_CACHE}, "
                f"gpu_cap_gb={GLM_ASR_GPU_MEMORY_GB or 0})"
            )
        
        return _glm_model, _glm_processor


def unload_glm_model():
    """Unload GLM-ASR model and free GPU memory."""
    global _glm_model, _glm_processor, _glm_model_name
    
    freed_gb = 0.0
    with _glm_model_lock:
        if _glm_model is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated() / 1e9
            
            del _glm_model
            del _glm_processor
            _glm_model = None
            _glm_processor = None
            _glm_model_name = None
            
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated() / 1e9
                freed_gb = before - after
            
            log_info(f"GLM-ASR model unloaded, freed {freed_gb:.2f}GB")
    
    return freed_gb


def transcribe_with_glm(audio_path: str, model_name: str = None, language: str = "en") -> dict:
    """Transcribe audio using GLM-ASR-Nano."""
    if not GLM_ASR_AVAILABLE:
        raise RuntimeError(f"GLM-ASR not available: {GLM_ASR_ERROR}")
    
    model, processor = load_glm_model(model_name)
    
    # Get audio info
    audio_info = sf.info(audio_path)
    total_duration = audio_info.duration
    
    # Load and preprocess audio
    audio_data, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample to processor's expected sample rate if needed
    target_sr = processor.feature_extractor.sampling_rate
    if sr != target_sr:
        import torchaudio.functional as F
        audio_tensor = torch.from_numpy(audio_data).float()
        audio_tensor = F.resample(audio_tensor, sr, target_sr)
        audio_data = audio_tensor.numpy()
    
    log_info(f"[GLM-ASR] Transcribing: {audio_path} ({total_duration:.1f}s)")
    t_start = time.perf_counter()
    
    # Process with GLM-ASR
    inputs = processor.apply_transcription_request(audio_data)
    inputs = inputs.to(model.device, dtype=model.dtype)
    
    # Generate transcription
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=GLM_ASR_MAX_NEW_TOKENS,
            use_cache=GLM_ASR_USE_CACHE,
        )
        
        # Decode output
        decoded = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
    del outputs
    del inputs
    
    full_text = decoded[0] if decoded else ""
    
    t_elapsed = time.perf_counter() - t_start
    rtf = t_elapsed / total_duration if total_duration > 0 else 0
    log_info(f"[GLM-ASR] Done: {len(full_text)} chars, RTF={rtf:.2f}")
    
    # GLM-ASR doesn't provide segment-level timestamps by default
    segments = [{
        "id": 0,
        "start": 0.0,
        "end": total_duration,
        "text": full_text.strip(),
    }]
    
    return {
        "text": full_text.strip(),
        "segments": segments,
        "language": language,
        "duration": total_duration,
    }


def load_parakeet_model(model_name: str = None):
    """Load Parakeet ASR model via NeMo (thread-safe, lazy). CPU-focused backend."""
    global _parakeet_model, _parakeet_model_name

    if not PARAKEET_AVAILABLE:
        raise RuntimeError(f"Parakeet backend not available: {PARAKEET_ERROR}")

    if model_name is None:
        model_name = DEFAULT_PARAKEET_MODEL

    model_id = PARAKEET_MODELS.get(model_name, model_name)

    with _parakeet_model_lock:
        if _parakeet_model is not None and _parakeet_model_name != model_name:
            log_info(f"Switching Parakeet model from {_parakeet_model_name} to {model_name}...")
            del _parakeet_model
            _parakeet_model = None
            import gc
            gc.collect()

        if _parakeet_model is None:
            log_info(f"Loading Parakeet model: {model_name} ({model_id}) on CPU")
            t_start = time.perf_counter()

            try:
                _parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_id,
                    map_location=torch.device("cpu"),
                )
                _parakeet_model = _parakeet_model.eval()
            except Exception as e:
                err_text = str(e)
                if "Unrecognized model" in err_text or "model_type" in err_text:
                    raise RuntimeError(
                        "Parakeet models are NeMo checkpoints (.nemo), not transformers-native models. "
                        "Install NeMo ASR in the ASR environment: pip install -U nemo_toolkit[asr]"
                    )
                raise

            _parakeet_model_name = model_name
            t_load = time.perf_counter() - t_start
            log_info(f"Parakeet model loaded in {t_load:.1f}s on CPU")

        return _parakeet_model


def unload_parakeet_model():
    """Unload Parakeet model and free memory."""
    global _parakeet_model, _parakeet_model_name

    with _parakeet_model_lock:
        if _parakeet_model is not None:
            del _parakeet_model
            _parakeet_model = None
            _parakeet_model_name = None

            import gc
            gc.collect()

            log_info("Parakeet model unloaded")

    return 0.0


def transcribe_with_parakeet(audio_path: str, model_name: str = None, language: str = "en") -> dict:
    """Transcribe audio using Parakeet on CPU."""
    if not PARAKEET_AVAILABLE:
        raise RuntimeError(f"Parakeet backend not available: {PARAKEET_ERROR}")

    asr_model = load_parakeet_model(model_name)

    audio_info = sf.info(audio_path)
    total_duration = audio_info.duration

    log_info(f"[PARAKEET] Transcribing: {audio_path} ({total_duration:.1f}s)")
    t_start = time.perf_counter()

    segments = []
    full_text = ""

    result = asr_model.transcribe([audio_path], timestamps=True)
    first = result[0] if isinstance(result, list) and result else result

    if isinstance(first, str):
        full_text = first.strip()
    elif isinstance(first, dict):
        full_text = (first.get("text") or "").strip()
        seg_ts = ((first.get("timestamp") or {}).get("segment") or []) if isinstance(first.get("timestamp"), dict) else []
        for idx, seg in enumerate(seg_ts):
            text = (seg.get("segment") or seg.get("text") or "").strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", total_duration) or total_duration)
            segments.append({"id": idx, "start": start, "end": end, "text": text})
    else:
        # NeMo Hypothesis-style object
        full_text = str(getattr(first, "text", "") or "").strip()
        ts = getattr(first, "timestamp", None)
        if isinstance(ts, dict):
            seg_ts = ts.get("segment") or []
            for idx, seg in enumerate(seg_ts):
                text = (seg.get("segment") or seg.get("text") or "").strip()
                if not text:
                    continue
                start = float(seg.get("start", 0.0) or 0.0)
                end = float(seg.get("end", total_duration) or total_duration)
                segments.append({"id": idx, "start": start, "end": end, "text": text})

    if not segments and full_text:
        segments = [{
            "id": 0,
            "start": 0.0,
            "end": total_duration,
            "text": full_text,
        }]

    t_elapsed = time.perf_counter() - t_start
    rtf = t_elapsed / total_duration if total_duration > 0 else 0
    log_info(f"[PARAKEET] Done: {len(full_text)} chars, RTF={rtf:.2f}")

    return {
        "text": full_text,
        "segments": segments,
        "language": language,
        "duration": total_duration,
    }


# ==========================================
# UNIFIED TRANSCRIPTION ROUTER
# ==========================================

def is_glm_model(model_name: str) -> bool:
    """Check if model name refers to GLM-ASR backend."""
    if not model_name:
        return False
    return model_name.lower().startswith("glm")


def is_parakeet_model(model_name: str) -> bool:
    """Check if model name refers to Parakeet backend."""
    if not model_name:
        return False
    return model_name.lower().startswith("parakeet")


def is_removed_glm_gguf_model(model_name: str) -> bool:
    """Check if model name refers to removed GLM-ASR GGUF backend."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return model_lower.startswith("glm-gguf") or model_lower.startswith("glm-asr-nano-gguf")


def transcribe(audio_path: str, model_name: str = None, language: str = "en") -> dict:
    """
    Unified transcription function that routes to the appropriate backend.
    
    Model name prefixes:
    - "glm-*" or "glm_*" -> GLM-ASR
    - "parakeet-*" -> Parakeet
    - "whisper-*" or anything else -> Whisper
    """
    if model_name is None:
        model_name = f"whisper-{DEFAULT_WHISPER_MODEL}"

    if is_removed_glm_gguf_model(model_name):
        raise RuntimeError(
            "GLM-ASR GGUF backend has been removed. Use model='glm-asr-nano' instead."
        )
    elif is_glm_model(model_name):
        # Use GLM-ASR
        if not GLM_ASR_AVAILABLE:
            raise RuntimeError(f"GLM-ASR not available: {GLM_ASR_ERROR}")
        return transcribe_with_glm(audio_path, model_name, language)
    elif is_parakeet_model(model_name):
        # Use Parakeet
        if not PARAKEET_AVAILABLE:
            raise RuntimeError(f"Parakeet backend not available: {PARAKEET_ERROR}")
        return transcribe_with_parakeet(audio_path, model_name, language)
    else:
        # Use Whisper
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(f"Whisper not available: {FASTER_WHISPER_ERROR}")
        
        # Strip "whisper-" prefix if present
        whisper_model = model_name
        if whisper_model.startswith("whisper-"):
            whisper_model = whisper_model.replace("whisper-", "")
        
        return transcribe_with_whisper(audio_path, whisper_model, language)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _clean_vocals(audio_path: str, skip_if_exists: bool = True, original_filename: str = None) -> str:
    """Clean vocals using preprocess microservice (UVR5)."""
    if not PREPROCESS_CLIENT_AVAILABLE:
        log_warn("Preprocess client not available, skipping vocal cleaning")
        return audio_path

    if not is_preprocess_server_available():
        log_warn("Preprocess server not available, skipping vocal cleaning")
        return audio_path

    try:
        vocals_path = clean_vocals_uvr5(
            audio_path, 
            aggression=10, 
            device=None,
            skip_if_cached=skip_if_exists,
            original_filename=original_filename,
        )
        return vocals_path
    except Exception as e:
        log_warn(f"Vocal cleaning failed: {e}")
        return audio_path


def _postprocess_audio(audio_path: str, params: dict = None) -> str:
    """Apply post-processing to audio before transcription."""
    if not POSTPROCESS_CLIENT_AVAILABLE:
        log_warn("Postprocess client not available, skipping audio enhancement")
        return audio_path

    if not is_postprocess_server_available():
        log_warn("Postprocess server not available, skipping audio enhancement")
        return audio_path

    try:
        if params is None:
            params = {
                "highpass": 80.0,
                "lowpass": 12000.0,
                "bass_freq": 60.0,
                "bass_gain": 0.0,
                "treble_freq": 8000.0,
                "treble_gain": 0.0,
                "reverb_delay": 0.0,
                "reverb_decay": 0.0,
                "crystalizer": 0.0,
                "deesser": 0.3,
            }
        
        log_info(f"[POSTPROCESS] Enhancing audio before transcription...")
        processed_path = run_postprocess(audio_path, params)
        log_info(f"[POSTPROCESS] Audio enhanced: {processed_path}")
        return processed_path
    except Exception as e:
        log_warn(f"Audio post-processing failed: {e}")
        return audio_path


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="VoiceForge ASR Server",
    description="Unified speech-to-text transcription (Whisper + GLM-ASR + Parakeet)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    whisper_available: bool
    whisper_model_loaded: bool
    whisper_model_name: Optional[str]
    glm_asr_available: bool
    glm_model_loaded: bool
    glm_model_name: Optional[str]
    parakeet_available: bool
    parakeet_model_loaded: bool
    parakeet_model_name: Optional[str]
    cuda_available: bool
    gpu_name: Optional[str]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
    
    return HealthResponse(
        status="healthy",
        whisper_available=FASTER_WHISPER_AVAILABLE,
        whisper_model_loaded=_whisper_model is not None,
        whisper_model_name=_whisper_model_name,
        glm_asr_available=GLM_ASR_AVAILABLE,
        glm_model_loaded=_glm_model is not None,
        glm_model_name=_glm_model_name,
        parakeet_available=PARAKEET_AVAILABLE,
        parakeet_model_loaded=_parakeet_model is not None,
        parakeet_model_name=_parakeet_model_name,
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
    )


@app.post("/warmup")
async def warmup(model: str = None):
    """Pre-load ASR model for faster first inference."""
    if model and is_removed_glm_gguf_model(model):
        raise HTTPException(
            status_code=400,
            detail="GLM-ASR GGUF backend has been removed. Use model='glm-asr-nano' instead.",
        )

    try:
        if model and is_glm_model(model):
            load_glm_model()
            return {"status": "ok", "message": "GLM-ASR model loaded", "backend": "glm"}
        elif model and is_parakeet_model(model):
            load_parakeet_model(model)
            return {"status": "ok", "message": "Parakeet model loaded", "backend": "parakeet"}
        else:
            load_whisper_model()
            return {"status": "ok", "message": "Whisper model loaded", "backend": "whisper"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu_memory")
async def gpu_memory():
    """Get GPU memory usage."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    return {
        "cuda_available": True,
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


@app.post("/clear_gpu_cache")
async def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1e9
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        after = torch.cuda.memory_allocated() / 1e9
        return {"freed_gb": round(before - after, 2), "current_gb": round(after, 2)}
    return {"message": "CUDA not available"}


@app.post("/unload")
async def unload_model(backend: str = None):
    """Unload ASR model(s) and free GPU memory."""
    freed = 0.0
    messages = []
    
    if backend is None or backend == "whisper":
        freed += unload_whisper_model()
        messages.append("Whisper unloaded")
    
    if backend is None or backend == "glm":
        freed += unload_glm_model()
        messages.append("GLM-ASR unloaded")

    if backend is None or backend == "parakeet":
        freed += unload_parakeet_model()
        messages.append("Parakeet unloaded")

    return {"success": True, "message": ", ".join(messages), "freed_gb": round(freed, 2)}


@app.get("/model_info")
async def get_model_info():
    """Get information about loaded models."""
    return {
        "whisper": {
            "available": FASTER_WHISPER_AVAILABLE,
            "loaded": _whisper_model is not None,
            "model_name": _whisper_model_name,
            "available_models": list(FASTER_WHISPER_MODELS.keys()),
        },
        "glm_asr": {
            "available": GLM_ASR_AVAILABLE,
            "loaded": _glm_model is not None,
            "model_name": _glm_model_name,
            "model_id": GLM_ASR_MODEL_ID,
            "available_models": ["glm-asr-nano"],
        },
        "parakeet": {
            "available": PARAKEET_AVAILABLE,
            "loaded": _parakeet_model is not None,
            "model_name": _parakeet_model_name,
            "available_models": list(PARAKEET_MODELS.keys()),
        },
    }


# ==========================================
# LIVE TRANSCRIPTION WEBSOCKET
# ==========================================

@app.websocket("/v1/audio/transcriptions/live")
async def live_transcription_websocket(
    websocket: WebSocket, 
    model: str = None, 
    language: str = "en",
    call_mode: bool = False,
    silence_threshold: float = 1.5
):
    """
    WebSocket endpoint for real-time live transcription.
    
    Parameters:
        model: ASR model to use (whisper-* or glm-*)
        language: Language code (default: en)
        call_mode: If True, accumulates audio and only sends when silence detected
        silence_threshold: Seconds of silence before sending transcript (call_mode only)
    
    Client sends: { "type": "audio", "data": "<base64 Float32Array>" }
    Client sends: { "type": "end" } to finish
    
    Server sends: { "type": "ready" }
    Server sends: { "type": "partial", "text": "..." } (not in call_mode)
    Server sends: { "type": "transcript", "text": "..." }
    Server sends: { "type": "complete", "text": "..." }
    Server sends: { "type": "error", "message": "..." }
    """
    await websocket.accept()
    log_info(f"[LIVE] WebSocket connected, model={model}, language={language}, call_mode={call_mode}")
    
    effective_model = model or f"whisper-{DEFAULT_WHISPER_MODEL}"
    use_glm = is_glm_model(effective_model)
    use_parakeet = is_parakeet_model(effective_model)
    
    # Check backend availability
    if is_removed_glm_gguf_model(effective_model):
        await websocket.send_json({
            "type": "error",
            "message": "GLM-ASR GGUF backend has been removed. Use model='glm-asr-nano' instead.",
        })
        await websocket.close()
        return
    if use_glm and not GLM_ASR_AVAILABLE:
        await websocket.send_json({"type": "error", "message": f"GLM-ASR not available: {GLM_ASR_ERROR}"})
        await websocket.close()
        return
    if use_parakeet and not PARAKEET_AVAILABLE:
        await websocket.send_json({"type": "error", "message": f"Parakeet not available: {PARAKEET_ERROR}"})
        await websocket.close()
        return
    if not use_glm and not use_parakeet and not FASTER_WHISPER_AVAILABLE:
        await websocket.send_json({"type": "error", "message": f"Whisper not available: {FASTER_WHISPER_ERROR}"})
        await websocket.close()
        return

    # Fail fast on model load errors so user gets clear message before streaming.
    if use_parakeet:
        try:
            load_parakeet_model(effective_model)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": f"Parakeet model load failed: {e}"})
            await websocket.close()
            return
    
    await websocket.send_json({"type": "ready", "call_mode": call_mode})
    
    # Accumulate audio chunks
    audio_chunks = []
    sample_rate = 16000  # Default, updated from client
    live_start = time.perf_counter()
    audio_messages = 0
    audio_samples_received = 0
    transcripts_sent = 0
    first_transcript_at = None
    
    # Client-side endpointing support
    client_flushes = 0
    
    # Process audio in ~3 second intervals (for non-call mode)
    chunk_interval = 3.0  # seconds
    samples_per_chunk = int(chunk_interval * sample_rate)
    accumulated_samples = 0
    full_transcript = []

    async def _flush_call_mode_buffer(trigger_reason: str):
        nonlocal audio_chunks, accumulated_samples, transcripts_sent, first_transcript_at

        if not audio_chunks:
            return

        audio_data = np.concatenate(audio_chunks)
        min_samples = int(max(LIVE_CALL_MIN_AUDIO_SECONDS, 0.1) * sample_rate)
        if len(audio_data) <= min_samples:
            audio_chunks = []
            accumulated_samples = 0
            return

        log_info(
            f"[LIVE-CALL] Processing {len(audio_data)/sample_rate:.1f}s of audio ({trigger_reason})..."
        )
        text = await _transcribe_audio_chunk(audio_data, sample_rate, effective_model, language)
        if text and text.strip():
            log_info(f"[LIVE-CALL] Transcript: {text.strip()[:50]}...")
            transcripts_sent += 1
            if first_transcript_at is None:
                first_transcript_at = time.perf_counter() - live_start
            await websocket.send_json({
                "type": "transcript",
                "text": text.strip()
            })

        audio_chunks = []
        accumulated_samples = 0
    
    try:
        while True:
            message = await websocket.receive_json()
            
            msg_type = message.get("type")
            
            if msg_type == "end":
                log_info("[LIVE] Client sent end signal")
                break

            elif msg_type == "flush":
                if call_mode:
                    client_flushes += 1
                    await _flush_call_mode_buffer("client_flush")
                continue
            
            elif msg_type == "audio":
                # Decode base64 Float32Array
                try:
                    audio_messages += 1
                    audio_b64 = message.get("data", "")
                    # Client always sends 16kHz audio
                    sample_rate = 16000
                    
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)
                    audio_chunks.append(audio_float32)
                    accumulated_samples += len(audio_float32)
                    audio_samples_received += len(audio_float32)
                    
                    if call_mode:
                        # Call mode requires explicit client flush endpointing.
                        pass
                    
                    else:
                        # Standard mode: process in chunks with partial updates
                        if accumulated_samples >= samples_per_chunk:
                            audio_data = np.concatenate(audio_chunks)
                            audio_chunks = []
                            accumulated_samples = 0
                            
                            chunk_text = await _transcribe_audio_chunk(
                                audio_data, sample_rate, effective_model, language
                            )
                            
                            if chunk_text and chunk_text.strip():
                                full_transcript.append(chunk_text.strip())
                                current_text = " ".join(full_transcript)
                                transcripts_sent += 1
                                if first_transcript_at is None:
                                    first_transcript_at = time.perf_counter() - live_start
                                
                                await websocket.send_json({
                                    "type": "partial",
                                    "text": chunk_text.strip()
                                })
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": current_text
                                })
                
                except Exception as e:
                    log_error(f"[LIVE] Audio/process error: {e}")
                    await websocket.send_json({"type": "error", "message": f"Audio/process error: {e}"})
        
        # Process any remaining audio
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
            if len(audio_data) > sample_rate * 0.5:  # At least 0.5 seconds
                chunk_text = await _transcribe_audio_chunk(
                    audio_data, sample_rate, effective_model, language
                )
                if chunk_text and chunk_text.strip():
                    if call_mode:
                        # In call mode, send as final transcript
                        transcripts_sent += 1
                        if first_transcript_at is None:
                            first_transcript_at = time.perf_counter() - live_start
                        await websocket.send_json({
                            "type": "transcript",
                            "text": chunk_text.strip()
                        })
                    else:
                        full_transcript.append(chunk_text.strip())
        
        # Send final result
        if not call_mode:
            final_text = " ".join(full_transcript)
            log_info(f"[LIVE] Complete: {len(final_text)} chars")
            await websocket.send_json({
                "type": "complete",
                "text": final_text
            })
        else:
            await websocket.send_json({"type": "complete", "text": ""})
        
    except WebSocketDisconnect:
        log_info("[LIVE] WebSocket disconnected")
    except Exception as e:
        log_error(f"[LIVE] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        elapsed = time.perf_counter() - live_start
        ttft = first_transcript_at if first_transcript_at is not None else elapsed
        duration_seconds = audio_samples_received / sample_rate if sample_rate else 0.0
        log_info(
            f"[LIVE] METRICS call_mode={call_mode} messages={audio_messages} flushes={client_flushes} "
            f"audio_seconds={duration_seconds:.2f} transcripts={transcripts_sent} "
            f"ttft_ms={ttft*1000:.0f} total_ms={elapsed*1000:.0f}"
        )
        try:
            await websocket.close()
        except:
            pass


async def _transcribe_audio_chunk(audio_data: np.ndarray, sample_rate: int, model_name: str, language: str) -> str:
    """Transcribe a chunk of audio data."""
    # Save to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    try:
        sf.write(temp_path, audio_data, sample_rate)
        
        # Run transcription in thread pool to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _cuda_executor,
            lambda: transcribe(temp_path, model_name, language)
        )
        
        return result.get("text", "")
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


# ==========================================
# STREAMING TRANSCRIPTION
# ==========================================

_streaming_queues = {}


def transcribe_streaming(
    audio_path: str,
    update_queue: queue.Queue,
    model_name: str = None,
    language: str = "en"
) -> str:
    """Transcribe audio with streaming updates to a queue."""
    
    def send_update(type_: str, data: dict):
        update_queue.put({"type": type_, **data})
    
    # Determine backend
    use_removed_glm_gguf = is_removed_glm_gguf_model(model_name) if model_name else False
    use_glm = is_glm_model(model_name) if model_name else False
    use_parakeet = is_parakeet_model(model_name) if model_name else False
    if use_removed_glm_gguf:
        raise RuntimeError("GLM-ASR GGUF backend has been removed. Use model='glm-asr-nano' instead.")
    
    # Load audio
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    total_duration = len(audio_data) / sr
    if use_glm:
        backend_name = "GLM-ASR"
    elif use_parakeet:
        backend_name = "PARAKEET"
    else:
        backend_name = "WHISPER"
    log_info(f"[{backend_name}-STREAM] Transcribing: {audio_path} ({total_duration:.1f}s)")
    t_start = time.perf_counter()

    # Chunked processing for real-time updates
    chunk_duration = 30.0  # seconds per chunk
    overlap_duration = 2.0  # overlap between chunks
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    
    all_text_parts = []
    processed_duration = 0.0
    chunk_idx = 0
    
    send_update("status", {"message": f"Processing {total_duration:.1f}s audio...", "progress": 10})
    
    if use_glm:
        # GLM-ASR streaming
        model, processor = load_glm_model(model_name)
        target_sr = processor.feature_extractor.sampling_rate
        
        if sr != target_sr:
            import torchaudio.functional as F
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = F.resample(audio_tensor, sr, target_sr)
            audio_data = audio_tensor.numpy()
            sr = target_sr
            chunk_samples = int(chunk_duration * sr)
            overlap_samples = int(overlap_duration * sr)
        
        pos = 0
        while pos < len(audio_data):
            chunk_end = min(pos + chunk_samples, len(audio_data))
            chunk = audio_data[pos:chunk_end]
            chunk_duration_actual = len(chunk) / sr
            
            try:
                inputs = processor.apply_transcription_request(chunk)
                inputs = inputs.to(model.device, dtype=model.dtype)
                
                with torch.inference_mode():
                    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
                    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                chunk_text = decoded[0].strip() if decoded else ""
                del outputs
                del inputs
                
                if chunk_text:
                    all_text_parts.append(chunk_text)
                    seg_data = {
                        "id": chunk_idx,
                        "start": processed_duration,
                        "end": processed_duration + chunk_duration_actual,
                        "text": chunk_text,
                    }
                    send_update("segment", {"segment": seg_data, "duration_processed": processed_duration + chunk_duration_actual, "total_duration": total_duration})
                    send_update("text", {"text": " ".join(all_text_parts), "chunk_text": chunk_text, "chunk_idx": chunk_idx, "duration_processed": processed_duration + chunk_duration_actual})
            except Exception as e:
                log_error(f"[GLM-ASR-STREAM] Chunk {chunk_idx} error: {e}")
            
            processed_duration += chunk_duration_actual - (overlap_duration if chunk_end < len(audio_data) else 0)
            progress = min(90, int(10 + (processed_duration / total_duration) * 80))
            send_update("progress", {"progress": progress, "duration_processed": processed_duration})
            
            pos += chunk_samples - overlap_samples
            chunk_idx += 1
    elif use_parakeet:
        # Parakeet streaming (CPU): transcribe full audio and emit one chunk.
        parakeet_result = transcribe_with_parakeet(audio_path, model_name, language)
        chunk_text = parakeet_result.get("text", "").strip()
        if chunk_text:
            all_text_parts.append(chunk_text)
            seg_data = {
                "id": 0,
                "start": 0.0,
                "end": total_duration,
                "text": chunk_text,
            }
            send_update("segment", {"segment": seg_data, "duration_processed": total_duration, "total_duration": total_duration})
            send_update("text", {"text": chunk_text, "chunk_text": chunk_text, "chunk_idx": 0, "duration_processed": total_duration})
        send_update("progress", {"progress": 90, "duration_processed": total_duration})
        chunk_idx = 1
    else:
        # Whisper streaming
        whisper_model = model_name.replace("whisper-", "") if model_name and model_name.startswith("whisper-") else (model_name or DEFAULT_WHISPER_MODEL)
        model = load_whisper_model(whisper_model)
        
        pos = 0
        while pos < len(audio_data):
            chunk_end = min(pos + chunk_samples, len(audio_data))
            chunk = audio_data[pos:chunk_end]
            chunk_duration_actual = len(chunk) / sr
            
            # Save chunk to temp file
            chunk_path = audio_path + f"_chunk{chunk_idx}.wav"
            sf.write(chunk_path, chunk, sr)
            
            try:
                segments_iter, info = model.transcribe(
                    chunk_path,
                    language=language if language != "auto" else None,
                    beam_size=5, best_of=3, temperature=0.0,
                    compression_ratio_threshold=2.4, no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
                )
                
                chunk_texts = []
                for segment in segments_iter:
                    text = segment.text.strip()
                    if text:
                        chunk_texts.append(text)
                        seg_data = {"id": len(all_text_parts) + len(chunk_texts) - 1, "start": processed_duration + segment.start, "end": processed_duration + segment.end, "text": text}
                        send_update("segment", {"segment": seg_data, "duration_processed": processed_duration + segment.end, "total_duration": total_duration})
                
                if chunk_texts:
                    chunk_text = " ".join(chunk_texts)
                    all_text_parts.append(chunk_text)
                    send_update("text", {"text": " ".join(all_text_parts), "chunk_text": chunk_text, "chunk_idx": chunk_idx, "duration_processed": processed_duration + chunk_duration_actual})
            finally:
                try:
                    os.unlink(chunk_path)
                except:
                    pass
            
            processed_duration += chunk_duration_actual - (overlap_duration if chunk_end < len(audio_data) else 0)
            progress = min(90, int(10 + (processed_duration / total_duration) * 80))
            send_update("progress", {"progress": progress, "duration_processed": processed_duration})
            
            pos += chunk_samples - overlap_samples
            chunk_idx += 1
    
    full_text = " ".join(all_text_parts)
    
    t_elapsed = time.perf_counter() - t_start
    rtf = t_elapsed / total_duration if total_duration > 0 else 0
    log_info(f"[{backend_name}-STREAM] Done: {len(full_text)} chars, {chunk_idx} chunks, RTF={rtf:.2f}")
    
    send_update("progress", {"progress": 100, "duration_processed": total_duration})
    send_update("complete", {"text": full_text})
    
    return full_text


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    clean_vocals: bool = Form(default=False),
    skip_existing_vocals: bool = Form(default=True),
    postprocess_audio: bool = Form(default=False),
    device: str = Form(default="gpu"),
    model: Optional[str] = Form(default=None)
):
    """Streaming transcription endpoint using Server-Sent Events."""
    
    effective_model = model or f"whisper-{DEFAULT_WHISPER_MODEL}"
    
    # Check backend availability
    if is_removed_glm_gguf_model(effective_model):
        raise HTTPException(
            status_code=400,
            detail="GLM-ASR GGUF backend has been removed. Use model='glm-asr-nano' instead.",
        )
    elif is_glm_model(effective_model):
        if not GLM_ASR_AVAILABLE:
            raise HTTPException(status_code=503, detail=f"GLM-ASR not available: {GLM_ASR_ERROR}")
    elif is_parakeet_model(effective_model):
        if not PARAKEET_AVAILABLE:
            raise HTTPException(status_code=503, detail=f"Parakeet not available: {PARAKEET_ERROR}")
    else:
        if not FASTER_WHISPER_AVAILABLE:
            raise HTTPException(status_code=503, detail=f"Whisper not available: {FASTER_WHISPER_ERROR}")
    
    async def generate() -> AsyncGenerator[str, None]:
        temp_path = None
        wav_path = None
        
        try:
            # Save uploaded file
            content = await file.read()
            fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or ".wav")[1])
            os.close(fd)
            with open(temp_path, "wb") as f:
                f.write(content)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing audio...', 'progress': 2})}\n\n"
            
            # Optional vocal cleaning
            audio_to_process = temp_path
            original_filename = file.filename or "audio"
            if clean_vocals:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Cleaning vocals...', 'progress': 5})}\n\n"
                vocals_path = _clean_vocals(temp_path, skip_if_exists=skip_existing_vocals, original_filename=original_filename)
                if vocals_path != temp_path:
                    audio_to_process = vocals_path
            
            # Optional post-processing
            if postprocess_audio:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Enhancing audio...', 'progress': 8})}\n\n"
                processed_path = _postprocess_audio(audio_to_process)
                if processed_path != audio_to_process:
                    audio_to_process = processed_path
            
            # Convert to mono WAV at 16kHz
            wav_path = temp_path + "_mono.wav"
            data, sr = sf.read(audio_to_process)
            
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            
            if sr != 16000:
                import torchaudio.functional as F
                data_tensor = torch.from_numpy(data).float()
                data_tensor = F.resample(data_tensor, sr, 16000)
                data = data_tensor.numpy()
                sr = 16000
            
            sf.write(wav_path, data, sr)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Transcribing...', 'progress': 10})}\n\n"
            
            # Set up queue for updates
            update_queue = queue.Queue()
            session_id = id(update_queue)
            _streaming_queues[session_id] = update_queue
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            
            def run_transcription():
                try:
                    transcribe_streaming(wav_path, update_queue, effective_model, language)
                except Exception as e:
                    log_error(f"Transcription error: {e}")
                    traceback.print_exc()
                    update_queue.put({"type": "error", "error": str(e)})
            
            future = loop.run_in_executor(_cuda_executor, run_transcription)
            
            # Stream updates
            while True:
                try:
                    update = await loop.run_in_executor(None, lambda: update_queue.get(timeout=0.1))
                    
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    if update.get("type") in ("complete", "error"):
                        break
                except queue.Empty:
                    if future.done():
                        break
                    continue
            
            await future
            
        except Exception as e:
            log_error(f"Stream error: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        finally:
            # Cleanup
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ==========================================
# MAIN TRANSCRIPTION ENDPOINT
# ==========================================

@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    response_format: Literal["json", "text", "verbose_json"] = Form(default="json"),
    clean_vocals: bool = Form(default=False),
    skip_existing_vocals: bool = Form(default=True),
    postprocess_audio: bool = Form(default=False),
    device: str = Form(default="gpu"),
    model: Optional[str] = Form(default=None)
):
    """
    Transcribe audio. Compatible with OpenAI API format.
    
    Supported models:
    - "glm-asr-nano": GLM-ASR-Nano-2512 via transformers (~5GB fp16/bf16, lower with 8/4-bit quantization)
    - "parakeet-tdt-0.6b-v2": NVIDIA Parakeet via NeMo (CPU)
    - "parakeet-tdt-0.6b-v3": NVIDIA Parakeet via NeMo (CPU)
    - "whisper-large-v3-turbo": Whisper large-v3-turbo (default)
    - "whisper-large-v3", "whisper-medium", "whisper-small", etc.
    """
    
    effective_model = model or f"whisper-{DEFAULT_WHISPER_MODEL}"
    
    # Check backend availability
    if is_removed_glm_gguf_model(effective_model):
        raise HTTPException(
            status_code=400,
            detail="GLM-ASR GGUF backend has been removed. Use model='glm-asr-nano' instead.",
        )
    elif is_glm_model(effective_model):
        if not GLM_ASR_AVAILABLE:
            raise HTTPException(status_code=503, detail=f"GLM-ASR not available: {GLM_ASR_ERROR}")
    elif is_parakeet_model(effective_model):
        if not PARAKEET_AVAILABLE:
            raise HTTPException(status_code=503, detail=f"Parakeet not available: {PARAKEET_ERROR}")
    else:
        if not FASTER_WHISPER_AVAILABLE:
            raise HTTPException(status_code=503, detail=f"Whisper not available: {FASTER_WHISPER_ERROR}")
    
    temp_path = None
    wav_path = None
    
    try:
        # Save uploaded file
        content = await file.read()
        fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or ".wav")[1])
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(content)
        
        log_info(f"Transcribing: {file.filename}, model={effective_model}, language={language}")
        
        # Optional vocal cleaning
        audio_to_process = temp_path
        original_filename = file.filename or "audio"
        if clean_vocals:
            log_info(f"Cleaning vocals for: {original_filename}")
            vocals_path = _clean_vocals(temp_path, skip_if_exists=skip_existing_vocals, original_filename=original_filename)
            if vocals_path != temp_path:
                audio_to_process = vocals_path
        
        # Optional post-processing
        if postprocess_audio:
            log_info(f"Enhancing audio before transcription...")
            processed_path = _postprocess_audio(audio_to_process)
            if processed_path != audio_to_process:
                audio_to_process = processed_path
        
        # Convert to mono WAV at 16kHz
        wav_path = temp_path + "_mono.wav"
        data, sr = sf.read(audio_to_process)
        
        log_info(f"Loaded audio: shape={data.shape}, dtype={data.dtype}, sr={sr}")
        
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        
        if sr != 16000:
            import torchaudio.functional as F
            data_tensor = torch.from_numpy(data).float()
            data_tensor = F.resample(data_tensor, sr, 16000)
            data = data_tensor.numpy()
            sr = 16000
        
        sf.write(wav_path, data, sr)
        
        # Transcribe using unified router
        t_start = time.perf_counter()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _cuda_executor,
            lambda: transcribe(wav_path, effective_model, language)
        )
        
        t_elapsed = time.perf_counter() - t_start
        log_info(f"Transcribed in {t_elapsed*1000:.0f}ms")
        
        text = result["text"]
        
        # Format response
        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": result.get("language", language),
                "duration": result.get("duration"),
                "text": text,
                "segments": result.get("segments", []),
            }
        else:  # json
            return {"text": text}
    
    except Exception as e:
        log_error(f"Transcription error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass


# ==========================================
# SIMPLE TRANSCRIPTION ENDPOINT
# ==========================================

@app.post("/transcribe")
async def simple_transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    model: Optional[str] = Form(default=None),
):
    """Simple transcription endpoint."""
    return await transcribe_endpoint(
        file=audio,
        language=language,
        model=model,
        response_format="json",
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiceForge Unified ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8889, help="Port to listen on")
    parser.add_argument("--warmup", action="store_true", help="Warmup model on startup")
    parser.add_argument("--warmup-model", default=None, help="Model to warmup (default: whisper)")
    
    args = parser.parse_args()
    
    if args.warmup:
        log_info("Warming up model...")
        try:
            if args.warmup_model and is_removed_glm_gguf_model(args.warmup_model):
                log_warn("GLM-ASR GGUF backend has been removed. Warmup skipped for requested model.")
            elif args.warmup_model and is_glm_model(args.warmup_model):
                load_glm_model()
            elif args.warmup_model and is_parakeet_model(args.warmup_model):
                load_parakeet_model(args.warmup_model)
            else:
                load_whisper_model()
            log_info("Model warmed up successfully")
        except Exception as e:
            log_warn(f"Warmup failed: {e}")
    
    log_info(f"Starting server on {args.host}:{args.port}")
    log_info(f"Whisper available: {FASTER_WHISPER_AVAILABLE}")
    log_info(f"GLM-ASR available: {GLM_ASR_AVAILABLE}")
    log_info(f"Parakeet available: {PARAKEET_AVAILABLE}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=os.getenv("VF_UVICORN_LOG_LEVEL", "warning").lower(),
        access_log=_env_flag("VF_ACCESS_LOGS", "0"),
    )
