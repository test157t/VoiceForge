# RVC FastAPI Server - Voice Conversion with Worker Pool

import os
import sys

# ============================================
# CONFIGURATION - Set up paths first
# ============================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # app/servers
APP_DIR = os.path.dirname(SCRIPT_DIR)  # app
ROOT_DIR = os.path.dirname(APP_DIR)  # VoiceForge root
UTIL_DIR = os.path.join(APP_DIR, "util")  # app/util
CONFIG_DIR = os.path.join(APP_DIR, "config")  # app/config
MODELS_DIR_PATH = os.path.join(APP_DIR, "models")  # app/models

# Add paths for imports
sys.path.insert(0, APP_DIR)
sys.path.insert(0, UTIL_DIR)
sys.path.insert(0, CONFIG_DIR)
sys.path.insert(0, MODELS_DIR_PATH)

# ============================================
# LOGGING - Use centralized logging
# ============================================

from logging_utils import (
    setup_rvc_logging,
    suppress_library_loggers,
    create_server_logger,
    get_logger
)

# Configure RVC environment
setup_rvc_logging()

# Create server-specific logging
log_info, log_warn, log_error = create_server_logger("RVC")
logger = get_logger("rvc_server")

import warnings
warnings.filterwarnings("ignore")

# Suppress library loggers
suppress_library_loggers()

# ============================================
# IMPORTS
# ============================================

import logging
import tempfile
import uuid
import base64
import io
import gc
import threading
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
sys.path.insert(0, os.path.join(APP_DIR, "assets", "custom_dependencies"))

# Import config for RVC parameters
from config import get_config

# RVC model directories
MODEL_DIR = os.path.join(APP_DIR, "models", "rvc_user")
RVC_MAIN_DIR = os.path.join(APP_DIR, "models", "rvc_main")  # hubert_base.pt, rmvpe.pt
HUBERT_PATH = os.path.join(RVC_MAIN_DIR, "hubert_base.pt")
RMVPE_PATH = os.path.join(RVC_MAIN_DIR, "rmvpe.pt")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable TF32 for Ampere GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info("TF32 and cuDNN benchmark enabled")

# Chunk duration for long audio processing
CHUNK_DURATION_SECONDS = 60

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(title="RVC Server", version="1.0.0")

# ============================================
# RVC WORKER CLASS
# ============================================

class RVCWorker:
    """Single RVC worker with its own model instance."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.lock = threading.Lock()
        self.loader = None
        self.loaded = False
        self.current_model = None
        self._loaded_models = {}  # Cache ALL loaded models: {params_key: True}
        self.warmed_up_model = None
    
    def load_base(self, only_cpu: bool = False):
        """Load base RVC components (HuBERT, RMVPE)."""
        if self.loaded and self.loader:
            return self.loader
        
        with self.lock:
            if self.loaded and self.loader:
                return self.loader
            
            logger.info(f"[Worker {self.worker_id}] Loading RVC base components...")
            
            from infer_rvc_python import BaseLoader
            self.loader = BaseLoader(
                only_cpu=only_cpu,
                hubert_path=HUBERT_PATH,
                rmvpe_path=RMVPE_PATH
            )
            self.loaded = True
            
            logger.info(f"[Worker {self.worker_id}] RVC base loaded on {DEVICE}")
            
        return self.loader
    
    def _build_params_key(self, model_name: str, rvc_params: dict) -> tuple:
        """Build a cache key from model name and params. NO DEFAULTS - params must be complete."""
        return (
            model_name,
            rvc_params.get("pitch_algo"),
            rvc_params.get("pitch_lvl"),
            rvc_params.get("index_influence"),
            rvc_params.get("respiration_median_filtering"),
            rvc_params.get("envelope_ratio"),
            rvc_params.get("consonant_breath_protection")
        )
    
    def load_model(self, model_name: str, model_path: str, index_path: str, rvc_params: dict):
        """Load a specific RVC model. Skips if this model+params combo was EVER loaded before."""
        if not self.loaded:
            self.load_base()
        
        # Build params key for caching - NO DEFAULTS
        params_key = self._build_params_key(model_name, rvc_params)
        
        # Skip if this model+params combo was EVER loaded before
        if params_key in self._loaded_models:
            logger.debug(f"[Worker {self.worker_id}] Model '{model_name}' already in cache, skipping load")
            self.current_model = model_name
            return
        
        with self.lock:
            # Double-check inside lock
            if params_key in self._loaded_models:
                logger.debug(f"[Worker {self.worker_id}] Model '{model_name}' already in cache (lock check)")
                self.current_model = model_name
                return
            
            logger.info(f"[Worker {self.worker_id}] Loading model '{model_name}' (first time with these params)")
            
            self.loader.apply_conf(
                tag=model_name,
                file_model=model_path,
                file_index=index_path,
                pitch_algo=rvc_params["pitch_algo"],
                pitch_lvl=rvc_params["pitch_lvl"],
                index_influence=rvc_params["index_influence"],
                respiration_median_filtering=rvc_params["respiration_median_filtering"],
                envelope_ratio=rvc_params["envelope_ratio"],
                consonant_breath_protection=rvc_params["consonant_breath_protection"]
            )
            self.current_model = model_name
            self._loaded_models[params_key] = True
            
        logger.info(f"[Worker {self.worker_id}] Model '{model_name}' loaded and cached")
    
    def convert(self, audio_data: np.ndarray, sample_rate: int, model_name: str) -> Tuple[np.ndarray, int]:
        """Convert audio using RVC."""
        if not self.loaded:
            raise RuntimeError("Worker not loaded")
        
        t_start = time.perf_counter()
        
        # Ensure audio is float32 and contiguous
        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
        
        with torch.inference_mode():
            result_audio, out_sr = self.loader.generate_from_cache(
                audio_data=(audio_data, sample_rate),
                tag=model_name
            )
        
        t_elapsed = time.perf_counter() - t_start
        audio_dur = len(audio_data) / sample_rate
        rtf = t_elapsed / audio_dur if audio_dur > 0 else 0
        logger.info(f"[Worker {self.worker_id}] Converted: {audio_dur:.1f}s audio in {t_elapsed:.1f}s (RTF: {rtf:.2f}x)")
        
        return result_audio, out_sr
    
    def warmup(self, model_name: str, model_path: str, index_path: str, rvc_params: dict):
        """Warm up the model with a dummy inference."""
        if not self.loaded:
            self.load_base()
        
        self.load_model(model_name, model_path, index_path, rvc_params)
        
        # Run a short dummy inference to warm up
        dummy_sr = 16000
        dummy = np.zeros(int(0.5 * dummy_sr), dtype=np.float32)
        
        with torch.inference_mode():
            self.loader.generate_from_cache(audio_data=(dummy, dummy_sr), tag=model_name)
        
        self.warmed_up_model = model_name
        logger.info(f"[Worker {self.worker_id}] Warmed up with model '{model_name}'")
    
    def is_warmed_up(self, model_name: str) -> bool:
        """Check if a specific model is already warmed up."""
        return self.warmed_up_model == model_name
    
    def unload(self):
        """Unload models to free VRAM."""
        with self.lock:
            if self.loader:
                try:
                    self.loader.unload_models()
                except:
                    pass
                del self.loader
                self.loader = None
            self.loaded = False
            self.current_model = None
            self._current_params_key = None
            self.warmed_up_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[Worker {self.worker_id}] Models unloaded")


class RVCWorkerPool:
    """Pool of RVC workers for parallel generation."""
    
    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self.workers: List[RVCWorker] = []
        self.executor: Optional[ThreadPoolExecutor] = None
        self._initialized = False
    
    def initialize(self, num_workers: int = None):
        """Initialize the worker pool."""
        if num_workers is not None:
            self.num_workers = num_workers
        
        self.shutdown()
        
        logger.info(f"Initializing RVC worker pool with {self.num_workers} worker(s)...")
        
        self.workers = [RVCWorker(i) for i in range(self.num_workers)]
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._initialized = True
        
        logger.info(f"RVC worker pool ready with {self.num_workers} worker(s)")
    
    def get_worker(self, index: int = 0) -> RVCWorker:
        """Get a specific worker."""
        if not self._initialized:
            self.initialize()
        return self.workers[index % self.num_workers]
    
    def shutdown(self):
        """Shutdown all workers."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        for worker in self.workers:
            worker.unload()
        
        self.workers = []
        self._initialized = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global worker pool - default to 2 workers for better parallelism
_saved_num_workers = 2  # Default to 2 workers with modern GPUs
try:
    import json
    config_path = os.path.join(APP_DIR, "config", "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            saved_config = json.load(f)
            if "rvc_workers" in saved_config:
                _saved_num_workers = int(saved_config["rvc_workers"])
                logger.info(f"Loaded rvc_workers={_saved_num_workers} from config")
except Exception as e:
    logger.warning(f"Could not load saved config: {e}")

WORKER_POOL = RVCWorkerPool(num_workers=int(os.getenv("RVC_WORKERS", str(_saved_num_workers))))
logger.info(f"RVC worker pool configured with {WORKER_POOL.num_workers} workers")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def resolve_model_path(model_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve paths for RVC model files."""
    if not model_name or model_name in ("(no models found)", "(no models folder)"):
        return None, None
    
    model_path = os.path.join(MODEL_DIR, model_name, "model.pth")
    index_path = os.path.join(MODEL_DIR, model_name, "model.index")
    
    if os.path.exists(model_path) and os.path.exists(index_path):
        return model_path, index_path
    
    return None, None


def list_available_models() -> List[str]:
    """List all available RVC models."""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models
    
    for name in os.listdir(MODEL_DIR):
        model_dir = os.path.join(MODEL_DIR, name)
        if os.path.isdir(model_dir):
            model_path = os.path.join(model_dir, "model.pth")
            index_path = os.path.join(model_dir, "model.index")
            if os.path.exists(model_path) and os.path.exists(index_path):
                models.append(name)
    
    return sorted(models)


def read_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Read audio file and return (mono_data, sample_rate)."""
    data, sr = sf.read(file_path, dtype='float32')
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data, sr


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    workers_loaded = sum(1 for w in WORKER_POOL.workers if w.loaded) if WORKER_POOL._initialized else 0
    
    vram_info = {}
    if torch.cuda.is_available():
        vram_info = {
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        }
    
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "workers": {
            "configured": WORKER_POOL.num_workers,
            "loaded": workers_loaded,
            "initialized": WORKER_POOL._initialized
        },
        "vram": vram_info,
        "models_available": len(list_available_models())
    }


@app.get("/models")
async def get_models():
    """List available RVC models."""
    models = list_available_models()
    return {
        "models": models,
        "count": len(models),
        "model_dir": MODEL_DIR
    }


@app.get("/workers")
async def get_workers():
    """Get worker pool status."""
    workers_status = []
    if WORKER_POOL._initialized:
        for w in WORKER_POOL.workers:
            workers_status.append({
                "id": w.worker_id,
                "loaded": w.loaded,
                "current_model": w.current_model,
                "warmed_up_model": w.warmed_up_model
            })
    
    return {
        "num_workers": WORKER_POOL.num_workers,
        "initialized": WORKER_POOL._initialized,
        "workers": workers_status
    }


@app.post("/workers")
async def set_workers(num_workers: int = Form(...)):
    """Set number of workers."""
    if num_workers < 1:
        raise HTTPException(status_code=400, detail="num_workers must be at least 1")
    if num_workers > 4:
        raise HTTPException(status_code=400, detail="num_workers cannot exceed 4 (VRAM limit)")
    
    try:
        WORKER_POOL.num_workers = num_workers
        WORKER_POOL.initialize(num_workers)
        return {
            "status": "ok",
            "message": f"Worker pool resized to {num_workers} workers",
            "num_workers": WORKER_POOL.num_workers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RVCRequest(BaseModel):
    """RVC conversion request body (for JSON API). All params default to config.json values."""
    audio_base64: Optional[str] = None
    model_name: str
    pitch_algo: Optional[str] = None  # None = read from config
    pitch_lvl: Optional[int] = None
    index_influence: Optional[float] = None
    respiration_median_filtering: Optional[int] = None
    envelope_ratio: Optional[float] = None
    consonant_breath_protection: Optional[float] = None


@app.post("/v1/rvc")
async def convert_rvc(
    audio: UploadFile = File(...),
    model_name: str = Form(...),
    pitch_algo: str = Form(None),  # None = read from config
    pitch_lvl: int = Form(None),
    index_influence: float = Form(None),
    respiration_median_filtering: int = Form(None),
    envelope_ratio: float = Form(None),
    consonant_breath_protection: float = Form(None),
    request_id: str = Form(None),
):
    """
    Convert audio using RVC voice conversion.
    
    All RVC params default to config.json values if not provided.
    """
    # Use client-provided request_id or generate one
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    # Fill missing params from config - NO HARDCODED DEFAULTS
    config = get_config()
    if pitch_algo is None:
        pitch_algo = config["pitch_algo"]
    if pitch_lvl is None:
        pitch_lvl = config.get("pitch_lvl", config.get("pitch_level"))
    if index_influence is None:
        index_influence = config["index_influence"]
    if respiration_median_filtering is None:
        respiration_median_filtering = config["respiration_median_filtering"]
    if envelope_ratio is None:
        envelope_ratio = config["envelope_ratio"]
    if consonant_breath_protection is None:
        consonant_breath_protection = config["consonant_breath_protection"]
    
    logger.info(f"[{request_id}] RVC request: model={model_name}, algo={pitch_algo}")
    
    t_start = time.perf_counter()
    
    try:
        # Validate model
        model_path, index_path = resolve_model_path(model_name)
        if model_path is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Save uploaded audio to temp file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_input.write(await audio.read())
        temp_input.close()
        
        t_load = time.perf_counter()
        
        # Initialize worker if needed
        if not WORKER_POOL._initialized:
            WORKER_POOL.initialize()
        
        worker = WORKER_POOL.get_worker(0)
        if not worker.loaded:
            worker.load_base()
        
        # Prepare RVC params
        rvc_params = {
            "pitch_algo": pitch_algo,
            "pitch_lvl": pitch_lvl,
            "index_influence": index_influence,
            "respiration_median_filtering": respiration_median_filtering,
            "envelope_ratio": envelope_ratio,
            "consonant_breath_protection": consonant_breath_protection
        }
        
        # Load model
        worker.load_model(model_name, model_path, index_path, rvc_params)
        
        t_model = time.perf_counter()
        
        # Read audio
        audio_data, sample_rate = read_audio(temp_input.name)
        logger.info(f"[{request_id}] Input: {len(audio_data)/sample_rate:.1f}s @ {sample_rate}Hz")
        
        t_read = time.perf_counter()
        
        # Convert
        result_audio, out_sr = worker.convert(audio_data, sample_rate, model_name)
        
        t_convert = time.perf_counter()
        
        # Save output (24-bit for quality preservation)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(output_path, result_audio, out_sr, subtype="PCM_24")
        
        t_save = time.perf_counter()
        
        # Cleanup input
        try:
            os.unlink(temp_input.name)
        except:
            pass
        
        # Log timing
        input_duration = len(audio_data) / sample_rate
        output_duration = len(result_audio) / out_sr
        total_time = t_save - t_start
        convert_time = t_convert - t_read
        rtf = convert_time / input_duration if input_duration > 0 else 0
        
        logger.info(f"[{request_id}] Timing: load={t_model-t_load:.1f}s, read={t_read-t_model:.1f}s, convert={convert_time:.1f}s, save={t_save-t_convert:.1f}s")
        logger.info(f"[{request_id}] Total: {total_time:.1f}s for {input_duration:.1f}s input -> {output_duration:.1f}s output (RTF: {rtf:.2f}x)")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="rvc_output.wav",
            headers={
                "X-Request-ID": request_id,
                "X-Processing-Time": str(round(total_time, 2)),
                "X-RTF": str(round(rtf, 2))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] RVC failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rvc/chunked")
async def convert_rvc_chunked(
    audio: UploadFile = File(...),
    model_name: str = Form(...),
    pitch_algo: str = Form(None),  # None = read from config
    pitch_lvl: int = Form(None),
    index_influence: float = Form(None),
    respiration_median_filtering: int = Form(None),
    envelope_ratio: float = Form(None),
    consonant_breath_protection: float = Form(None),
    chunk_duration: int = Form(60),
    request_id: str = Form(None),
):
    """
    Chunked RVC conversion for long audio files.
    
    All RVC params default to config.json values if not provided.
    """
    # Use client-provided request_id or generate one
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    # Fill missing params from config - NO HARDCODED DEFAULTS
    config = get_config()
    if pitch_algo is None:
        pitch_algo = config["pitch_algo"]
    if pitch_lvl is None:
        pitch_lvl = config.get("pitch_lvl", config.get("pitch_level"))
    if index_influence is None:
        index_influence = config["index_influence"]
    if respiration_median_filtering is None:
        respiration_median_filtering = config["respiration_median_filtering"]
    if envelope_ratio is None:
        envelope_ratio = config["envelope_ratio"]
    if consonant_breath_protection is None:
        consonant_breath_protection = config["consonant_breath_protection"]
    
    logger.info(f"[{request_id}] Chunked RVC request: model={model_name}, algo={pitch_algo}")
    
    t_start = time.perf_counter()
    
    try:
        # Validate model
        model_path, index_path = resolve_model_path(model_name)
        if model_path is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Save uploaded audio
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_input.write(await audio.read())
        temp_input.close()
        
        # Initialize worker
        if not WORKER_POOL._initialized:
            WORKER_POOL.initialize()
        
        worker = WORKER_POOL.get_worker(0)
        if not worker.loaded:
            worker.load_base()
        
        rvc_params = {
            "pitch_algo": pitch_algo,
            "pitch_lvl": pitch_lvl,
            "index_influence": index_influence,
            "respiration_median_filtering": respiration_median_filtering,
            "envelope_ratio": envelope_ratio,
            "consonant_breath_protection": consonant_breath_protection
        }
        
        worker.load_model(model_name, model_path, index_path, rvc_params)
        
        # Warmup with dummy only if not already warmed up for this model
        if not worker.is_warmed_up(model_name):
            logger.info(f"[{request_id}] Warming up model...")
            dummy_sr = 16000
            dummy = np.zeros(int(0.5 * dummy_sr), dtype=np.float32)
            with torch.inference_mode():
                worker.loader.generate_from_cache(audio_data=(dummy, dummy_sr), tag=model_name)
            worker.warmed_up_model = model_name
        
        # Read audio
        audio_data, sample_rate = read_audio(temp_input.name)
        total_duration = len(audio_data) / sample_rate
        logger.info(f"[{request_id}] Input: {total_duration:.1f}s @ {sample_rate}Hz")
        
        # Split into chunks
        chunk_size = int(chunk_duration * sample_rate)
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        logger.info(f"[{request_id}] Split into {len(chunks)} chunks")
        
        # Process each chunk
        t_gen_start = time.perf_counter()
        results = []
        out_sr = None
        
        for i, chunk in enumerate(chunks):
            t_chunk_start = time.perf_counter()
            logger.info(f"[{request_id}] Processing chunk {i+1}/{len(chunks)}...")
            
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            
            with torch.inference_mode():
                result_audio, out_sr = worker.loader.generate_from_cache(
                    audio_data=(chunk, sample_rate),
                    tag=model_name
                )
            
            results.append(result_audio)
            
            chunk_dur = len(chunk) / sample_rate
            t_chunk = time.perf_counter() - t_chunk_start
            logger.info(f"[{request_id}] Chunk {i+1}: {chunk_dur:.1f}s -> {len(result_audio)/out_sr:.1f}s in {t_chunk:.1f}s")
        
        t_gen = time.perf_counter() - t_gen_start
        
        # Concatenate results
        concatenated = np.concatenate(results) if results else np.zeros(1, dtype=np.float32)
        if out_sr is None:
            out_sr = sample_rate
        
        # Save output (24-bit for quality preservation)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(output_path, concatenated, out_sr, subtype="PCM_24")
        
        t_save = time.perf_counter()
        
        # Cleanup
        try:
            os.unlink(temp_input.name)
        except:
            pass
        
        total_time = t_save - t_start
        output_duration = len(concatenated) / out_sr
        rtf = t_gen / total_duration if total_duration > 0 else 0
        
        logger.info(f"[{request_id}] Total: {total_time:.1f}s for {total_duration:.1f}s input -> {output_duration:.1f}s output")
        logger.info(f"[{request_id}] Generation RTF: {rtf:.2f}x ({len(chunks)} chunks)")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="rvc_output.wav",
            headers={
                "X-Request-ID": request_id,
                "X-Processing-Time": str(round(total_time, 2)),
                "X-RTF": str(round(rtf, 2)),
                "X-Chunks": str(len(chunks))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Chunked RVC failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rvc/stream")
async def convert_rvc_stream(
    audio: UploadFile = File(...),
    model_name: str = Form(...),
    pitch_algo: str = Form(None),
    pitch_lvl: int = Form(None),
    index_influence: float = Form(None),
    respiration_median_filtering: int = Form(None),
    envelope_ratio: float = Form(None),
    consonant_breath_protection: float = Form(None),
    request_id: str = Form(None),
):
    """
    Streaming RVC conversion - processes audio and streams result via SSE.
    
    Audio is processed as a single unit (no splitting) since VoiceForge already
    sends pre-chunked audio from the TTS pipeline.
    
    All RVC params default to config.json values if not provided.
    
    Returns: SSE stream with base64-encoded WAV audio
    """
    import base64
    import json
    import io
    
    # Use client-provided request_id or generate one
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    # Fill missing params from config
    config = get_config()
    if pitch_algo is None:
        pitch_algo = config["pitch_algo"]
    if pitch_lvl is None:
        pitch_lvl = config.get("pitch_lvl", config.get("pitch_level"))
    if index_influence is None:
        index_influence = config["index_influence"]
    if respiration_median_filtering is None:
        respiration_median_filtering = config["respiration_median_filtering"]
    if envelope_ratio is None:
        envelope_ratio = config["envelope_ratio"]
    if consonant_breath_protection is None:
        consonant_breath_protection = config["consonant_breath_protection"]
    
    logger.info(f"[{request_id}] Streaming RVC request: model={model_name}, algo={pitch_algo}")
    
    # Save uploaded audio to temp file first (before generator)
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_input.write(await audio.read())
    temp_input.close()
    input_path = temp_input.name
    
    async def generate_stream():
        """Generator that yields SSE events with converted audio."""
        t_start = time.perf_counter()
        
        try:
            # Validate model
            model_path, index_path = resolve_model_path(model_name)
            if model_path is None:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Model not found: {model_name}'})}\n\n"
                return
            
            # Initialize worker
            if not WORKER_POOL._initialized:
                WORKER_POOL.initialize()
            
            worker = WORKER_POOL.get_worker(0)
            if not worker.loaded:
                worker.load_base()
            
            rvc_params = {
                "pitch_algo": pitch_algo,
                "pitch_lvl": pitch_lvl,
                "index_influence": index_influence,
                "respiration_median_filtering": respiration_median_filtering,
                "envelope_ratio": envelope_ratio,
                "consonant_breath_protection": consonant_breath_protection
            }
            
            worker.load_model(model_name, model_path, index_path, rvc_params)
            
            # Warmup if needed
            if not worker.is_warmed_up(model_name):
                logger.info(f"[{request_id}] Warming up model...")
                dummy_sr = 16000
                dummy = np.zeros(int(0.5 * dummy_sr), dtype=np.float32)
                with torch.inference_mode():
                    worker.loader.generate_from_cache(audio_data=(dummy, dummy_sr), tag=model_name)
                worker.warmed_up_model = model_name
            
            # Read audio - process as single unit (already chunked by TTS pipeline)
            audio_data, sample_rate = read_audio(input_path)
            input_duration = len(audio_data) / sample_rate
            logger.info(f"[{request_id}] Input: {input_duration:.1f}s @ {sample_rate}Hz")
            
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'sample_rate': sample_rate, 'duration': round(input_duration, 2)})}\n\n"
            
            # Process audio directly (no chunking - audio is already pre-chunked by VoiceForge)
            t_convert_start = time.perf_counter()
            audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
                
            with torch.inference_mode():
                    result_audio, out_sr = worker.loader.generate_from_cache(
                    audio_data=(audio_data, sample_rate),
                        tag=model_name
                    )
                
            t_convert = time.perf_counter() - t_convert_start
            result_duration = len(result_audio) / out_sr
            
                # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, result_audio, out_sr, format='WAV', subtype='PCM_24')
            wav_bytes = wav_buffer.getvalue()
            wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                
            logger.info(f"[{request_id}] Converted: {input_duration:.1f}s -> {result_duration:.1f}s in {t_convert:.1f}s")
                
            # Send audio chunk
            yield f"data: {json.dumps({'type': 'chunk', 'index': 0, 'total': 1, 'audio': wav_b64, 'duration': round(result_duration, 2), 'processing_time': round(t_convert, 2)})}\n\n"
            
            # Send completion event
            total_time = time.perf_counter() - t_start
            logger.info(f"[{request_id}] Complete in {total_time:.1f}s")
            yield f"data: {json.dumps({'type': 'complete', 'total_time': round(total_time, 2)})}\n\n"
            
        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Cleanup
            try:
                os.unlink(input_path)
            except:
                pass
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "close", "X-Accel-Buffering": "no", "X-Request-ID": request_id}
    )


@app.post("/load_model")
async def load_model(
    model_name: str = Form(...)
):
    """Pre-load an RVC model to reduce first conversion latency.
    
    Uses RVC parameters ONLY from config.json - no hardcoded defaults.
    """
    request_id = str(uuid.uuid4())[:8]
    
    try:
        model_path, index_path = resolve_model_path(model_name)
        if model_path is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        if not WORKER_POOL._initialized:
            WORKER_POOL.initialize()
        
        worker = WORKER_POOL.get_worker(0)
        
        # Load RVC params ONLY from config.json - NO HARDCODED DEFAULTS
        config = get_config()
        rvc_params = {
            "pitch_algo": config["pitch_algo"],
            "pitch_lvl": config.get("pitch_lvl", config.get("pitch_level")),
            "index_influence": config["index_influence"],
            "respiration_median_filtering": config["respiration_median_filtering"],
            "envelope_ratio": config["envelope_ratio"],
            "consonant_breath_protection": config["consonant_breath_protection"]
        }
        
        logger.info(f"[{request_id}] Preloading model '{model_name}' with config: {rvc_params}")
        
        worker.warmup(model_name, model_path, index_path, rvc_params)
        
        return {
            "status": "ok",
            "message": f"Model '{model_name}' loaded and cached",
            "model_name": model_name,
            "params": rvc_params
        }
        
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing config key: {e}. Check your config.json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Load model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload():
    """Unload all RVC models to free GPU memory."""
    before_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    
    workers_loaded = sum(1 for w in WORKER_POOL.workers if w.loaded) if WORKER_POOL._initialized else 0
    
    if workers_loaded == 0:
        return {
            "success": True,
            "message": "No models loaded",
            "freed_gb": 0
        }
    
    WORKER_POOL.shutdown()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    after_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    freed = before_reserved - after_reserved
    
    logger.info(f"RVC models unloaded, freed {freed:.2f}GB VRAM")
    
    return {
        "success": True,
        "message": f"Unloaded {workers_loaded} worker(s)",
        "freed_gb": round(freed, 2)
    }


@app.get("/model_info")
async def get_model_info():
    """Get information about currently loaded models."""
    ESTIMATED_SIZE_PER_WORKER_GB = 2.5  # RVC uses less VRAM than TTS
    
    workers_loaded = sum(1 for w in WORKER_POOL.workers if w.loaded) if WORKER_POOL._initialized else 0
    current_models = []
    
    if WORKER_POOL._initialized:
        for w in WORKER_POOL.workers:
            if w.current_model:
                current_models.append(w.current_model)
    
    actual_vram_gb = 0
    if torch.cuda.is_available():
        actual_vram_gb = torch.cuda.memory_allocated() / 1e9
    
    return {
        "model_id": "rvc",
        "model_name": "RVC Voice Conversion",
        "loaded": workers_loaded > 0,
        "workers_loaded": workers_loaded,
        "current_models": list(set(current_models)),
        "estimated_size_gb": ESTIMATED_SIZE_PER_WORKER_GB * max(workers_loaded, 1),
        "actual_vram_gb": round(actual_vram_gb, 2),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.get("/gpu_memory")
async def gpu_memory():
    """Get GPU memory usage."""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = props.total_memory / 1e9
    free = total - reserved
    
    return {
        "available": True,
        "device": torch.cuda.get_device_name(),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "usage_percent": round((reserved / total) * 100, 1)
    }


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse

    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8891, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--warmup", type=str, default=None, help="Model name to warmup on startup")
    args = parser.parse_args()
    
    logger.info(f"Starting RVC server on {args.host}:{args.port}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Total VRAM: {total_vram:.1f}GB")
    
    logger.info(f"Model directory: {MODEL_DIR}")
    models = list_available_models()
    logger.info(f"Available models: {models}")
    
    # Initialize worker pool
    WORKER_POOL.initialize(args.workers)
    
    if args.warmup:
        logger.info(f"Warming up with model: {args.warmup}")
        model_path, index_path = resolve_model_path(args.warmup)
        if model_path:
            worker = WORKER_POOL.get_worker(0)
            # Use config values ONLY - NO HARDCODED DEFAULTS
            config = get_config()
            try:
                warmup_params = {
                    "pitch_algo": config["pitch_algo"],
                    "pitch_lvl": config.get("pitch_lvl", config.get("pitch_level")),
                    "index_influence": config["index_influence"],
                    "respiration_median_filtering": config["respiration_median_filtering"],
                    "envelope_ratio": config["envelope_ratio"],
                    "consonant_breath_protection": config["consonant_breath_protection"]
                }
                logger.info(f"Warmup params from config: {warmup_params}")
                worker.warmup(args.warmup, model_path, index_path, warmup_params)
            except KeyError as e:
                logger.error(f"Missing config key for warmup: {e}. Check your config.json")
        else:
            logger.warning(f"Warmup model '{args.warmup}' not found")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=3600,
        log_level=os.getenv("VF_UVICORN_LOG_LEVEL", "warning").lower(),
        access_log=_env_flag("VF_ACCESS_LOGS", "0"),
    )
