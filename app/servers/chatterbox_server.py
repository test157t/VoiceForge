# Chatterbox-Turbo TTS FastAPI Server
# https://huggingface.co/ResembleAI/chatterbox-turbo
# Copyright (c) 2025

import os
import sys
import asyncio
import glob

# Add app directory to path for imports
_APP_DIR = os.path.dirname(os.path.dirname(__file__))
_MODEL_CACHE_DIR = os.path.join(_APP_DIR, "models", "chatterbox")
os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
# Use HF_HOME to keep model cache under app/models
os.environ.setdefault("HF_HOME", _MODEL_CACHE_DIR)

# Custom fine-tuned models directory
_CUSTOM_MODELS_DIR = os.path.join(_APP_DIR, "models", "chatterbox_custom")
os.makedirs(_CUSTOM_MODELS_DIR, exist_ok=True)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Disable HF token requirement for non-gated models
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Patch huggingface_hub early to not require tokens for non-gated models
def _patch_huggingface_hub():
    """Monkey-patch huggingface_hub to not require tokens."""
    try:
        import huggingface_hub
        from huggingface_hub import hf_hub_download as original_download
        from huggingface_hub import snapshot_download as original_snapshot
        
        def patched_hf_hub_download(*args, **kwargs):
            # Remove token=True, allow None or False
            if kwargs.get('token') is True:
                kwargs['token'] = None
            return original_download(*args, **kwargs)
        
        def patched_snapshot_download(*args, **kwargs):
            # Remove token=True, allow None or False
            if kwargs.get('token') is True:
                kwargs['token'] = None
            return original_snapshot(*args, **kwargs)
        
        huggingface_hub.hf_hub_download = patched_hf_hub_download
        huggingface_hub.snapshot_download = patched_snapshot_download
        
        # Also patch the file_download module directly
        try:
            from huggingface_hub import file_download
            file_download.hf_hub_download = patched_hf_hub_download
        except:
            pass
            
    except ImportError:
        pass

_patch_huggingface_hub()

import logging
import tempfile
import uuid
from typing import Optional
from pathlib import Path
import threading
import io
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Text chunking is handled by each endpoint using util.text_utils.split_text()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy loggers
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Chatterbox-Turbo TTS Server", version="1.0.0")

# Enable CORS for UI model switching
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Worker pool configuration - each worker has its own model instance
# This allows true parallel TTS generation without model state corruption
CHATTERBOX_WORKERS = int(os.getenv("CHATTERBOX_WORKERS", "1"))


class ChatterboxWorker:
    """Single Chatterbox worker with its own model instance."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.lock = threading.Lock()
        self.model = None
        self.loaded = False
        self.sample_rate = None
        self.current_model_name = None
    
    def load_model(self, custom_model_path: str = None, custom_model_name: str = None):
        """Load this worker's model instance, optionally with custom checkpoint."""
        logger.info(f"[Worker {self.worker_id}] load_model called with path={custom_model_path}, name={custom_model_name}")
        # Check if we need to reload due to model change
        target_model = custom_model_name or "default"
        if self.loaded and self.model is not None and self.current_model_name == target_model:
            logger.info(f"[Worker {self.worker_id}] Model already loaded: {target_model}")
            return self.model
        
        with self.lock:
            # Double-check after acquiring lock
            if self.loaded and self.model is not None and self.current_model_name == target_model:
                return self.model
            
            # If model is loaded but we need a different one, unload first
            if self.model is not None and self.current_model_name != target_model:
                logger.info(f"[Worker {self.worker_id}] Unloading current model ({self.current_model_name}) for switch to {target_model}")
                del self.model
                self.model = None
                self.loaded = False
                torch.cuda.empty_cache()
            
            logger.info(f"[Worker {self.worker_id}] Loading Chatterbox-Turbo model...")
            
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            
            try:
                self.model = ChatterboxTurboTTS.from_pretrained(device=DEVICE, token=False)
            except TypeError:
                self.model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
            
            # Load custom checkpoint weights if specified
            if custom_model_path:
                self._load_custom_checkpoint(custom_model_path, custom_model_name)
            
            self.loaded = True
            self.sample_rate = self.model.sr
            self.current_model_name = target_model
            
            if custom_model_path:
                logger.info(f"[Worker {self.worker_id}] Chatterbox-Turbo loaded on {DEVICE}, sample_rate={self.model.sr}")
                logger.info(f"[Worker {self.worker_id}] Using CUSTOM model: {target_model} from {custom_model_path}")
            else:
                logger.info(f"[Worker {self.worker_id}] Chatterbox-Turbo loaded on {DEVICE}, sample_rate={self.model.sr}")
                logger.info(f"[Worker {self.worker_id}] Using DEFAULT pretrained model")
            
            return self.model
    
    def _load_custom_checkpoint(self, model_path: str, model_name: str):
        """Load custom fine-tuned checkpoint weights onto the model.
        
        Handles vocabulary size mismatches between fine-tuned checkpoints (which may have
        expanded vocabulary for multi-language support) and the base model.
        """
        from safetensors.torch import load_file
        
        # Find the checkpoint file - prefer t3_turbo_finetuned.safetensors or latest checkpoint
        checkpoint_file = None
        
        # Check for direct finetuned file
        finetuned_path = os.path.join(model_path, "t3_turbo_finetuned.safetensors")
        if os.path.exists(finetuned_path):
            checkpoint_file = finetuned_path
        else:
            # Look for checkpoint directories
            checkpoint_dirs = sorted(glob.glob(os.path.join(model_path, "checkpoint-*")), 
                                     key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
                                     reverse=True)
            if checkpoint_dirs:
                # Use latest checkpoint
                latest_checkpoint = checkpoint_dirs[0]
                safetensor_path = os.path.join(latest_checkpoint, "model.safetensors")
                if os.path.exists(safetensor_path):
                    checkpoint_file = safetensor_path
        
        if not checkpoint_file:
            logger.warning(f"[Worker {self.worker_id}] No checkpoint file found in {model_path}, using base model")
            return
        
        logger.info(f"[Worker {self.worker_id}] Loading custom checkpoint: {checkpoint_file}")
        
        try:
            # Load the safetensors checkpoint
            state_dict = load_file(checkpoint_file)
            
            # The fine-tuned weights are for the T3 model (text-to-speech transformer)
            # Load them into the t3 component
            if hasattr(self.model, 't3') and self.model.t3 is not None:
                # Filter keys that match t3 model
                t3_state_dict = {}
                for key, value in state_dict.items():
                    # Remove 'model.' prefix if present (from trainer)
                    clean_key = key.replace('model.', '') if key.startswith('model.') else key
                    t3_state_dict[clean_key] = value
                
                # Get current model state dict for shape comparison
                model_state_dict = self.model.t3.state_dict()
                
                # Handle vocabulary size mismatch for embedding and output head layers
                # Fine-tuned models may have expanded vocabulary (e.g., 52260 vs base 50276)
                vocab_layers = ['text_emb.weight', 'text_head.weight']
                
                for layer_name in vocab_layers:
                    if layer_name in t3_state_dict and layer_name in model_state_dict:
                        checkpoint_shape = t3_state_dict[layer_name].shape
                        model_shape = model_state_dict[layer_name].shape
                        
                        if checkpoint_shape != model_shape:
                            # Vocabulary size mismatch - copy only what fits
                            checkpoint_vocab_size = checkpoint_shape[0]
                            model_vocab_size = model_shape[0]
                            
                            if checkpoint_vocab_size > model_vocab_size:
                                # Checkpoint has expanded vocab - truncate to base model size
                                logger.info(f"[Worker {self.worker_id}] {layer_name}: checkpoint vocab ({checkpoint_vocab_size}) > model vocab ({model_vocab_size}), truncating")
                                t3_state_dict[layer_name] = t3_state_dict[layer_name][:model_vocab_size, :]
                            else:
                                # Model has larger vocab than checkpoint - partial copy
                                logger.info(f"[Worker {self.worker_id}] {layer_name}: checkpoint vocab ({checkpoint_vocab_size}) < model vocab ({model_vocab_size}), partial copy")
                                # Create new tensor with model shape, copy checkpoint weights
                                new_weights = model_state_dict[layer_name].clone()
                                new_weights[:checkpoint_vocab_size, :] = t3_state_dict[layer_name]
                                t3_state_dict[layer_name] = new_weights
                
                # Load with strict=False to allow partial loading
                missing, unexpected = self.model.t3.load_state_dict(t3_state_dict, strict=False)
                
                if missing:
                    logger.debug(f"[Worker {self.worker_id}] Missing keys (expected for partial load): {len(missing)}")
                if unexpected:
                    logger.debug(f"[Worker {self.worker_id}] Unexpected keys: {len(unexpected)}")
                
                logger.info(f"[Worker {self.worker_id}] Custom checkpoint '{model_name}' loaded successfully")
            else:
                logger.warning(f"[Worker {self.worker_id}] Model doesn't have t3 component, cannot load custom weights")
        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Failed to load custom checkpoint: {e}")
            logger.info(f"[Worker {self.worker_id}] Falling back to base model")
    
    def generate(self, text: str, audio_prompt_path: str, seed: int = 0, 
                 custom_model_path: str = None, custom_model_name: str = None):
        """Generate audio using this worker's model instance."""
        with self.lock:
            model = self.load_model(custom_model_path, custom_model_name)
            
            if seed > 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            with torch.inference_mode():
                return model.generate(text, audio_prompt_path=audio_prompt_path)
    
    def unload(self):
        """Unload model to free memory."""
        with self.lock:
            if self.model is not None:
                del self.model
                self.model = None
            self.loaded = False
            self.sample_rate = None


class ChatterboxWorkerPool:
    """Pool of Chatterbox workers for parallel generation."""
    
    def __init__(self, num_workers: int = 2):
        self.num_workers = num_workers
        self.workers: list = []
        self.executor: Optional[ThreadPoolExecutor] = None
        self._initialized = False
        self._worker_index = 0
        self._index_lock = threading.Lock()
    
    def initialize(self):
        """Initialize the worker pool."""
        if self._initialized:
            return
        
        logger.info(f"Initializing Chatterbox worker pool with {self.num_workers} worker(s)...")
        
        self.workers = [ChatterboxWorker(i) for i in range(self.num_workers)]
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="chatterbox_worker")
        self._initialized = True
        
        logger.info(f"Chatterbox worker pool ready with {self.num_workers} worker(s)")
    
    def get_worker(self) -> ChatterboxWorker:
        """Get next worker in round-robin fashion."""
        if not self._initialized:
            self.initialize()
        
        with self._index_lock:
            worker = self.workers[self._worker_index % self.num_workers]
            self._worker_index += 1
            return worker
    
    def shutdown(self):
        """Shutdown all workers."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        for worker in self.workers:
            worker.unload()
        
        self.workers = []
        self._initialized = False
    
    @property
    def sample_rate(self):
        """Get sample rate from first loaded worker."""
        if self.workers:
            for w in self.workers:
                if w.loaded:
                    return w.sample_rate
        return 24000  # Default


# Global worker pool
WORKER_POOL = ChatterboxWorkerPool(num_workers=CHATTERBOX_WORKERS)
logger.info(f"Chatterbox configured with {CHATTERBOX_WORKERS} workers")


def get_model():
    """Legacy function - returns a worker's model for backwards compatibility."""
    worker = WORKER_POOL.get_worker()
    custom_path = _current_custom_model.get("path")
    custom_name = _current_custom_model.get("name")
    return worker.load_model(custom_path, custom_name)


@app.get("/health")
async def health():
    """Health check endpoint."""
    vram_info = {}
    if torch.cuda.is_available():
        vram_info = {
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        }
    
    # Count loaded workers
    workers_loaded = sum(1 for w in WORKER_POOL.workers if w.loaded)
    
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "workers": {
            "configured": WORKER_POOL.num_workers,
            "loaded": workers_loaded,
            "initialized": WORKER_POOL._initialized
        },
        "sample_rate": WORKER_POOL.sample_rate,
        "vram": vram_info,
        "features": {
            "paralinguistic_tags": True,
            "voice_cloning": True,
            "supported_tags": ["[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]", "[groan]", "[yawn]", "[clear throat]"]
        }
    }


@app.post("/warmup")
async def warmup():
    """Pre-load all Chatterbox workers."""
    try:
        WORKER_POOL.initialize()
        # Load model on all workers (with custom checkpoint if set)
        custom_path = _current_custom_model.get("path")
        custom_name = _current_custom_model.get("name")
        for worker in WORKER_POOL.workers:
            worker.load_model(custom_path, custom_name)
        return {
            "status": "ok",
            "message": f"Chatterbox-Turbo loaded on {WORKER_POOL.num_workers} workers (model: {custom_name or 'default'})",
            "sample_rate": WORKER_POOL.sample_rate,
            "device": DEVICE
        }
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model():
    """Unload all Chatterbox-Turbo models to free GPU memory."""
    # Get memory before unload
    before_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    
    workers_loaded = sum(1 for w in WORKER_POOL.workers if w.loaded)
    if workers_loaded == 0:
        return {
            "success": True,
            "message": "No models loaded",
            "freed_gb": 0
        }
    
    # Unload all workers
    for worker in WORKER_POOL.workers:
        worker.unload()
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get memory after unload
    after_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    freed = before_reserved - after_reserved
    
    logger.info(f"Chatterbox-Turbo model unloaded, freed {freed:.2f}GB VRAM")
    
    return {
        "success": True,
        "message": "Model unloaded",
        "freed_gb": round(freed, 2)
    }


@app.get("/model_info")
async def get_model_info():
    """Get information about the currently loaded model."""
    # Estimated model size for Chatterbox-Turbo (350M params)
    ESTIMATED_SIZE_GB = 2.0
    
    actual_vram_gb = 0
    if torch.cuda.is_available():
        actual_vram_gb = torch.cuda.memory_allocated() / 1e9
    
    workers_loaded = sum(1 for w in WORKER_POOL.workers if w.loaded)
    
    return {
        "model_id": "chatterbox-turbo",
        "model_name": "Chatterbox-Turbo",
        "model_size": "350M parameters",
        "workers": {
            "configured": WORKER_POOL.num_workers,
            "loaded": workers_loaded
        },
        "sample_rate": WORKER_POOL.sample_rate,
        "estimated_size_gb": ESTIMATED_SIZE_GB * WORKER_POOL.num_workers,
        "actual_vram_gb": round(actual_vram_gb, 2),
        "device": DEVICE,
        "features": [
            "Zero-shot voice cloning",
            "Paralinguistic tags ([laugh], [chuckle], etc.)",
            "Low latency (optimized for voice agents)",
            "Single-step mel decoding",
            f"Parallel generation ({WORKER_POOL.num_workers} workers)"
        ]
    }


@app.get("/gpu_memory")
async def gpu_memory():
    """Get GPU memory usage information."""
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


# ============================================================================
# TEXT SPLITTING UTILITIES
# ============================================================================

import re

from util.text_utils import split_text


@app.post("/v1/tts/chunked")
async def generate_tts_chunked(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    seed: int = Form(0),
    max_tokens: int = Form(200),
    request_id: str = Form(None),  # Accept request_id from client for unified tracking
):
    """
    Chunked TTS - splits long text into sentences/chunks, generates each, and concatenates.
    
    Ideal for long-form content like scripts or articles.
    
    - text: Text to synthesize (will be split into chunks)
    - prompt_audio: Reference audio file for voice cloning (5+ seconds required, 10+ recommended)
    - seed: Random seed for reproducibility (0 = random)
    - max_tokens: Approximate max tokens per chunk (default 200)
    - request_id: Optional request ID for unified logging across services
    
    Note: Chatterbox-Turbo doesn't support exaggeration/cfg_weight controls.
    """
    # Use client-provided request_id or generate one
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    try:
        import time
        t_start = time.perf_counter()
        
        # Save uploaded audio to temp file
        temp_prompt = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_prompt.write(await prompt_audio.read())
        temp_prompt.close()
        prompt_path = temp_prompt.name
        
        # Check audio duration
        try:
            waveform, sr = ta.load(prompt_path)
            duration = waveform.shape[1] / sr
            if duration < 5.0:
                os.unlink(prompt_path)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Prompt audio is too short ({duration:.1f}s). Chatterbox requires at least 5 seconds."
                )
            logger.info(f"[{request_id}] Prompt audio: {duration:.1f}s @ {sr}Hz")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"[{request_id}] Could not verify prompt duration: {e}")
        
        # Split text into chunks for better quality on long text
        chunks = split_text(text, max_tokens=max_tokens, token_method="tiktoken")
        logger.info(f"[{request_id}] Chunked TTS: {len(chunks)} chunks from {len(text)} chars (max_tokens={max_tokens})")
        
        if not chunks:
            os.unlink(prompt_path)
            raise HTTPException(status_code=400, detail="No valid text to synthesize")
        
        # Get a dedicated worker for this request
        worker = WORKER_POOL.get_worker()
        # Load model with custom checkpoint if set
        custom_path = _current_custom_model.get("path")
        custom_name = _current_custom_model.get("name")
        worker.load_model(custom_path, custom_name)
        
        # Run generation in executor to not block the async event loop
        # This allows other requests to be received while GPU is busy
        def do_generate():
            """Blocking generation - runs in thread pool executor."""
            # Generate each chunk using worker's generate method (handles locking)
            all_audio = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{request_id}-{i+1}"
                logger.info(f"[{chunk_id}] Chunk {i+1}/{len(chunks)}: {chunk[:60]}...")
                
                t_chunk_start = time.perf_counter()
                
                # Set seed (increment for each chunk for variety while maintaining reproducibility)
                # -1 = random each time, 0 = no seeding, >0 = specific seed
                if seed == -1:
                    chunk_seed = torch.randint(0, 2**31, (1,)).item() + i
                elif seed > 0:
                    chunk_seed = seed + i
                else:
                    chunk_seed = 0
                
                # Generate using worker's generate method (handles locking and seeding)
                # Pass custom model info if set
                custom_path = _current_custom_model.get("path")
                custom_name = _current_custom_model.get("name")
                if i == 0:  # Log once at start
                    if custom_path:
                        logger.info(f"[{chunk_id}] Generating with custom model: {custom_name}")
                    else:
                        logger.info(f"[{chunk_id}] Generating with default model")
                wav = worker.generate(chunk, prompt_path, chunk_seed, custom_path, custom_name)
                
                all_audio.append(wav)
                
                t_chunk_end = time.perf_counter()
                chunk_duration = wav.shape[1] / worker.sample_rate
                chunk_time = t_chunk_end - t_chunk_start
                logger.info(f"[{chunk_id}] Done: {chunk_duration:.1f}s audio in {chunk_time:.1f}s")
            
            return all_audio
        
        # Run in worker pool executor so we don't block the event loop
        loop = asyncio.get_event_loop()
        all_audio = await loop.run_in_executor(WORKER_POOL.executor, do_generate)
        logger.info(f"[{request_id}] Generation complete")
        
        # Concatenate all audio (outside the lock - we have our data)
        combined = torch.cat(all_audio, dim=1)
        
        # Save output
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        ta.save(output_path, combined, worker.sample_rate)
        
        # Cleanup
        try:
            os.unlink(prompt_path)
        except:
            pass
        
        # Log timing
        total_duration = combined.shape[1] / worker.sample_rate
        total_time = time.perf_counter() - t_start
        rtf = total_time / total_duration if total_duration > 0 else 0
        
        logger.info(f"[{request_id}] Chunked TTS complete: {total_duration:.1f}s audio in {total_time:.1f}s (RTF: {rtf:.2f}x, chunks: {len(chunks)})")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Chunked TTS failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/stream")
async def generate_tts_stream(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    seed: int = Form(0),
    max_tokens: int = Form(200),
    request_id: str = Form(None),  # Accept request_id from client for unified tracking
):
    """
    Streaming TTS - streams audio chunks as they are generated via SSE.
    
    Each chunk is generated and sent immediately, allowing playback to start
    before the full audio is complete. Ideal for real-time/interactive use.
    
    - text: Text to synthesize (will be split into chunks and streamed)
    - prompt_audio: Reference audio file for voice cloning (5+ seconds required, 10+ recommended)
    - seed: Random seed for reproducibility (0 = random)
    - max_tokens: Approximate max tokens per chunk (default 200)
    - request_id: Optional request ID for unified logging across services
    
    Returns: SSE stream with base64-encoded WAV chunks
    """
    import base64
    import json
    import time
    
    # Use client-provided request_id or generate one
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    # Save uploaded audio to temp file first (before generator)
    temp_prompt = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_prompt.write(await prompt_audio.read())
    temp_prompt.close()
    prompt_path = temp_prompt.name
    
    # Check audio duration
    try:
        waveform, sr = ta.load(prompt_path)
        duration = waveform.shape[1] / sr
        if duration < 5.0:
            os.unlink(prompt_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Prompt audio is too short ({duration:.1f}s). Chatterbox requires at least 5 seconds."
            )
        logger.info(f"[{request_id}] Stream: Prompt audio: {duration:.1f}s @ {sr}Hz")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"[{request_id}] Could not verify prompt duration: {e}")
    
    # Split text into chunks for streaming
    from util.text_utils import split_text
    chunks = split_text(text, max_tokens=max_tokens, token_method="tiktoken")
    
    if not chunks:
        os.unlink(prompt_path)
        raise HTTPException(status_code=400, detail="No valid text to synthesize")
    
    logger.info(f"[{request_id}] Streaming TTS: {len(chunks)} chunks from {len(text)} chars")
    
    async def generate_stream():
        """Generator that yields SSE events with audio chunks."""
        t_start = time.perf_counter()
        
        # Get a dedicated worker for this request (round-robin from pool)
        worker = WORKER_POOL.get_worker()
        # Load model with custom checkpoint if set
        custom_path = _current_custom_model.get("path")
        custom_name = _current_custom_model.get("name")
        worker.load_model(custom_path, custom_name)
        sample_rate = worker.sample_rate
        
        try:
            # Send initial event with metadata
            yield f"data: {json.dumps({'type': 'start', 'chunks': len(chunks), 'sample_rate': sample_rate})}\n\n"
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{request_id}-{i+1}"
                logger.info(f"[{chunk_id}] Streaming chunk {i+1}/{len(chunks)}: {chunk[:60]}...")
                
                t_chunk_start = time.perf_counter()
                
                # Set seed (-1 = random each time, 0 = no seeding, >0 = specific seed)
                if seed == -1:
                    chunk_seed = torch.randint(0, 2**31, (1,)).item() + i
                elif seed > 0:
                    chunk_seed = seed + i
                else:
                    chunk_seed = 0
                
                # Use worker's generate method (handles locking and seeding internally)
                # Pass custom model info if set
                custom_path = _current_custom_model.get("path")
                custom_name = _current_custom_model.get("name")
                if i == 0:  # Log once at start
                    logger.info(f"[{request_id}] Model state: name={custom_name}, path={custom_path}")
                    if custom_path:
                        logger.info(f"[{request_id}] Streaming with custom model: {custom_name}")
                    else:
                        logger.info(f"[{request_id}] Streaming with default model")
                def do_generate(w=worker, text=chunk, prompt=prompt_path, seed_val=chunk_seed,
                               c_path=custom_path, c_name=custom_name):
                    """Blocking TTS generation - runs in thread pool."""
                    return w.generate(text, prompt, seed_val, c_path, c_name)
                
                # Run in worker pool's executor
                loop = asyncio.get_event_loop()
                wav = await loop.run_in_executor(WORKER_POOL.executor, do_generate)
                
                # Convert to WAV bytes
                wav_buffer = io.BytesIO()
                ta.save(wav_buffer, wav, sample_rate, format="wav")
                wav_bytes = wav_buffer.getvalue()
                wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                
                chunk_duration = wav.shape[1] / sample_rate
                chunk_time = time.perf_counter() - t_chunk_start
                
                logger.info(f"[{chunk_id}] Done: {chunk_duration:.1f}s audio in {chunk_time:.1f}s")
                
                # Send chunk event
                yield f"data: {json.dumps({'type': 'chunk', 'index': i, 'total': len(chunks), 'audio': wav_b64, 'duration': round(chunk_duration, 2), 'text': chunk[:100]})}\n\n"
            
            # Send completion event
            total_time = time.perf_counter() - t_start
            logger.info(f"[{request_id}] Stream complete in {total_time:.1f}s")
            yield f"data: {json.dumps({'type': 'complete', 'total_time': round(total_time, 2)})}\n\n"
            
        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Cleanup temp file (but NOT the shared executor)
            try:
                os.unlink(prompt_path)
            except:
                pass
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "close", "X-Accel-Buffering": "no"}
    )


# ============================================
# Custom Model Management
# ============================================

# Track current custom model
_current_custom_model = {"name": "default", "path": None}


@app.get("/v1/models")
async def list_models():
    """List available Chatterbox models (default + custom fine-tuned)."""
    models = [
        {
            "name": "default",
            "type": "default",
            "path": None,
            "description": "Default Chatterbox-Turbo model from ResembleAI"
        }
    ]
    
    # Scan custom models directory
    if os.path.exists(_CUSTOM_MODELS_DIR):
        for name in os.listdir(_CUSTOM_MODELS_DIR):
            model_path = os.path.join(_CUSTOM_MODELS_DIR, name)
            if os.path.isdir(model_path):
                # Check for fine-tuned checkpoint files
                safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
                is_turbo = any('turbo' in f.lower() for f in safetensors_files)
                
                models.append({
                    "name": name,
                    "type": "custom",
                    "path": model_path,
                    "is_turbo": is_turbo,
                    "checkpoint_files": safetensors_files,
                    "valid": len(safetensors_files) > 0
                })
    
    return {
        "models": models,
        "current": _current_custom_model["name"]
    }


@app.post("/v1/models/switch")
async def switch_model(model_name: str = "default"):
    """
    Switch to a different Chatterbox model.
    
    Note: Custom model loading requires unloading current model and 
    loading with modified checkpoint. The implementation depends on 
    the chatterbox-finetuning toolkit's model format.
    """
    global _current_custom_model
    
    logger.info(f"=== MODEL SWITCH REQUEST: '{model_name}' ===")
    
    if model_name == "default":
        # Switch back to default - need to reload
        if _current_custom_model["name"] != "default":
            logger.info("Switching back to default model - unloading workers")
            for worker in WORKER_POOL.workers:
                worker.unload()
        
        _current_custom_model = {"name": "default", "path": None}
        return {"success": True, "model": "default", "message": "Switched to default model. Workers will load default model on next generation."}
    
    # Check custom models
    model_path = os.path.join(_CUSTOM_MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Check for checkpoint files
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    if not safetensors_files:
        raise HTTPException(status_code=400, detail=f"No checkpoint files found in '{model_name}'")
    
    # Unload current model if switching to different one
    if _current_custom_model["name"] != model_name:
        logger.info(f"Switching to custom model '{model_name}' - unloading workers")
        for worker in WORKER_POOL.workers:
            worker.unload()
    
    _current_custom_model = {"name": model_name, "path": model_path}
    
    logger.info(f"Custom model set: {model_name}")
    logger.info(f"Checkpoint files: {safetensors_files}")
    
    return {
        "success": True,
        "model": model_name,
        "path": model_path,
        "checkpoint_files": safetensors_files,
        "message": f"Switched to model '{model_name}'. Note: Full custom checkpoint loading requires integration with chatterbox-finetuning."
    }


@app.get("/v1/models/current")
async def current_model():
    """Get the currently configured model."""
    return {
        "name": _current_custom_model["name"],
        "path": _current_custom_model["path"],
        "workers_loaded": WORKER_POOL._initialized,
        "workers_count": len(WORKER_POOL.workers) if WORKER_POOL.workers else 0
    }


if __name__ == "__main__":
    import argparse

    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8893, help="Port to bind to")
    parser.add_argument("--warmup", action="store_true", help="Load model on startup")
    args = parser.parse_args()
    
    logger.info(f"Starting Chatterbox-Turbo TTS server on {args.host}:{args.port}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Total VRAM: {total_vram:.1f}GB")
    
    if args.warmup:
        logger.info("Pre-loading model...")
        get_model()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=3600,
        log_level=os.getenv("VF_UVICORN_LOG_LEVEL", "warning").lower(),
        access_log=_env_flag("VF_ACCESS_LOGS", "0"),
    )

