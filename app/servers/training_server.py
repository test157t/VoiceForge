# TTS Training Server for VoiceForge
# Supports Chatterbox fine-tuning
# Copyright (c) 2026

import os
import sys
import asyncio
import json
import uuid
import subprocess
import threading
import time
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add app directory to path for imports
_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceForge Training Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
TRAINING_DIR = os.path.join(_APP_DIR, "training")
DATASETS_DIR = os.path.join(_APP_DIR, "datasets")
MODELS_DIR = os.path.join(_APP_DIR, "models")
CHATTERBOX_CUSTOM_DIR = os.path.join(MODELS_DIR, "chatterbox_custom")

# Ensure directories exist
for d in [TRAINING_DIR, DATASETS_DIR, CHATTERBOX_CUSTOM_DIR]:
    os.makedirs(d, exist_ok=True)


def find_dataset_path(name: str) -> str:
    """Find a dataset by name, checking multiple possible locations.
    Returns the path if found, raises HTTPException if not found."""
    # Check locations in order of priority
    possible_paths = [
        os.path.join(DATASETS_DIR, name),  # Direct in datasets/
        os.path.join(DATASETS_DIR, "chatterbox", name),  # Legacy chatterbox location
        os.path.join(TRAINING_DIR, "chatterbox-finetuning", name),  # Training dir
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    return None

# Conda paths - find the base conda installation
CONDA_BASE = None

# First try common locations
for path in [
    os.path.join(os.environ.get("USERPROFILE", ""), "miniconda3"),
    os.path.join(os.environ.get("USERPROFILE", ""), "anaconda3"),
    "C:\\ProgramData\\miniconda3",
    "C:\\ProgramData\\anaconda3"
]:
    if os.path.exists(os.path.join(path, "Scripts", "conda.exe")):
        CONDA_BASE = path
        break

# Fallback to CONDA_PREFIX if common locations don't work
if not CONDA_BASE:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        # Extract base from envs path
        if "\\envs\\" in conda_prefix:
            CONDA_BASE = conda_prefix.split("\\envs\\")[0]
        elif "/envs/" in conda_prefix:
            CONDA_BASE = conda_prefix.split("/envs/")[0]
        else:
            # Might be base env itself
            if os.path.exists(os.path.join(conda_prefix, "Scripts", "conda.exe")):
                CONDA_BASE = conda_prefix


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    job_id: str
    backend: str  # "chatterbox"
    status: JobStatus
    config: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    output_model_path: Optional[str] = None


# Job storage (in-memory for now)
jobs: Dict[str, TrainingJob] = {}
job_lock = threading.Lock()

# WebSocket connections for progress streaming
ws_connections: List[WebSocket] = []


# ============================================
# Request Models
# ============================================

class ChatterboxTrainRequest(BaseModel):
    """Chatterbox fine-tuning configuration."""
    model_name: str = Field(..., description="Name for the output model")
    dataset_path: str = Field(..., description="Path to dataset folder (LJSpeech format)")
    is_turbo: bool = Field(default=True, description="Use Turbo mode (GPT-2) vs Standard (Llama)")
    epochs: int = Field(default=150, ge=1, le=1000)
    learning_rate: float = Field(default=5e-5, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=4, ge=1, le=32)
    gradient_accumulation: int = Field(default=8, ge=1, le=64)
    speaker_reference: Optional[str] = Field(default=None, description="Path to speaker reference audio")
    preprocess: bool = Field(default=True, description="Run preprocessing before training")


class DatasetPrepareRequest(BaseModel):
    """Dataset preparation request."""
    name: str = Field(..., description="Dataset name")
    backend: str = Field(..., description="Target backend: chatterbox")
    source_files: List[str] = Field(default=[], description="List of source audio file paths")
    transcribe: bool = Field(default=True, description="Auto-transcribe using Whisper")
    segment: bool = Field(default=True, description="Segment into 3-10s chunks using VAD")


# ============================================
# WebSocket Progress Streaming
# ============================================

async def broadcast_progress(job_id: str, data: Dict[str, Any]):
    """Broadcast progress update to all connected WebSocket clients."""
    message = json.dumps({"job_id": job_id, **data})
    if len(ws_connections) == 0:
        logger.warning(f"[{job_id}] No WebSocket clients connected to broadcast to")
        return
        
    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_text(message)
            logger.debug(f"[{job_id}] Sent WS message: {data.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"[{job_id}] WebSocket send error: {e}")
            disconnected.append(ws)
    
    # Clean up disconnected clients
    for ws in disconnected:
        if ws in ws_connections:
            ws_connections.remove(ws)
            logger.info(f"Removed disconnected WebSocket. Remaining: {len(ws_connections)}")


def parse_training_output(line: str, backend: str) -> Optional[Dict[str, Any]]:
    """Parse training output line to extract progress information."""
    progress = {}
    
    # Clean line of ANSI codes and carriage returns
    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    clean_line = re.sub(r'[\r]', ' ', clean_line)
    
    # tqdm progress format: "| 12/1650 [00:22<54:22, 1.99s/it]" or "50%|#####     | 11/22 [00:11<00:11, 1.06s/it]"
    tqdm_match = re.search(r'(\d+)%?\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]', clean_line)
    if tqdm_match:
        progress["step"] = int(tqdm_match.group(2))
        progress["total_steps"] = int(tqdm_match.group(3))
        progress["percent"] = int(tqdm_match.group(1)) if tqdm_match.group(1) else (int(tqdm_match.group(2)) * 100 // int(tqdm_match.group(3)))
        # Parse time info like "00:22<54:22"
        time_info = tqdm_match.group(4)
        eta_match = re.search(r'<([\d:]+)', time_info)
        if eta_match:
            eta_str = eta_match.group(1)
            parts = eta_str.split(':')
            if len(parts) == 3:
                progress["eta_seconds"] = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                progress["eta_seconds"] = int(parts[0]) * 60 + int(parts[1])
    
    # Chatterbox dict output: {'loss': 12.8334, 'grad_norm': ..., 'epoch': 1.0}
    # Extract values directly with simple regexes instead of trying to parse JSON
    if "'loss'" in clean_line:
        loss_val = re.search(r"'loss':\s*([\d.]+)", clean_line)
        if loss_val:
            progress["loss"] = float(loss_val.group(1))
        
        grad_val = re.search(r"'grad_norm':\s*([\d.]+)", clean_line)
        if grad_val:
            progress["grad_norm"] = float(grad_val.group(1))
        
        epoch_val = re.search(r"'epoch':\s*([\d.]+)", clean_line)
        if epoch_val:
            progress["epoch"] = int(float(epoch_val.group(1)))
        
        lr_val = re.search(r"'learning_rate':\s*([\d.e-]+)", clean_line)
        if lr_val:
            progress["learning_rate"] = float(lr_val.group(1))
    
    # Common patterns - use clean_line for all
    epoch_match = re.search(r'[Ee]poch[:\s]+(\d+)(?:/(\d+))?', clean_line)
    if epoch_match:
        progress["epoch"] = int(epoch_match.group(1))
        if epoch_match.group(2):
            progress["total_epochs"] = int(epoch_match.group(2))
    
    # Audio loss pattern
    audio_loss_match = re.search(r'audio loss[:\s]+([\d.]+)', clean_line)
    if audio_loss_match:
        progress["loss"] = float(audio_loss_match.group(1))
    elif "loss" not in progress:
        # Generic loss extraction
        loss_match = re.search(r'[Ll]oss[:\s]+([\d.]+)', clean_line)
        if loss_match:
            progress["loss"] = float(loss_match.group(1))
    
    lr_match = re.search(r'[Ll]r[:\s]+([\d.e-]+)', clean_line)
    if lr_match:
        progress["learning_rate"] = float(lr_match.group(1))
    
    # Step match - but avoid false positives like "step 250,000"  
    step_match = re.search(r'[Ss]tep[:\s]+(\d+)(?:/(\d+))?(?![,\d])', clean_line)
    if step_match and ',' not in clean_line[step_match.start():step_match.end()+5]:
        progress["step"] = int(step_match.group(1))
        if step_match.group(2):
            progress["total_steps"] = int(step_match.group(2))
    
    # GPU memory
    mem_match = re.search(r'(?:GPU|VRAM|[Mm]em(?:ory)?)[:\s]+([\d.]+)\s*(?:GB|MB|G|M)', clean_line)
    if mem_match:
        progress["gpu_memory"] = mem_match.group(1)
    
    return progress if progress else None


async def run_training_process(job: TrainingJob):
    """Run training process in background and stream progress."""
    try:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        
        await broadcast_progress(job.job_id, {
            "type": "status",
            "status": "running",
            "message": f"Starting {job.backend} training..."
        })
        
        if job.backend == "chatterbox":
            await run_chatterbox_training(job)
        else:
            raise ValueError(f"Unknown backend: {job.backend}")
        
        if job.status == JobStatus.RUNNING:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            
            await broadcast_progress(job.job_id, {
                "type": "status",
                "status": "completed",
                "message": "Training completed successfully!",
                "output_model": job.output_model_path
            })
            
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now().isoformat()
        logger.error(f"Training job {job.job_id} failed: {e}")
        
        await broadcast_progress(job.job_id, {
            "type": "status",
            "status": "failed",
            "error": str(e)
        })


async def run_chatterbox_training(job: TrainingJob):
    """Run Chatterbox fine-tuning."""
    config = job.config
    env_name = "chatterbox_train"
    finetune_dir = os.path.join(TRAINING_DIR, "chatterbox-finetuning")
    
    if not os.path.exists(finetune_dir):
        raise FileNotFoundError(f"Chatterbox-finetuning not found at {finetune_dir}. Run install_chatterbox_train.bat first.")
    
    output_dir = os.path.join(CHATTERBOX_CUSTOM_DIR, config["model_name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Modify the config.py file directly in the finetuning repo
    config_file = os.path.join(finetune_dir, "src", "config.py")
    config_backup = os.path.join(finetune_dir, "src", "config.py.backup")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")
    
    # Backup original config
    import shutil
    shutil.copy(config_file, config_backup)
    
    try:
        # Generate new config.py with user's settings using TrainConfig dataclass
        is_turbo = config.get("is_turbo", True)
        dataset_path_escaped = config["dataset_path"].replace("\\", "\\\\")
        output_dir_escaped = output_dir.replace("\\", "\\\\")
        preprocess_dir = os.path.join(output_dir, "preprocess").replace("\\", "\\\\")
        
        new_config = f'''from dataclasses import dataclass

@dataclass
class TrainConfig:
    # --- Paths ---
    # Directory where setup.py downloaded the files
    model_dir: str = "./pretrained_models"
    
    # Path to your metadata CSV (Format: ID|RawText|NormText)
    csv_path: str = r"{dataset_path_escaped}\\metadata.csv"
    metadata_path: str = "./metadata.json"
    
    # Directory containing WAV files
    wav_dir: str = r"{dataset_path_escaped}\\wavs"
    
    preprocessed_dir = r"{preprocess_dir}"
    
    # Output directory for the finetuned model
    output_dir: str = r"{output_dir_escaped}"
    
    is_inference = False
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "Test text for inference."

    ljspeech = True  # LJSpeech format dataset
    json_format = False
    preprocess = {config.get("preprocess", True)}
    
    is_turbo: bool = {is_turbo}

    # --- Vocabulary ---
    new_vocab_size: int = 52260 if {is_turbo} else 2454 

    # --- Hyperparameters ---
    batch_size: int = {config.get("batch_size", 4)}
    grad_accum: int = {config.get("gradient_accumulation", 8)}
    learning_rate: float = {config.get("learning_rate", 5e-5)}
    num_epochs: int = {config.get("epochs", 150)}
    
    save_steps: int = 500
    save_total_limit: int = 2
    dataloader_num_workers: int = 0  # Must be 0 on Windows to avoid multiprocessing issues

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = 3.0
'''
        
        with open(config_file, "w") as f:
            f.write(new_config)
        
        logger.info(f"Updated config.py with dataset: {config['dataset_path']}, epochs: {config.get('epochs', 150)}")
        
        # Run setup.py to download required models for the selected mode (turbo vs standard)
        await broadcast_progress(job.job_id, {
            "status": "running",
            "message": "Checking/downloading required model files..."
        })
        
        setup_script = os.path.join(finetune_dir, "setup.py")
        if os.path.exists(setup_script):
            env_dir = os.path.join(CONDA_BASE, "envs", env_name)
            python_exe = os.path.join(env_dir, "python.exe")
            
            if os.path.exists(python_exe):
                logger.info("Running setup.py to ensure models are downloaded...")
                setup_process = subprocess.Popen(
                    [python_exe, setup_script],
                    cwd=finetune_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )
                
                # Stream setup output
                for line in iter(setup_process.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        logger.info(f"[setup] {line}")
                        await broadcast_progress(job.job_id, {
                            "status": "running",
                            "message": f"Setup: {line[:100]}"
                        })
                
                setup_process.wait()
                if setup_process.returncode != 0:
                    logger.warning(f"setup.py returned code {setup_process.returncode}, continuing anyway...")
        
        # Create simple training launcher script with Windows multiprocessing guards
        train_script = f'''
import sys
import os
from multiprocessing import freeze_support

def main():
    # Set working directory to chatterbox-finetuning
    os.chdir(r"{finetune_dir}")
    sys.path.insert(0, r"{finetune_dir}")

    print("[INFO] Chatterbox training started")
    print("[INFO] Mode: {'Turbo (GPT-2)' if {is_turbo} else 'Standard (Llama)'}")
    print("[INFO] Dataset: {config['dataset_path']}")
    print("[INFO] Output: {output_dir}")
    print("[INFO] Epochs: {config.get('epochs', 150)}")
    print("[INFO] Learning Rate: {config.get('learning_rate', 5e-5)}")
    print("[INFO] Batch Size: {config.get('batch_size', 4)}")
    print("[INFO] Gradient Accumulation: {config.get('gradient_accumulation', 8)}")
    print()

    # Import and run training
    try:
        import train
        train.main()
        print("[INFO] Training completed!")
    except Exception as e:
        print(f"[ERROR] Training failed: {{e}}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    freeze_support()
    main()
'''
        
        script_path = os.path.join(output_dir, "train_chatterbox.py")
        with open(script_path, "w") as f:
            f.write(train_script)
        
        # Run training in conda environment
        await run_conda_script(job, env_name, script_path, "chatterbox")
        
    finally:
        # Restore original config
        if os.path.exists(config_backup):
            shutil.copy(config_backup, config_file)
            os.remove(config_backup)
    
    model_file = "t3_turbo_finetuned.safetensors" if config.get("is_turbo", True) else "t3_finetuned.safetensors"
    job.output_model_path = os.path.join(output_dir, model_file)


async def run_conda_script(job: TrainingJob, env_name: str, script_path: str, backend: str):
    """Run a Python script in a conda environment and stream output."""
    if not CONDA_BASE:
        raise EnvironmentError("Conda not found. Please install Miniconda.")
    
    env_dir = os.path.join(CONDA_BASE, "envs", env_name)
    if not os.path.exists(env_dir):
        raise EnvironmentError(f"Conda environment '{env_name}' not found. Run install_{backend}_train.bat first.")
    
    python_exe = os.path.join(env_dir, "python.exe")
    if not os.path.exists(python_exe):
        raise EnvironmentError(f"Python not found in environment '{env_name}'")
    
    # Use asyncio subprocess for non-blocking I/O
    process = await asyncio.create_subprocess_exec(
        python_exe, script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    
    job.process = process
    
    # Stream output asynchronously
    # tqdm uses \r (carriage return) to update progress in-place, not \n
    # So we read raw bytes and split on both \r and \n
    buffer = b""
    last_progress_time = 0
    try:
        while True:
            if job.status == JobStatus.CANCELLED:
                process.terminate()
                break
            
            # Read available data with timeout (don't block forever waiting for \n)
            try:
                chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=0.5)
            except asyncio.TimeoutError:
                # Process buffer if we have data but no newline yet
                if buffer:
                    line = buffer.decode('utf-8', errors='replace').strip()
                    buffer = b""
                    if line:
                        # Only log periodically to avoid spam
                        now = time.time()
                        if now - last_progress_time > 2.0:  # At most every 2 seconds
                            last_progress_time = now
                            log_line = line[:500] + "..." if len(line) > 500 else line
                            logger.info(f"[{job.job_id}] {log_line}")
                            
                            # Parse progress
                            progress = parse_training_output(line, backend)
                            if progress:
                                job.progress.update(progress)
                                logger.info(f"[{job.job_id}] Progress update: {job.progress}")
                                await broadcast_progress(job.job_id, {
                                    "type": "progress",
                                    "backend": backend,
                                    **job.progress
                                })
                continue
            
            if not chunk:
                # EOF - process remaining buffer
                if buffer:
                    line = buffer.decode('utf-8', errors='replace').strip()
                    if line:
                        log_line = line[:500] + "..." if len(line) > 500 else line
                        logger.info(f"[{job.job_id}] {log_line}")
                        progress = parse_training_output(line, backend)
                        if progress:
                            job.progress.update(progress)
                            await broadcast_progress(job.job_id, {
                                "type": "progress",
                                "backend": backend,
                                **job.progress
                            })
                break
            
            buffer += chunk
            
            # Split on both \r and \n (tqdm uses \r for in-place updates)
            while b'\r' in buffer or b'\n' in buffer:
                # Find the first separator
                r_idx = buffer.find(b'\r')
                n_idx = buffer.find(b'\n')
                
                if r_idx == -1:
                    sep_idx = n_idx
                elif n_idx == -1:
                    sep_idx = r_idx
                else:
                    sep_idx = min(r_idx, n_idx)
                
                line_bytes = buffer[:sep_idx]
                # Skip the separator (and handle \r\n as single separator)
                if sep_idx + 1 < len(buffer) and buffer[sep_idx:sep_idx+2] == b'\r\n':
                    buffer = buffer[sep_idx+2:]
                else:
                    buffer = buffer[sep_idx+1:]
                
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                
                # Truncate very long lines for logging
                log_line = line[:500] + "..." if len(line) > 500 else line
                logger.info(f"[{job.job_id}] {log_line}")
                
                # Parse progress
                progress = parse_training_output(line, backend)
                if progress:
                    job.progress.update(progress)
                    logger.info(f"[{job.job_id}] Progress update: {job.progress}")
                    await broadcast_progress(job.job_id, {
                        "type": "progress",
                        "backend": backend,
                        **job.progress
                    })
                
                # Also broadcast raw log (truncated for WebSocket)
                await broadcast_progress(job.job_id, {
                    "type": "log",
                    "message": log_line
                })
        
        await process.wait()
        
        if process.returncode != 0 and job.status != JobStatus.CANCELLED:
            raise RuntimeError(f"Training process exited with code {process.returncode}")
            
    except Exception as e:
        if process.returncode is None:
            process.terminate()
        raise


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    return {"service": "VoiceForge Training Server", "status": "running"}


@app.get("/v1/status")
async def get_status():
    """Get server status and active jobs."""
    active_jobs = [j.job_id for j in jobs.values() if j.status == JobStatus.RUNNING]
    return {
        "status": "running",
        "active_jobs": len(active_jobs),
        "total_jobs": len(jobs),
        "conda_base": CONDA_BASE,
        "training_dir": TRAINING_DIR
    }


@app.post("/v1/chatterbox/train")
async def start_chatterbox_training(request: ChatterboxTrainRequest, background_tasks: BackgroundTasks):
    """Start Chatterbox fine-tuning job."""
    job_id = str(uuid.uuid4())[:8]
    
    job = TrainingJob(
        job_id=job_id,
        backend="chatterbox",
        status=JobStatus.PENDING,
        config=request.model_dump(),
        created_at=datetime.now().isoformat()
    )
    
    with job_lock:
        jobs[job_id] = job
    
    # Start training in background
    background_tasks.add_task(run_training_process, job)
    
    return {"job_id": job_id, "status": "pending", "message": "Chatterbox training job created"}


@app.get("/v1/jobs")
async def list_jobs():
    """List all training jobs."""
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "backend": j.backend,
                "status": j.status.value,
                "config": j.config,
                "created_at": j.created_at,
                "started_at": j.started_at,
                "completed_at": j.completed_at,
                "error": j.error,
                "progress": j.progress,
                "output_model": j.output_model_path
            }
            for j in jobs.values()
        ]
    }


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a specific training job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    j = jobs[job_id]
    # Only log occasionally to reduce spam
    # logger.debug(f"Job status poll: {job_id} - status={j.status.value}")
    return {
        "job_id": j.job_id,
        "backend": j.backend,
        "status": j.status.value,
        "config": j.config,
        "created_at": j.created_at,
        "started_at": j.started_at,
        "completed_at": j.completed_at,
        "error": j.error,
        "progress": j.progress,
        "output_model": j.output_model_path
    }


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a training job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.RUNNING:
        return {"message": f"Job is not running (status: {job.status.value})"}
    
    job.status = JobStatus.CANCELLED
    
    if job.process and job.process.returncode is None:
        job.process.terminate()
    
    await broadcast_progress(job_id, {
        "type": "status",
        "status": "cancelled",
        "message": "Training cancelled by user"
    })
    
    return {"message": "Job cancelled", "job_id": job_id}


@app.get("/v1/datasets")
async def list_datasets():
    """List available datasets for Chatterbox training (LJSpeech format)."""
    all_datasets = []
    seen_names = set()
    
    def scan_dataset_dir(base_dir):
        """Scan a directory for LJSpeech-format datasets."""
        if not os.path.exists(base_dir):
            return
        for name in os.listdir(base_dir):
            if name in seen_names:
                continue  # Skip duplicates
            dataset_path = os.path.join(base_dir, name)
            if os.path.isdir(dataset_path):
                # Check for LJSpeech format (metadata.csv + wavs/)
                metadata_file = os.path.join(dataset_path, "metadata.csv")
                wavs_dir = os.path.join(dataset_path, "wavs")
                
                has_metadata = os.path.exists(metadata_file)
                has_wavs = os.path.exists(wavs_dir)
                
                # Skip if no metadata.csv at all
                if not has_metadata:
                    continue
                
                info = {
                    "name": name,
                    "path": dataset_path,
                    "valid": has_metadata,
                    "has_wavs": has_wavs,
                    "samples": 0,
                    "duration_minutes": 0
                }
                
                # Count samples
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        info["samples"] = sum(1 for _ in f)
                except:
                    pass
                
                all_datasets.append(info)
                seen_names.add(name)
    
    # Scan main datasets directory (datasets are shared between backends)
    logger.info(f"Scanning datasets directory: {DATASETS_DIR}")
    scan_dataset_dir(DATASETS_DIR)
    logger.info(f"Found {len(all_datasets)} datasets so far")
    
    # Also scan backend-specific subdirs for backwards compatibility
    chatterbox_backend_dir = os.path.join(DATASETS_DIR, "chatterbox")
    scan_dataset_dir(chatterbox_backend_dir)
    
    # Also scan training directories for existing datasets
    chatterbox_training_dir = os.path.join(TRAINING_DIR, "chatterbox-finetuning")
    scan_dataset_dir(chatterbox_training_dir)
    
    return {"chatterbox": all_datasets}


@app.post("/v1/datasets/validate")
async def validate_dataset(path: str = Form(...)):
    """Validate a dataset folder."""
    if not os.path.exists(path):
        return {"valid": False, "error": "Path does not exist"}
    
    metadata_file = os.path.join(path, "metadata.csv")
    wavs_dir = os.path.join(path, "wavs")
    
    errors = []
    warnings = []
    
    if not os.path.exists(metadata_file):
        errors.append("Missing metadata.csv")
    
    if not os.path.exists(wavs_dir):
        errors.append("Missing wavs/ directory")
    
    samples = 0
    missing_audio = []
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        samples += 1
                        audio_file = os.path.join(wavs_dir, f"{parts[0]}.wav")
                        if not os.path.exists(audio_file):
                            missing_audio.append(parts[0])
                    else:
                        warnings.append(f"Line {line_num}: Invalid format")
        except Exception as e:
            errors.append(f"Error reading metadata.csv: {e}")
    
    if missing_audio:
        if len(missing_audio) > 5:
            errors.append(f"Missing {len(missing_audio)} audio files (e.g., {', '.join(missing_audio[:5])}...)")
        else:
            errors.append(f"Missing audio files: {', '.join(missing_audio)}")
    
    return {
        "valid": len(errors) == 0,
        "samples": samples,
        "errors": errors,
        "warnings": warnings
    }


@app.post("/v1/datasets/create")
async def create_dataset(
    name: str = Form(...),
    backend: str = Form(None),  # Backend param kept for compatibility but not used for path
):
    """Create a new empty dataset folder. Datasets are shared between backends."""
    # Sanitize name
    safe_name = re.sub(r'[^\w\-_]', '_', name)
    dataset_path = os.path.join(DATASETS_DIR, safe_name)
    
    if os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Dataset already exists")
    
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "wavs"), exist_ok=True)
    
    # Create empty metadata.csv
    with open(os.path.join(dataset_path, "metadata.csv"), "w", encoding="utf-8") as f:
        pass  # Empty file
    
    return {
        "success": True,
        "name": safe_name,
        "path": dataset_path
    }


@app.post("/v1/datasets/{backend}/{name}/upload")
async def upload_audio_to_dataset(
    backend: str,
    name: str,
    file: UploadFile = File(...),
    transcription: Optional[str] = Form(None),
):
    """Upload an audio file to a dataset."""
    dataset_path = find_dataset_path(name)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    wavs_dir = os.path.join(dataset_path, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    # Generate unique filename
    file_id = str(uuid.uuid4())[:8]
    original_name = Path(file.filename).stem
    safe_name = re.sub(r'[^\w\-_]', '_', original_name)
    audio_filename = f"{safe_name}_{file_id}.wav"
    audio_path = os.path.join(wavs_dir, audio_filename)
    
    # Save uploaded file
    content = await file.read()
    with open(audio_path, "wb") as f:
        f.write(content)
    
    # If transcription provided, add to metadata
    if transcription:
        metadata_path = os.path.join(dataset_path, "metadata.csv")
        with open(metadata_path, "a", encoding="utf-8") as f:
            # LJSpeech format: filename|raw_text|normalized_text
            normalized = transcription.lower().strip()
            f.write(f"{audio_filename[:-4]}|{transcription}|{normalized}\n")
    
    return {
        "success": True,
        "filename": audio_filename,
        "path": audio_path,
        "has_transcription": transcription is not None
    }


@app.post("/v1/datasets/{backend}/{name}/transcribe")
async def transcribe_dataset(
    backend: str,
    name: str,
    background_tasks: BackgroundTasks,
):
    """Transcribe all audio files in a dataset using ASR."""
    dataset_path = find_dataset_path(name)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    wavs_dir = os.path.join(dataset_path, "wavs")
    if not os.path.exists(wavs_dir):
        raise HTTPException(status_code=400, detail="No wavs directory in dataset")
    
    # Get list of wav files
    wav_files = [f for f in os.listdir(wavs_dir) if f.endswith(".wav")]
    if not wav_files:
        raise HTTPException(status_code=400, detail="No wav files in dataset")
    
    # Create transcription job
    job_id = f"transcribe_{str(uuid.uuid4())[:8]}"
    
    async def transcribe_files():
        """Background task to transcribe all files."""
        import httpx
        
        metadata_path = os.path.join(dataset_path, "metadata.csv")
        
        # Read existing entries
        existing = set()
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if parts:
                        existing.add(parts[0])
        
        results = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, wav_file in enumerate(wav_files):
                file_id = wav_file[:-4]  # Remove .wav
                
                if file_id in existing:
                    continue  # Already transcribed
                
                wav_path = os.path.join(wavs_dir, wav_file)
                
                try:
                    # Call ASR server
                    with open(wav_path, "rb") as f:
                        files = {"audio": (wav_file, f, "audio/wav")}
                        response = await client.post(
                            "http://0.0.0.0:8889/transcribe",
                            files=files
                        )
                    
                    if response.status_code == 200:
                        data = response.json()
                        transcription = data.get("text", "").strip()
                        
                        if transcription:
                            # Add to metadata
                            with open(metadata_path, "a", encoding="utf-8") as f:
                                normalized = transcription.lower().strip()
                                f.write(f"{file_id}|{transcription}|{normalized}\n")
                            
                            results.append({"file": wav_file, "text": transcription})
                    
                    # Broadcast progress
                    await broadcast_progress(job_id, {
                        "type": "transcribe_progress",
                        "current": i + 1,
                        "total": len(wav_files),
                        "file": wav_file
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to transcribe {wav_file}: {e}")
        
        await broadcast_progress(job_id, {
            "type": "transcribe_complete",
            "total": len(results)
        })
    
    background_tasks.add_task(transcribe_files)
    
    return {
        "job_id": job_id,
        "total_files": len(wav_files),
        "message": "Transcription started"
    }


@app.post("/v1/datasets/{backend}/{name}/segment")
async def segment_audio(
    backend: str,
    name: str,
    source_file: str = Form(...),
    min_duration: float = Form(3.0),
    max_duration: float = Form(10.0),
    background_tasks: BackgroundTasks = None,
):
    """Segment a long audio file into chunks using VAD."""
    dataset_path = find_dataset_path(name)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not os.path.exists(source_file):
        raise HTTPException(status_code=404, detail="Source file not found")
    
    wavs_dir = os.path.join(dataset_path, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    job_id = f"segment_{str(uuid.uuid4())[:8]}"
    
    async def do_segmentation():
        """Background task to segment audio using VAD."""
        try:
            import torch
            import torchaudio
            
            # Load silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            
            # Load audio
            wav = read_audio(source_file, sampling_rate=16000)
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                wav, model,
                min_speech_duration_ms=int(min_duration * 1000),
                max_speech_duration_s=max_duration,
                min_silence_duration_ms=500,
                sampling_rate=16000
            )
            
            # Save segments
            base_name = Path(source_file).stem
            segments = []
            
            for i, ts in enumerate(speech_timestamps):
                start_sample = ts['start']
                end_sample = ts['end']
                segment = wav[start_sample:end_sample]
                
                segment_name = f"{base_name}_seg{i:04d}"
                segment_path = os.path.join(wavs_dir, f"{segment_name}.wav")
                
                # Save segment
                torchaudio.save(
                    segment_path,
                    segment.unsqueeze(0),
                    16000
                )
                
                duration = (end_sample - start_sample) / 16000
                segments.append({
                    "name": segment_name,
                    "duration": duration,
                    "path": segment_path
                })
                
                await broadcast_progress(job_id, {
                    "type": "segment_progress",
                    "current": i + 1,
                    "total": len(speech_timestamps),
                    "segment": segment_name,
                    "duration": duration
                })
            
            await broadcast_progress(job_id, {
                "type": "segment_complete",
                "total_segments": len(segments)
            })
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            await broadcast_progress(job_id, {
                "type": "segment_error",
                "error": str(e)
            })
    
    if background_tasks:
        background_tasks.add_task(do_segmentation)
    
    return {
        "job_id": job_id,
        "message": "Segmentation started"
    }


@app.post("/v1/datasets/{backend}/{name}/segment-all")
async def segment_all_audio(
    backend: str,
    name: str,
    min_duration: float = Form(3.0),
    max_duration: float = Form(10.0),
    background_tasks: BackgroundTasks = None,
):
    """Segment all audio files in the dataset's wavs/ folder using VAD."""
    dataset_path = find_dataset_path(name)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    wavs_dir = os.path.join(dataset_path, "wavs")
    if not os.path.exists(wavs_dir):
        raise HTTPException(status_code=400, detail="No wavs directory in dataset")
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    audio_files = [f for f in os.listdir(wavs_dir) 
                   if os.path.splitext(f)[1].lower() in audio_extensions]
    
    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio files found in dataset")
    
    job_id = f"segment_all_{str(uuid.uuid4())[:8]}"
    
    async def do_segmentation_all():
        """Background task to segment all audio files using VAD."""
        try:
            import torch
            import torchaudio
            
            # Load silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            
            total_segments = 0
            
            for file_idx, audio_file in enumerate(audio_files):
                source_path = os.path.join(wavs_dir, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                
                await broadcast_progress(job_id, {
                    "type": "segment_file_start",
                    "file": audio_file,
                    "file_index": file_idx + 1,
                    "total_files": len(audio_files)
                })
                
                try:
                    # Load audio
                    wav = read_audio(source_path, sampling_rate=16000)
                    
                    # Get speech timestamps
                    speech_timestamps = get_speech_timestamps(
                        wav, model,
                        min_speech_duration_ms=int(min_duration * 1000),
                        max_speech_duration_s=max_duration,
                        min_silence_duration_ms=500,
                        sampling_rate=16000
                    )
                    
                    if len(speech_timestamps) <= 1:
                        # File is already short enough, skip segmentation
                        logger.info(f"File {audio_file} doesn't need segmentation (already short)")
                        continue
                    
                    # Delete original file (will be replaced by segments)
                    os.remove(source_path)
                    
                    # Save segments
                    for i, ts in enumerate(speech_timestamps):
                        start_sample = ts['start']
                        end_sample = ts['end']
                        segment = wav[start_sample:end_sample]
                        
                        segment_name = f"{base_name}_seg{i:04d}"
                        segment_path = os.path.join(wavs_dir, f"{segment_name}.wav")
                        
                        torchaudio.save(
                            segment_path,
                            segment.unsqueeze(0),
                            16000
                        )
                        total_segments += 1
                    
                    logger.info(f"Segmented {audio_file} into {len(speech_timestamps)} segments")
                    
                except Exception as e:
                    logger.error(f"Failed to segment {audio_file}: {e}")
            
            await broadcast_progress(job_id, {
                "type": "segment_all_complete",
                "total_files": len(audio_files),
                "total_segments": total_segments
            })
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            await broadcast_progress(job_id, {
                "type": "segment_error",
                "error": str(e)
            })
    
    if background_tasks:
        background_tasks.add_task(do_segmentation_all)
    
    return {
        "job_id": job_id,
        "total_files": len(audio_files),
        "message": f"Segmenting {len(audio_files)} audio files..."
    }


@app.get("/v1/datasets/{backend}/{name}/files")
async def list_dataset_files(backend: str, name: str):
    """List all files in a dataset."""
    dataset_path = find_dataset_path(name)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    wavs_dir = os.path.join(dataset_path, "wavs")
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    
    # Load transcribed files from metadata
    transcribed = set()
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if parts:
                        transcribed.add(parts[0])
        except:
            pass
    
    files = []
    if os.path.exists(wavs_dir):
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        for f in sorted(os.listdir(wavs_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in audio_extensions:
                file_id = os.path.splitext(f)[0]
                files.append({
                    "name": f,
                    "id": file_id,
                    "transcribed": file_id in transcribed
                })
    
    return {
        "files": files,
        "total": len(files),
        "transcribed_count": len([f for f in files if f["transcribed"]])
    }


@app.delete("/v1/datasets/{backend}/{name}")
async def delete_dataset(backend: str, name: str):
    """Delete a dataset."""
    dataset_path = find_dataset_path(name)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    import shutil
    shutil.rmtree(dataset_path)
    
    return {"success": True, "message": f"Dataset '{name}' deleted"}


@app.get("/v1/models")
async def list_custom_models():
    """List custom trained models."""
    models = {"chatterbox": []}
    
    # Chatterbox models
    if os.path.exists(CHATTERBOX_CUSTOM_DIR):
        for name in os.listdir(CHATTERBOX_CUSTOM_DIR):
            model_path = os.path.join(CHATTERBOX_CUSTOM_DIR, name)
            if os.path.isdir(model_path):
                # Look for safetensors file
                safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
                models["chatterbox"].append({
                    "name": name,
                    "path": model_path,
                    "files": safetensors_files,
                    "created": datetime.fromtimestamp(os.path.getctime(model_path)).isoformat()
                })
    
    return models


@app.websocket("/ws/training")
async def websocket_training_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time training progress."""
    await websocket.accept()
    ws_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(ws_connections)}")
    
    # Send immediate confirmation
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Training WebSocket connected"
        }))
    except Exception as e:
        logger.error(f"Failed to send connection confirmation: {e}")
    
    try:
        while True:
            # Use asyncio.wait_for to allow periodic heartbeats
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Client can request status of specific job
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "subscribe" and msg.get("job_id"):
                        job_id = msg["job_id"]
                        if job_id in jobs:
                            job = jobs[job_id]
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "job_id": job_id,
                                "status": job.status.value,
                                "progress": job.progress
                            }))
                    elif msg.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
                except Exception:
                    break
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(ws_connections)}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import argparse

    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(description="VoiceForge Training Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8895, help="Port to bind to")
    args = parser.parse_args()
    
    logger.info(f"Starting VoiceForge Training Server on {args.host}:{args.port}")
    logger.info(f"Training directory: {TRAINING_DIR}")
    logger.info(f"Datasets directory: {DATASETS_DIR}")
    logger.info(f"Conda base: {CONDA_BASE}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=os.getenv("VF_UVICORN_LOG_LEVEL", "warning").lower(),
        access_log=_env_flag("VF_ACCESS_LOGS", "0"),
    )
