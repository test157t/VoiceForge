"""
Files Router - File upload and management endpoints.

Handles:
- /api/upload - General file upload
- /api/scripts/* - Script management
- /api/background-audio/* - Background audio management
"""

import asyncio
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import Response, FileResponse

# Import common (sets up sys.path)
from .common import verify_auth, ASSETS_DIR, SOUNDS_DIR, SCRIPT_DIR, APP_DIR

from util.clients import run_blend
from util.audio_utils import convert_to_wav
from util.file_utils import resolve_audio_path
from config import get_config, ensure_dir


router = APIRouter(tags=["Files"])

_executor: Optional[ThreadPoolExecutor] = None


UPLOAD_CHUNK_SIZE = 1024 * 1024


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", os.cpu_count() or 4)))
    return _executor


async def _write_upload_to_path(upload: UploadFile, dest_path: str, chunk_size: int = UPLOAD_CHUNK_SIZE) -> int:
    """Write an UploadFile to disk incrementally to avoid large in-memory reads."""
    total_bytes = 0
    await upload.seek(0)
    with open(dest_path, "wb") as out_file:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            total_bytes += len(chunk)
    return total_bytes


# ======================
# File Serving
# ======================

@router.get("/api/file")
async def serve_file(
    path: str = Query(..., description="Path to the file to serve"),
    _: bool = Depends(verify_auth)
):
    """
    Serve a file by path.
    
    Only serves files from allowed directories (assets, sounds, output, temp).
    Used for streaming background audio playback.
    """
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security: only allow files from known safe directories
    abs_path = os.path.abspath(path)
    allowed_dirs = [
        os.path.abspath(ASSETS_DIR),
        os.path.abspath(SOUNDS_DIR),
        os.path.abspath(os.path.join(APP_DIR, "output")),
        os.path.abspath(tempfile.gettempdir()),
    ]
    
    is_allowed = any(abs_path.startswith(d) for d in allowed_dirs)
    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")
    
    # Determine media type
    ext = os.path.splitext(path)[1].lower()
    media_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
    }
    media_type = media_types.get(ext, 'application/octet-stream')
    
    return FileResponse(path, media_type=media_type)


# ======================
# Audio Prompts
# ======================

AUDIO_PROMPTS_DIR = os.path.join(ASSETS_DIR, "audio_prompts")

@router.get("/api/audio-prompts")
async def list_audio_prompts(_: bool = Depends(verify_auth)):
    """
    List available audio prompt files for voice cloning.
    
    Returns files from assets/audio_prompts/ folder.
    """
    prompts = []
    
    if os.path.exists(AUDIO_PROMPTS_DIR):
        for f in os.listdir(AUDIO_PROMPTS_DIR):
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                filepath = os.path.join(AUDIO_PROMPTS_DIR, f)
                prompts.append({
                    "name": os.path.splitext(f)[0],
                    "filename": f,
                    "path": filepath
                })
    
    # Sort by name
    prompts.sort(key=lambda x: x["name"].lower())
    
    return {"prompts": prompts, "folder": AUDIO_PROMPTS_DIR}


# ======================
# General Upload
# ======================

@router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    _: bool = Depends(verify_auth)
):
    """
    Upload a file and return its path.
    
    Used for prompt audio and other file uploads.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.opus']
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Only audio files are allowed. Supported: {', '.join(allowed_extensions)}"
        )
    
    fd, temp_path = tempfile.mkstemp(suffix=file_ext, prefix="prompt_audio_")
    os.close(fd)
    
    try:
        await _write_upload_to_path(file, temp_path)
        
        return {"path": temp_path, "filename": file.filename}
    except Exception as e:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")


# ======================
# Scripts
# ======================

@router.get("/api/scripts")
async def list_scripts(_: bool = Depends(verify_auth)):
    """List available script files."""
    scripts = []
    if os.path.exists(SCRIPT_DIR):
        for f in os.listdir(SCRIPT_DIR):
            if f.endswith('.txt'):
                scripts.append(f)
    return {"scripts": scripts}


@router.get("/api/scripts/{script_name}")
async def get_script(script_name: str, _: bool = Depends(verify_auth)):
    """Get content of a script file."""
    try:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        if not os.path.exists(script_path):
            raise HTTPException(status_code=404, detail="Script not found")
        
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content, "name": script_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading script: {str(e)}")


@router.post("/api/scripts/upload")
async def upload_script(
    file: UploadFile = File(...),
    _: bool = Depends(verify_auth)
):
    """Upload a script file."""
    if not file.filename or not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    ensure_dir(SCRIPT_DIR)
    dest_path = os.path.join(SCRIPT_DIR, file.filename)
    
    try:
        await _write_upload_to_path(file, dest_path)
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# Background Audio
# ======================

@router.get("/api/background-audio")
async def get_background_audio(_: bool = Depends(verify_auth)):
    """List available background audio files."""
    files = []
    audio_extensions = ('.mp3', '.wav', '.ogg', '.flac')
    
    if os.path.exists(SOUNDS_DIR):
        for dir_path, _, filenames in os.walk(SOUNDS_DIR):
            for f in filenames:
                if f.lower().endswith(audio_extensions):
                    # Return absolute paths so resolve_audio_path can find them
                    abs_path = os.path.abspath(os.path.join(dir_path, f))
                    files.append(abs_path.replace("\\", "/"))
    
    return {"files": files}


@router.post("/api/background-audio/upload")
async def upload_background_audio(
    file: UploadFile = File(...), 
    _: bool = Depends(verify_auth)
):
    """Upload a background audio file."""
    audio_extensions = ('.mp3', '.wav', '.ogg', '.flac')
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    
    if file_ext not in audio_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Supported: {audio_extensions}"
        )
    
    ensure_dir(SOUNDS_DIR)
    dest_path = os.path.join(SOUNDS_DIR, file.filename)
    
    # Avoid overwriting
    base_name = os.path.splitext(file.filename)[0]
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(SOUNDS_DIR, f"{base_name}_{counter}{file_ext}")
        counter += 1
    
    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        
        rel_path = os.path.relpath(dest_path, ASSETS_DIR).replace("\\", "/")
        return {"filename": os.path.basename(dest_path), "path": rel_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/background-audio/blend")
async def blend_background_audio(
    request: Request,
    main_audio: UploadFile = File(...),
    bg_tracks_json: Optional[str] = Form(None),
    main_volume: Optional[float] = Form(None),
    _: bool = Depends(verify_auth)
):
    """
    Blend a main audio file with background tracks.
    
    Args:
        main_audio: Main audio file
        bg_tracks_json: JSON array of {file, volume, delay} objects
        main_volume: Volume for main audio
    """
    temp_files = []
    
    try:
        # Load config defaults
        config = get_config()
        default_bg_tracks = config.get("bg_tracks", [])
        default_main_volume = config.get("main_audio_volume", 1.0)
        
        if main_volume is None:
            main_volume = default_main_volume
        
        # Parse background tracks
        bg_files = []
        bg_volumes = []
        bg_delays = []
        bg_fade_ins = []
        bg_fade_outs = []
        
        if bg_tracks_json:
            try:
                tracks_data = json.loads(bg_tracks_json)
                for track in tracks_data:
                    if isinstance(track, dict) and "file" in track:
                        bg_files.append(str(track["file"]))
                        bg_volumes.append(float(track.get("volume", 0.3)))
                        bg_delays.append(float(track.get("delay", 0)))
                        bg_fade_ins.append(float(track.get("fade_in", 0)))
                        bg_fade_outs.append(float(track.get("fade_out", 0)))
            except:
                pass
        
        # Fall back to config tracks
        if not bg_files and default_bg_tracks:
            for track in default_bg_tracks:
                if track and track.get("file"):
                    bg_files.append(str(track["file"]))
                    bg_volumes.append(float(track.get("volume", 0.3)))
                    bg_delays.append(float(track.get("delay", 0)))
                    bg_fade_ins.append(float(track.get("fade_in", 0)))
                    bg_fade_outs.append(float(track.get("fade_out", 0)))
        
        if not bg_files:
            raise HTTPException(status_code=400, detail="No background tracks provided")
        
        # Save main audio
        file_ext = os.path.splitext(main_audio.filename)[1] if main_audio.filename else ".tmp"
        fd_main, tmp_main = tempfile.mkstemp(suffix=file_ext)
        os.close(fd_main)
        temp_files.append(tmp_main)
        
        await _write_upload_to_path(main_audio, tmp_main)
        
        # Convert to WAV, preserving original sample rate and quality
        from util.audio_utils import get_audio_info
        
        fd_wav, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd_wav)
        temp_files.append(tmp_wav)
        
        if not tmp_main.lower().endswith('.wav'):
            # Get original audio info to preserve quality
            audio_info = get_audio_info(tmp_main)
            original_sr = audio_info.get('samplerate', 44100)
            original_channels = audio_info.get('channels', 2)
            # Preserve original sample rate and channels for quality
            convert_to_wav(tmp_main, tmp_wav, sample_rate=original_sr, channels=original_channels)
            main_wav_path = tmp_wav
        else:
            # Already WAV, just copy to preserve original quality
            shutil.copy2(tmp_main, tmp_wav)
            main_wav_path = tmp_wav
        
        # Resolve background paths
        active_tracks = []
        for i, bg_file in enumerate(bg_files):
            if bg_file:
                vol = bg_volumes[i] if i < len(bg_volumes) else 0.3
                delay = bg_delays[i] if i < len(bg_delays) else 0.0
                fade_in = bg_fade_ins[i] if i < len(bg_fade_ins) else 0.0
                fade_out = bg_fade_outs[i] if i < len(bg_fade_outs) else 0.0
                if vol > 0:
                    resolved = resolve_audio_path(bg_file)
                    if resolved and os.path.exists(resolved):
                        active_tracks.append((resolved, vol, delay, fade_in, fade_out))
        
        if not active_tracks:
            raise HTTPException(status_code=400, detail="No valid background tracks found")
        
        # Blend
        bg_paths = [p for p, _, _, _, _ in active_tracks]
        bg_vols = [v for _, v, _, _, _ in active_tracks]
        bg_delay_list = [d for _, _, d, _, _ in active_tracks]
        bg_fade_in_list = [fi for _, _, _, fi, _ in active_tracks]
        bg_fade_out_list = [fo for _, _, _, _, fo in active_tracks]
        
        def do_blend():
            return run_blend(main_wav_path, bg_paths, bg_vols, main_volume, lambda s: None, bg_delay_list, bg_fade_in_list, bg_fade_out_list)
        
        executor = _get_executor()
        blended_path = await asyncio.get_event_loop().run_in_executor(executor, do_blend)
        temp_files.append(blended_path)
        
        # Return result
        with open(blended_path, "rb") as f:
            audio_data = f.read()
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="blended_audio.wav"'}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
