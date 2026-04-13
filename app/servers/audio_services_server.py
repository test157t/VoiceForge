"""
Unified Audio Services microservice for VoiceForge.

Combines:
- Post-processing (FFmpeg-based): /v1/postprocess, /v1/blend, /v1/master, /v1/save
- Preprocess (UVR5): /v1/preprocess/uvr5/clean-vocals, /v1/preprocess/uvr5/unload
- Background audio listing: /v1/background/list

Runs in a single conda env: "audio_services".
"""

import os
import sys
import json
import io
import math
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from starlette.background import BackgroundTask
from pydantic import BaseModel

# This file is in app/servers/, set up paths
SCRIPT_DIR = Path(__file__).parent  # app/servers
APP_DIR = SCRIPT_DIR.parent  # app
UTIL_DIR = APP_DIR / "util"
CONFIG_DIR = APP_DIR / "config"
MODELS_DIR = APP_DIR / "models"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(UTIL_DIR))
sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(MODELS_DIR))

from logging_utils import create_server_logger, suppress_library_loggers, configure_server_warnings

log_info, log_warn, log_error = create_server_logger("AUDIO")
configure_server_warnings()
suppress_library_loggers()

from config import OUTPUT_DIR, ensure_dir
from audio_utils import convert_to_wav, is_wav_file
from file_utils import list_background_audio
from models import uvr5 as uvr5_models
from spatial_audio import process_spatial_audio_file, process_spatial_audio_buffer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time


# =============================================================================
# Streaming Background Mixer - Maintains state across chunk requests
# =============================================================================

class StreamingBackgroundMixer:
    """
    Efficiently mix background audio into streaming chunks.
    
    Features:
    - Memory cache: Keeps loaded audio in RAM for instant reuse
    - Disk cache: Saves pre-processed .npy files for fast reload after restart
    - Parallel loading: Loads all tracks simultaneously
    - Uses PyAV (direct FFmpeg binding) for fast loading
    """
    
    # Class-level cache: maps file path -> (numpy_array, sample_rate)
    _audio_cache: Dict[str, tuple] = {}
    _disk_cache_dir: Optional[str] = None
    _cache_lock = threading.Lock()
    
    @classmethod
    def _get_disk_cache_dir(cls) -> str:
        """Get or create disk cache directory."""
        if cls._disk_cache_dir is None:
            cls._disk_cache_dir = os.path.join(tempfile.gettempdir(), "voiceforge_bg_cache")
            os.makedirs(cls._disk_cache_dir, exist_ok=True)
        return cls._disk_cache_dir
    
    @classmethod
    def _get_cache_path(cls, audio_path: str, sample_rate: int) -> str:
        """Get the .npy cache path for an audio file."""
        import hashlib
        key = f"{audio_path}_{sample_rate}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(cls._get_disk_cache_dir(), f"{basename}_{hash_key}.npy")
    
    @classmethod
    def _get_cached_audio(cls, path: str, sample_rate: int) -> Optional[np.ndarray]:
        """Get audio from memory cache or disk cache."""
        with cls._cache_lock:
            if path in cls._audio_cache:
                cached_array, cached_sr = cls._audio_cache[path]
                if cached_sr == sample_rate:
                    return cached_array
        
        cache_path = cls._get_cache_path(path, sample_rate)
        if os.path.exists(cache_path):
            try:
                if os.path.getmtime(cache_path) >= os.path.getmtime(path):
                    samples = np.load(cache_path, mmap_mode='r').copy()
                    with cls._cache_lock:
                        cls._audio_cache[path] = (samples, sample_rate)
                    return samples
            except Exception:
                pass
        return None
    
    @classmethod
    def _cache_audio(cls, path: str, array: np.ndarray, sample_rate: int):
        """Store audio in memory and disk cache."""
        with cls._cache_lock:
            cls._audio_cache[path] = (array, sample_rate)
        try:
            cache_path = cls._get_cache_path(path, sample_rate)
            np.save(cache_path, array)
        except Exception as e:
            log_warn(f"[BGMixer] Failed to write disk cache: {e}")
    
    @classmethod
    def clear_cache(cls, memory_only: bool = False):
        """Clear cached audio."""
        with cls._cache_lock:
            cls._audio_cache.clear()
        if not memory_only:
            cache_dir = cls._get_disk_cache_dir()
            for f in os.listdir(cache_dir):
                if f.endswith('.npy'):
                    try:
                        os.remove(os.path.join(cache_dir, f))
                    except Exception:
                        pass
        log_info(f"[BGMixer] Cache cleared ({'memory only' if memory_only else 'memory + disk'})")
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """Get cache statistics."""
        with cls._cache_lock:
            total_samples = sum(arr.shape[0] for arr, _ in cls._audio_cache.values())
            memory_files = len(cls._audio_cache)
        disk_files = 0
        disk_size = 0
        try:
            cache_dir = cls._get_disk_cache_dir()
            for f in os.listdir(cache_dir):
                if f.endswith('.npy'):
                    disk_files += 1
                    disk_size += os.path.getsize(os.path.join(cache_dir, f))
        except Exception:
            pass
        return {
            "memory_files": memory_files,
            "memory_samples": total_samples,
            "memory_duration_sec": total_samples / 44100 if total_samples > 0 else 0,
            "memory_mb": total_samples * 4 / (1024 * 1024),
            "disk_files": disk_files,
            "disk_mb": disk_size / (1024 * 1024),
        }
    
    @classmethod
    def _load_single_track(cls, path: str, sample_rate: int) -> Optional[np.ndarray]:
        """Load audio file using PyAV (direct FFmpeg binding, no subprocess overhead)."""
        start = time.perf_counter()
        
        # Try PyAV first (fastest)
        try:
            import av
            
            container = av.open(path)
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"  # Multi-threaded decoding
            
            resampler = av.AudioResampler(format='s16', layout='mono', rate=sample_rate)
            
            frames = []
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                for r in resampled:
                    frames.append(r.to_ndarray())
            container.close()
            
            if not frames:
                raise ValueError("No audio frames decoded")
            
            samples = np.concatenate(frames, axis=1).flatten().astype(np.float32)
            samples = samples / 32768.0
            
            elapsed = time.perf_counter() - start
            log_info(f"[BGMixer] Loaded (PyAV): {os.path.basename(path)}, {len(samples)/sample_rate:.1f}s in {elapsed:.2f}s")
            return samples
            
        except ImportError:
            pass
        except Exception as e:
            log_warn(f"[BGMixer] PyAV failed for {os.path.basename(path)}: {e}, trying pydub...")
        
        # Fallback to pydub
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(path)
            seg = seg.set_channels(1).set_frame_rate(sample_rate)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0
            
            elapsed = time.perf_counter() - start
            log_info(f"[BGMixer] Loaded (pydub): {os.path.basename(path)}, {len(samples)/sample_rate:.1f}s in {elapsed:.2f}s")
            return samples
        except Exception as e:
            log_error(f"[BGMixer] Failed to load {path}: {e}")
            return None
    
    def __init__(self, tracks: List[Dict], sample_rate: int = 44100):
        """Initialize mixer with background tracks."""
        start_time = time.perf_counter()
        self.sample_rate = sample_rate
        self.current_sample = 0
        self.bg_arrays = []
        
        valid_tracks = []
        for track in tracks:
            path = track.get("file")
            vol = float(track.get("volume", 0.3))
            delay = float(track.get("delay", 0))
            if path and os.path.exists(path) and vol > 0:
                valid_tracks.append((path, vol, delay))
        
        if not valid_tracks:
            log_info("[BGMixer] No valid tracks to load")
            return
        
        cache_hits = 0
        to_load = []
        cached_results = []
        
        for path, vol, delay in valid_tracks:
            samples = self._get_cached_audio(path, sample_rate)
            if samples is not None:
                cache_hits += 1
                cached_results.append((samples, vol, delay))
                log_info(f"[BGMixer] Cache hit: {os.path.basename(path)}")
            else:
                to_load.append((path, vol, delay))
        
        if to_load:
            log_info(f"[BGMixer] Loading {len(to_load)} tracks in parallel...")
            
            def load_track(args):
                path, vol, delay = args
                samples = self._load_single_track(path, sample_rate)
                if samples is not None:
                    self._cache_audio(path, samples, sample_rate)
                return (samples, vol, delay, path)
            
            with ThreadPoolExecutor(max_workers=min(4, len(to_load))) as pool:
                futures = {pool.submit(load_track, t): t for t in to_load}
                for future in as_completed(futures):
                    try:
                        samples, vol, delay, path = future.result()
                        if samples is not None:
                            cached_results.append((samples, vol, delay))
                    except Exception as e:
                        log_error(f"[BGMixer] Error loading track: {e}")
        
        for samples, vol, delay in cached_results:
            delay_samples = int(delay * sample_rate)
            self.bg_arrays.append((samples, vol, delay_samples))
        
        elapsed = time.perf_counter() - start_time
        cache_info = self.get_cache_info()
        log_info(f"[BGMixer] Ready: {len(self.bg_arrays)} tracks in {elapsed:.2f}s "
                 f"(cache: {cache_hits} hits, {len(to_load)} loaded, "
                 f"mem: {cache_info['memory_mb']:.1f}MB, disk: {cache_info['disk_mb']:.1f}MB)")
    
    def mix_chunk(self, chunk_samples: np.ndarray) -> np.ndarray:
        """Mix background audio into a chunk (in-place modification)."""
        if not self.bg_arrays:
            return chunk_samples
        
        chunk_len = len(chunk_samples)
        
        for bg_array, vol, delay_samples in self.bg_arrays:
            start_pos = self.current_sample - delay_samples
            end_pos = start_pos + chunk_len
            
            bg_start = max(0, start_pos)
            bg_end = min(len(bg_array), end_pos)
            
            if bg_start >= bg_end or bg_start >= len(bg_array):
                continue
            
            chunk_start = max(0, delay_samples - self.current_sample)
            samples_to_mix = bg_end - bg_start
            chunk_end = chunk_start + samples_to_mix
            
            if chunk_end > chunk_len:
                chunk_end = chunk_len
                samples_to_mix = chunk_end - chunk_start
            
            if samples_to_mix > 0 and chunk_start < chunk_len:
                bg_portion = bg_array[bg_start:bg_start + samples_to_mix] * vol
                chunk_samples[chunk_start:chunk_end] += bg_portion[:chunk_end - chunk_start]
        
        self.current_sample += chunk_len
        return np.clip(chunk_samples, -1.0, 1.0)


# Session management for streaming background mixing
_bg_mixer_sessions: Dict[str, StreamingBackgroundMixer] = {}
_bg_mixer_lock = threading.Lock()


def get_or_create_bg_mixer(session_id: str, tracks: List[Dict], sample_rate: int = 44100) -> StreamingBackgroundMixer:
    """Get existing mixer session or create new one."""
    with _bg_mixer_lock:
        if session_id not in _bg_mixer_sessions:
            _bg_mixer_sessions[session_id] = StreamingBackgroundMixer(tracks, sample_rate)
        return _bg_mixer_sessions[session_id]


def end_bg_mixer_session(session_id: str):
    """End a mixer session."""
    with _bg_mixer_lock:
        if session_id in _bg_mixer_sessions:
            del _bg_mixer_sessions[session_id]
            log_info(f"[BGMixer] Session ended: {session_id}")


app = FastAPI(title="VoiceForge Audio Services Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_unlink(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _safe_rmtree(path: str) -> None:
    try:
        if path and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _cleanup_many(paths: List[str]) -> None:
    for p in paths:
        if not p:
            continue
        if os.path.isdir(p):
            _safe_rmtree(p)
        else:
            _safe_unlink(p)


# =========================================================
# Health / Background Audio
# =========================================================

@app.get("/health")
async def health():
    # Include bg mixer cache info
    bg_cache = StreamingBackgroundMixer.get_cache_info()
    uvr5_info: dict
    try:
        # Get info for all available models
        uvr5_info = uvr5_models.get_model_info()
    except Exception as e:
        uvr5_info = {"error": str(e)}

    return {
        "status": "ok",
        "service": "audio_services",
        "components": {
            "postprocess": True,
            "preprocess": True,
            "background_audio": True,
        },
        "uvr5": uvr5_info,
        "bg_cache": bg_cache,
    }


@app.get("/v1/background/list")
async def background_list():
    return {"files": list_background_audio()}


@app.get("/v1/background/cache-info")
async def background_cache_info():
    """Get background audio cache statistics."""
    return StreamingBackgroundMixer.get_cache_info()


@app.post("/v1/background/cache-clear")
async def background_cache_clear(memory_only: bool = False):
    """Clear background audio cache."""
    StreamingBackgroundMixer.clear_cache(memory_only=memory_only)
    return {"status": "ok", "memory_only": memory_only}


class PreloadRequest(BaseModel):
    files: List[str] = []
    sample_rate: int = 44100

@app.post("/v1/background/preload")
async def background_preload(request: PreloadRequest):
    """
    Pre-load background audio files into cache.
    Call this at startup or before streaming to ensure fast mixing.
    """
    tracks = [{"file": f, "volume": 1.0, "delay": 0} for f in request.files if os.path.exists(f)]
    if tracks:
        log_info(f"[BGMixer] Preloading {len(tracks)} tracks...")
        # Create a temporary mixer just to load files into cache
        StreamingBackgroundMixer(tracks, request.sample_rate)
    return StreamingBackgroundMixer.get_cache_info()


class StreamingMixerState:
    """Track position in background audio for streaming."""
    def __init__(self):
        self.elapsed_seconds = 0.0

# Session state for streaming - just tracks elapsed time
_streaming_sessions: Dict[str, StreamingMixerState] = {}
_streaming_lock = threading.Lock()

def get_streaming_state(session_id: str) -> StreamingMixerState:
    """Get or create streaming state for a session."""
    with _streaming_lock:
        if session_id not in _streaming_sessions:
            _streaming_sessions[session_id] = StreamingMixerState()
        return _streaming_sessions[session_id]

def end_streaming_session(session_id: str):
    """Clean up streaming session."""
    with _streaming_lock:
        if session_id in _streaming_sessions:
            del _streaming_sessions[session_id]


@app.post("/v1/background/mix-chunk")
async def mix_chunk(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    tracks_json: str = Form(default="[]"),  # JSON array of {file, volume, delay, fade_in, fade_out}
    sample_rate: int = Form(44100),
):
    """
    Mix background audio into a streaming chunk using FFmpeg (preserves quality).
    Supports fade_in for streaming mode. fade_out is applied proportionally if track has one.
    """
    tracks = json.loads(tracks_json) if tracks_json else []
    
    if not tracks:
        # No background, return original
        audio_data = await audio.read()
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(audio_data)
        return FileResponse(tmp, media_type="audio/wav", background=BackgroundTask(_safe_unlink, tmp))
    
    # Get streaming state (tracks elapsed time for delays)
    state = get_streaming_state(session_id)
    
    # Save uploaded chunk
    audio_data = await audio.read()
    fd, tmp_in = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    try:
        with open(tmp_in, "wb") as f:
            f.write(audio_data)
        
        # Get chunk duration
        chunk_duration = 0.0
        try:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", tmp_in
            ], capture_output=True, text=True, check=True)
            chunk_duration = float(probe.stdout.strip())
        except:
            pass
        
        # Build FFmpeg command - use amix for proper mixing
        args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", tmp_in]
        
        # Add background tracks with volume and delay adjustment
        filter_inputs = ["[0:a]anull[main]"]
        mix_inputs = ["[main]"]
        
        for i, track in enumerate(tracks):
            bg_file = track.get("file", "")
            vol = float(track.get("volume", 0.3))
            delay_s = float(track.get("delay", 0))
            fade_in = float(track.get("fade_in", 0))
            fade_out = float(track.get("fade_out", 0))
            
            if not bg_file or not os.path.exists(bg_file) or vol <= 0:
                continue
            
            # Calculate where we are in the background track
            # (accounting for delay and elapsed time)
            bg_start = max(0, state.elapsed_seconds - delay_s)
            
            args += ["-ss", str(bg_start), "-t", str(chunk_duration), "-i", bg_file]
            input_idx = len(mix_inputs)
            
            # Build filter chain for this track
            vol_db = 20 * math.log10(max(0.001, vol))
            filter_chain = f"[{input_idx}:a]"
            
            # Calculate fade-in multiplier for this chunk
            # If we're still within the fade-in period, apply partial fade
            if fade_in > 0 and state.elapsed_seconds < fade_in:
                # How much of fade-in remains
                fade_remaining = fade_in - state.elapsed_seconds
                if fade_remaining > 0:
                    # Apply fade starting at 0 for this chunk, duration is min(remaining, chunk_duration)
                    fade_dur = min(fade_remaining, chunk_duration)
                    # Calculate starting volume based on where we are in fade
                    start_vol = state.elapsed_seconds / fade_in
                    # Use volume filter with gradual increase
                    filter_chain += f"afade=t=in:st=0:d={fade_dur:.3f}:curve=tri,"
            
            # Apply volume
            filter_chain += f"volume={vol_db:.1f}dB[bg{i}]"
            filter_inputs.append(filter_chain)
            mix_inputs.append(f"[bg{i}]")
        
        if len(mix_inputs) == 1:
            # No valid background tracks, return original
            return FileResponse(tmp_in, media_type="audio/wav", background=BackgroundTask(_safe_unlink, tmp_in))
        
        # Mix all inputs
        filter_str = ";".join(filter_inputs) + ";" + "".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:normalize=0[out]"
        
        fd, tmp_out = tempfile.mkstemp(suffix="_mixed.wav")
        os.close(fd)
        
        args += ["-filter_complex", filter_str, "-map", "[out]", "-c:a", "pcm_f32le", tmp_out]
        
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            log_error(f"[BGMixer] FFmpeg failed: {result.stderr}")
            # Return original on error
            _safe_unlink(tmp_out)
            return FileResponse(tmp_in, media_type="audio/wav", background=BackgroundTask(_safe_unlink, tmp_in))
        
        # Update elapsed time
        state.elapsed_seconds += chunk_duration
        
        _safe_unlink(tmp_in)
        return FileResponse(tmp_out, media_type="audio/wav", background=BackgroundTask(_safe_unlink, tmp_out))
        
    except Exception as e:
        _safe_unlink(tmp_in)
        log_error(f"[BGMixer] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/background/end-session")
async def end_session(session_id: str = Form(...)):
    """End a streaming background mix session."""
    end_streaming_session(session_id)
    end_bg_mixer_session(session_id)  # Also clean up old mixer if any
    return {"status": "ok", "session_id": session_id}


# Active background streams (session_id -> file_path or process)
_bg_stream_processes: Dict[str, subprocess.Popen] = {}
_bg_stream_lock = threading.Lock()

# Character-based background stream tracking
# Maps character_name -> {session_id, cache_path, tracks_hash}
_active_character_streams: Dict[str, Dict] = {}
_character_stream_lock = threading.Lock()


def _get_background_stream_cache_path(tracks: List[Dict], duration: int, sample_rate: int) -> str:
    """Get cache path for background stream mix based on tracks and parameters."""
    import hashlib
    
    cache_dir = Path(OUTPUT_DIR) / "background_streams"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a deterministic cache key from tracks and parameters
    # Sort tracks by file path for consistent hashing
    sorted_tracks = sorted(tracks, key=lambda t: t.get("file", ""))
    cache_data = {
        "tracks": [
            {
                "file": track.get("file", ""),
                "volume": round(float(track.get("volume", 0.3)), 3),
                "delay": round(float(track.get("delay", 0)), 3),
                "fade_in": round(float(track.get("fade_in", 0)), 3),
                "fade_out": round(float(track.get("fade_out", 0)), 3),
            }
            for track in sorted_tracks
        ],
        "duration": duration,
        "sample_rate": sample_rate,
    }
    
    # Include file modification times in hash to detect when source files change
    for track in cache_data["tracks"]:
        file_path = track["file"]
        if os.path.exists(file_path):
            track["mtime"] = os.path.getmtime(file_path)
    
    # Create hash from cache data
    cache_json = json.dumps(cache_data, sort_keys=True)
    cache_hash = hashlib.md5(cache_json.encode()).hexdigest()[:16]
    
    cache_filename = f"bgmix_{cache_hash}_{duration}s.wav"
    return str(cache_dir / cache_filename)


class BackgroundStreamRequest(BaseModel):
    """Request for streaming background audio."""
    session_id: str
    tracks: List[Dict] = []  # [{file, volume, delay, fade_in, fade_out}]
    sample_rate: int = 44100


@app.get("/v1/background/stream")
async def background_stream(
    session_id: str,
    tracks_json: str,  # JSON-encoded tracks array
    sample_rate: int = 44100,
    duration: int = 600,  # 10 minutes default - loops anyway, smaller file = faster load
    character: str = None  # Character name - if same character already playing, return existing stream
):
    """
    Generate mixed background audio as MP3 file (GET for direct audio element src).
    Returns a finite-duration file that client can loop.
    Uses MP3 format for much smaller file sizes (~30MB vs ~1GB for 10 min at 320kbps vs WAV).
    
    If `character` is provided and that character already has an active stream,
    returns the existing cached file instead of creating a new one.
    """
    import json
    import hashlib
    
    try:
        tracks = json.loads(tracks_json)
    except:
        raise HTTPException(status_code=400, detail="Invalid tracks_json")
    
    if not tracks:
        raise HTTPException(status_code=400, detail="No tracks provided")
    
    # Create a hash of the tracks for comparison
    tracks_hash = hashlib.md5(tracks_json.encode()).hexdigest()[:16]
    
    # Check if same character already has an active stream with same tracks
    if character:
        with _character_stream_lock:
            if character in _active_character_streams:
                existing = _active_character_streams[character]
                if existing.get("tracks_hash") == tracks_hash and os.path.exists(existing.get("cache_path", "")):
                    log_info(f"[BGStream] Same character '{character}' already streaming, reusing existing stream")
                    # Update session_id mapping for the new session
                    old_session = existing.get("session_id")
                    with _bg_stream_lock:
                        if old_session in _bg_stream_processes:
                            # Transfer the cache path to new session
                            _bg_stream_processes[session_id] = _bg_stream_processes.pop(old_session)
                    existing["session_id"] = session_id
                    
                    return FileResponse(
                        existing["cache_path"],
                        media_type="audio/wav",
                        filename=f"background_{session_id}.wav",
                        headers={
                            "X-Session-ID": session_id,
                            "X-Cached": "true",
                            "X-Same-Character": "true",
                            "Cache-Control": "public, max-age=31536000",
                        },
                    )
    
    # Validate tracks
    valid_tracks = []
    for track in tracks:
        bg_file = track.get("file", "")
        vol = float(track.get("volume", 0.3))
        delay = float(track.get("delay", 0))
        fade_in = float(track.get("fade_in", 0))
        fade_out = float(track.get("fade_out", 0))
        
        if bg_file and os.path.exists(bg_file) and vol > 0:
            valid_tracks.append({
                "file": bg_file,
                "volume": vol,
                "delay": delay,
                "fade_in": fade_in,
                "fade_out": fade_out,
            })
    
    if not valid_tracks:
        raise HTTPException(status_code=400, detail="No valid tracks found")
    
    # Check cache first
    cache_path = _get_background_stream_cache_path(valid_tracks, duration, sample_rate)
    if os.path.exists(cache_path):
        log_info(f"[BGStream] Using cached background mix: {os.path.basename(cache_path)} ({duration}s)")
        
        # Store path for cleanup tracking
        with _bg_stream_lock:
            _bg_stream_processes[session_id] = cache_path
        
        # Track character stream if character provided
        if character:
            with _character_stream_lock:
                _active_character_streams[character] = {
                    "session_id": session_id,
                    "cache_path": cache_path,
                    "tracks_hash": tracks_hash,
                }
                log_info(f"[BGStream] Tracking background for character '{character}'")
        
        return FileResponse(
            cache_path,
            media_type="audio/wav",
            filename=f"background_{session_id}.wav",
            headers={
                "X-Session-ID": session_id,
                "X-Cached": "true",
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
            },
        )
    
    log_info(f"[BGStream] Generating {duration}s background mix for session {session_id} with {len(valid_tracks)} tracks (not cached)")
    
    # Build FFmpeg command to mix all tracks for specified duration
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    
    # Add all input files with infinite loop
    for track in valid_tracks:
        args += ["-stream_loop", "-1", "-i", track["file"]]
    
    # Build filter graph
    filter_parts = []
    mix_inputs = []
    
    for i, track in enumerate(valid_tracks):
        vol_db = 20 * math.log10(max(0.001, track["volume"]))
        chain = f"[{i}:a]"
        
        # Apply delay if needed
        if track["delay"] > 0:
            chain += f"adelay={int(track['delay']*1000)}|{int(track['delay']*1000)},"
        
        # Apply fade in
        if track["fade_in"] > 0:
            chain += f"afade=t=in:st=0:d={track['fade_in']:.2f},"
        
        # Apply volume
        chain += f"volume={vol_db:.1f}dB[a{i}]"
        filter_parts.append(chain)
        mix_inputs.append(f"[a{i}]")
    
    # Mix all tracks
    if len(mix_inputs) > 1:
        filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[out]")
        filter_str = ";".join(filter_parts)
        args += ["-filter_complex", filter_str, "-map", "[out]"]
    else:
        filter_str = filter_parts[0].replace(f"[a0]", "")
        args += ["-af", filter_str]
    
    # Output to cache path (persistent, not temp)
    # Use high quality WAV (32-bit float)
    args += [
        "-t", str(duration),  # Limit duration
        "-c:a", "pcm_f32le",  # 32-bit float WAV
        "-ar", str(sample_rate),
        "-ac", "2",
        cache_path
    ]
    
    try:
        # Run FFmpeg in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        
        def generate_bg_mix():
            result = subprocess.run(args, capture_output=True, text=True, timeout=300)  # 5 min timeout for long mixes
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:200]}")
            return cache_path
        
        # Generate asynchronously
        output_path = await loop.run_in_executor(None, generate_bg_mix)
        
        log_info(f"[BGStream] Generated and cached background mix: {os.path.basename(output_path)}")
        
        # Store path for cleanup tracking
        with _bg_stream_lock:
            _bg_stream_processes[session_id] = output_path
        
        # Track character stream if character provided
        if character:
            with _character_stream_lock:
                _active_character_streams[character] = {
                    "session_id": session_id,
                    "cache_path": output_path,
                    "tracks_hash": tracks_hash,
                }
                log_info(f"[BGStream] Tracking background for character '{character}'")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"background_{session_id}.wav",
            headers={
                "X-Session-ID": session_id,
                "X-Cached": "false",
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
            },
        )
        
    except subprocess.TimeoutExpired:
        if os.path.exists(cache_path):
            _safe_unlink(cache_path)
        raise HTTPException(status_code=500, detail="Background mix generation timed out (took >5 minutes)")
    except Exception as e:
        if os.path.exists(cache_path):
            _safe_unlink(cache_path)
        log_error(f"[BGStream] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/background/stop-stream")
async def stop_background_stream(session_id: str = Form(...), character: str = Form(None)):
    """Stop/cleanup a background audio session.
    
    If character is provided, also clears the character tracking.
    """
    # Clear character tracking if character provided
    if character:
        with _character_stream_lock:
            if character in _active_character_streams:
                del _active_character_streams[character]
                log_info(f"[BGStream] Cleared background tracking for character '{character}'")
    else:
        # Find and clear character tracking by session_id
        with _character_stream_lock:
            to_remove = None
            for char_name, info in _active_character_streams.items():
                if info.get("session_id") == session_id:
                    to_remove = char_name
                    break
            if to_remove:
                del _active_character_streams[to_remove]
                log_info(f"[BGStream] Cleared background tracking for character '{to_remove}' (by session)")
    
    with _bg_stream_lock:
        if session_id in _bg_stream_processes:
            item = _bg_stream_processes.pop(session_id)
            # Item could be a file path or process handle
            if isinstance(item, str):
                # Only delete if it's a temp file, not a cached file
                # Cached files are in OUTPUT_DIR/background_streams/ and should persist
                cache_dir = str(Path(OUTPUT_DIR) / "background_streams")
                if cache_dir in item:
                    # It's a cached file, keep it for future use
                    log_info(f"[BGStream] Keeping cached file for session {session_id}: {os.path.basename(item)}")
                else:
                    # It's a temp file, safe to delete
                    _safe_unlink(item)
            else:
                try:
                    item.terminate()
                    item.wait(timeout=1)
                except:
                    try:
                        item.kill()
                    except:
                        pass
            log_info(f"[BGStream] Stopped session {session_id}")
            return {"status": "ok", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# =========================================================
# Preprocess (UVR5)
# =========================================================

def _get_cleaned_vocals_cache_path(original_filename: str, file_size: int) -> str:
    """Get cache path for cleaned vocals based on original filename."""
    import re
    
    cache_dir = Path(OUTPUT_DIR) / "cleaned_vocals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    base_name = os.path.splitext(original_filename)[0]
    base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
    base_name = base_name[:100]
    
    # Include file size in cache key to detect different files with same name
    cache_filename = f"{base_name}-cleaned.wav"
    
    return str(cache_dir / cache_filename)


@app.post("/v1/preprocess/uvr5/clean-vocals")
async def clean_vocals(
    audio: UploadFile = File(...),
    aggression: int = Form(10),
    device: Optional[str] = Form(None),
    skip_if_cached: bool = Form(True),
    model_key: str = Form("hp5_vocals"),
):
    """
    Process audio using UVR5 models.
    Returns processed WAV.
    
    Supported models:
    - hp5_vocals: Isolate vocals from music/background
    - deecho_normal: Standard echo removal
    - deecho_aggressive: Aggressive echo/delay removal
    - deecho_dereverb: Remove echo and reverb together
    
    If skip_if_cached=True and a cached version exists, returns that instead.
    """
    original_filename = audio.filename or "audio.wav"
    
    # Read content to get file size for cache key
    content = await audio.read()
    file_size = len(content)
    
    # Check cache first (include model_key in cache path)
    cache_path = _get_cleaned_vocals_cache_path(f"{model_key}_{original_filename}", file_size)
    if skip_if_cached and os.path.exists(cache_path):
        log_info(f"[UVR5:{model_key}] Using cached output: {os.path.basename(cache_path)}")
        suffix = "vocals" if model_key == "hp5_vocals" else "cleaned"
        return FileResponse(
            cache_path,
            media_type="audio/wav",
            filename=f"{os.path.splitext(original_filename)[0]}-{suffix}.wav",
        )
    
    log_info(f"[UVR5:{model_key}] Processing: {original_filename}")
    
    ext = os.path.splitext(original_filename)[1].lower() or ".bin"
    fd_in, tmp_in = tempfile.mkstemp(suffix=ext)
    os.close(fd_in)

    fd_wav, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd_wav)

    out_dir = tempfile.mkdtemp(prefix="voiceforge_uvr5_")

    try:
        # Write content to temp file
        with open(tmp_in, "wb") as f:
            f.write(content)

        if not is_wav_file(tmp_in):
            convert_to_wav(tmp_in, tmp_wav, sample_rate=44100, channels=1)
            input_for_uvr5 = tmp_wav
        else:
            shutil.copy2(tmp_in, tmp_wav)
            input_for_uvr5 = tmp_wav

        if device is None:
            device = "cuda"

        output_path = uvr5_models.separate_vocals(
            audio_path=input_for_uvr5,
            output_dir=out_dir,
            device=device,
            aggression=max(0, min(20, int(aggression))),
            model_key=model_key,
        )

        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="UVR5 produced no output file")

        # Save to cache
        try:
            shutil.copy2(output_path, cache_path)
            log_info(f"[UVR5:{model_key}] Cached output: {os.path.basename(cache_path)}")
        except Exception as e:
            log_error(f"[UVR5:{model_key}] Failed to cache output: {e}")

        # Return the cached file (persistent) instead of temp file
        _cleanup_many([tmp_in, tmp_wav, out_dir])
        
        suffix = "vocals" if model_key == "hp5_vocals" else "cleaned"
        return FileResponse(
            cache_path,
            media_type="audio/wav",
            filename=f"{os.path.splitext(original_filename)[0]}-{suffix}.wav",
        )
    except HTTPException:
        _cleanup_many([tmp_in, tmp_wav, out_dir])
        raise
    except Exception as e:
        log_error(f"UVR5 clean vocals failed: {e}")
        _cleanup_many([tmp_in, tmp_wav, out_dir])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/preprocess/uvr5/unload")
async def unload_uvr5():
    """Unload UVR5 models to free GPU memory."""
    try:
        uvr5_models.unload()
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# =========================================================
# Postprocess (FFmpeg-based)
# =========================================================

def build_ffmpeg_base_cmd(
    input_path: str,
    output_path: str,
    filters: Optional[str] = None,
    filter_complex: Optional[str] = None,
    output_channels: Optional[int] = None,
    map_output: Optional[str] = None,
    bit_depth: str = "pcm_f32le",  # Use 32-bit float for lossless precision
) -> List[str]:
    """
    Build FFmpeg command for audio processing.
    
    bit_depth options:
      - "pcm_f32le": 32-bit float (lossless, maximum precision for processing)
      - "pcm_s24le": 24-bit signed (lossless, good for final output)
      - "pcm_s16le": 16-bit signed (smaller files, slight quality loss)
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
    ]

    if filter_complex:
        cmd.extend(["-filter_complex", filter_complex])
        if map_output:
            cmd.extend(["-map", map_output])
    elif filters:
        cmd.extend(["-af", filters])

    cmd.extend(["-c:a", bit_depth])

    if output_channels:
        cmd.extend(["-ac", str(output_channels)])

    cmd.append(output_path)
    return cmd


def _run_ffmpeg_filter_bytes(
    input_wav_bytes: bytes,
    filters: Optional[str] = None,
    output_channels: Optional[int] = None,
) -> bytes:
    """Run ffmpeg filter chain on WAV bytes and return WAV bytes."""
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
    ]

    if filters:
        cmd.extend(["-af", filters])

    cmd.extend(["-c:a", "pcm_f32le"])
    if output_channels:
        cmd.extend(["-ac", str(output_channels)])

    cmd.extend(["-f", "wav", "pipe:1"])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(input=input_wav_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg bytes filter failed: {err.decode('utf-8', errors='ignore')}")
    return out


def post_process_voice_bytes(main_wav_bytes: bytes, post_params: Dict[str, Any]) -> bytes:
    """In-memory variant of post_process_voice for streaming chunks."""
    t_start = time.perf_counter()

    # ===== Read and analyze input =====
    with sf.SoundFile(io.BytesIO(main_wav_bytes)) as f:
        input_channels = f.channels
        sample_rate = f.samplerate

    # Keep existing post FX assumptions (44100 processing path)
    # If chunk is not 44.1k, resample once up front in-memory.
    current_bytes = main_wav_bytes
    if sample_rate != 44100:
        data, _ = sf.read(io.BytesIO(main_wav_bytes), dtype='float32')
        import soxr
        data = soxr.resample(data, sample_rate, 44100, quality='VHQ')
        buf = io.BytesIO()
        sf.write(buf, data.astype(np.float32), 44100, format='WAV')
        current_bytes = buf.getvalue()

    # ===== Existing effect flag logic (mirrors file-based version) =====
    highpass = float(post_params.get("highpass", 0) or 0)
    lowpass = float(post_params.get("lowpass", 0) or 0)
    bass_freq = float(post_params.get("bass_freq", 200) or 200)
    bass_gain = float(post_params.get("bass_gain", 0) or 0)
    treble_freq = float(post_params.get("treble_freq", 4000) or 4000)
    treble_gain = float(post_params.get("treble_gain", 0) or 0)
    reverb_delay = float(post_params.get("reverb_delay", 0) or 0)
    reverb_decay = float(post_params.get("reverb_decay", 0) or 0)
    crystalizer = float(post_params.get("crystalizer", 0) or 0)
    deesser = float(post_params.get("deesser", 0) or 0)

    audio_8d_enabled = bool(post_params.get("audio_8d_enabled", False))
    asmr_enabled = bool(post_params.get("asmr_enabled", False))
    asmr_tingles = float(post_params.get("asmr_tingles", 50) or 50)
    asmr_breathiness = float(post_params.get("asmr_breathiness", 50) or 50)
    asmr_crispness = float(post_params.get("asmr_crispness", 50) or 50)

    pitch_shift_enabled = bool(post_params.get("pitch_shift_enabled", False))
    pitch_shift_semitones = float(post_params.get("pitch_shift_semitones", 0) or 0)

    _log_effect_interactions(post_params, asmr_enabled, audio_8d_enabled)

    filters = []
    needs_stereo = input_channels == 1 and (audio_8d_enabled or asmr_enabled)

    if highpass > 0:
        filters.append(f"highpass=f={highpass:.1f}")
    if lowpass > 0:
        filters.append(f"lowpass=f={lowpass:.1f}")
    if abs(bass_gain) > 0.01:
        filters.append(f"equalizer=f={bass_freq:.1f}:width_type=o:width=1:g={bass_gain:.2f}")
    if abs(treble_gain) > 0.01:
        filters.append(f"equalizer=f={treble_freq:.1f}:width_type=o:width=1:g={treble_gain:.2f}")

    if asmr_enabled:
        # Reuse same intent as existing path
        if asmr_tingles > 0:
            gain = (asmr_tingles / 100.0) * 6.0
            filters.append(f"equalizer=f=8000:width_type=o:width=1:g={gain:.2f}")
        if asmr_breathiness > 0:
            gain = (asmr_breathiness / 100.0) * 4.0
            filters.append(f"equalizer=f=12000:width_type=o:width=1:g={gain:.2f}")
        if asmr_crispness > 0:
            gain = (asmr_crispness / 100.0) * 4.0
            filters.append(f"equalizer=f=5000:width_type=o:width=1:g={gain:.2f}")

    if reverb_delay > 0 and reverb_decay > 0:
        d = max(10.0, reverb_delay)
        dec = max(0.1, min(0.99, reverb_decay / 100.0 if reverb_decay > 1 else reverb_decay))
        filters.append(f"aecho=0.8:0.88:{d:.0f}:{dec:.3f}")

    if crystalizer > 0:
        amt = crystalizer / 100.0
        filters.append(f"acompressor=threshold={-24 + 8*amt:.1f}dB:ratio={2 + 2*amt:.2f}:attack=5:release=50")
    if deesser > 0:
        amt = deesser / 100.0
        filters.append(f"deesser=i={max(0.1, amt):.3f}")

    use_python_spatial = False
    spatial_params = {}
    avg_distance = 0.5
    if audio_8d_enabled:
        mode = post_params.get("audio_8d_mode", "rotate")
        speed = post_params.get("audio_8d_speed", 0.1)
        pan_arc_degrees = post_params.get("audio_8d_depth", 180.0)
        if pan_arc_degrees <= 2.0:
            pan_arc_degrees = pan_arc_degrees * 180.0
        half_arc = pan_arc_degrees / 2.0
        avg_distance = post_params.get("audio_8d_distance", 0.3)
        volume_db = 2.0 - 4.0 * avg_distance
        if abs(volume_db) > 0.3:
            filters.append(f"volume={volume_db:.1f}dB")

        use_python_spatial = True
        if mode == "center":
            spatial_params = {"mode": "static", "start_angle": 0.0, "head_shadow": True, "head_shadow_intensity": 0.4}
        elif mode == "static":
            spatial_params = {"mode": "static", "start_angle": -90.0, "head_shadow": True, "head_shadow_intensity": 0.5}
        elif mode == "static_right":
            spatial_params = {"mode": "static", "start_angle": 90.0, "head_shadow": True, "head_shadow_intensity": 0.5}
        else:
            spatial_params = {
                "mode": mode,
                "speed_hz": speed,
                "start_angle": -half_arc,
                "end_angle": half_arc,
                "head_shadow": True,
                "head_shadow_intensity": 0.5 * (1.0 - avg_distance),
            }

    if asmr_enabled:
        filters.append("alimiter=limit=0.95:attack=5:release=50")

    filter_str = ",".join(filters) if filters else "anull"
    current_bytes = _run_ffmpeg_filter_bytes(current_bytes, filters=filter_str, output_channels=(2 if needs_stereo else None))

    if pitch_shift_enabled and abs(pitch_shift_semitones) > 0.001:
        pitch_factor = 2 ** (pitch_shift_semitones / 12.0)
        pitch_filter = f"asetrate=44100*{pitch_factor:.6f},aresample=44100,atempo=1"
        current_bytes = _run_ffmpeg_filter_bytes(current_bytes, filters=pitch_filter, output_channels=None)

    if use_python_spatial and spatial_params:
        audio_data, sample_rate = sf.read(io.BytesIO(current_bytes))
        stereo_output = process_spatial_audio_buffer(
            audio_data,
            sample_rate,
            mode=spatial_params.get("mode", "sweep"),
            speed_hz=spatial_params.get("speed_hz", 0.1),
            start_angle=spatial_params.get("start_angle", -90.0),
            end_angle=spatial_params.get("end_angle", 90.0),
            head_shadow=spatial_params.get("head_shadow", True),
            head_shadow_intensity=spatial_params.get("head_shadow_intensity", 0.4),
            quality=post_params.get("audio_8d_quality", "balanced"),
            distance=avg_distance,
            itd_enabled=post_params.get("audio_8d_itd", None),
            proximity_enabled=post_params.get("audio_8d_proximity", None),
            crossfeed_enabled=post_params.get("audio_8d_crossfeed", None),
            micro_movements=post_params.get("audio_8d_micro_movements", None),
            speech_aware=post_params.get("audio_8d_speech_aware", True),
            time_offset=post_params.get("spatial_time_offset", 0.0),
        )
        buf = io.BytesIO()
        sf.write(buf, stereo_output, sample_rate, format='WAV')
        current_bytes = buf.getvalue()

    t_end = time.perf_counter()
    log_info(f"[POST-BYTES] Total: {(t_end - t_start)*1000:.0f}ms")
    return current_bytes


def _log_effect_interactions(post_params: Dict[str, Any], asmr_enabled: bool, audio_8d_enabled: bool) -> None:
    """
    Log warnings about potentially problematic effect combinations.
    
    After consolidation, the main concerns are:
    - Multiple bass boosts (bass_gain + spatial proximity)
    - Multiple high boosts (ASMR tingles/crispness/breathiness + treble)
    
    Note: Spatial effects are now unified - no more ASMR vs 8D clashes!
    Note: ASMR warmth consolidated into Spatial Proximity effect.
    """
    warnings = []
    
    # Count bass boost sources (bass_gain + spatial proximity)
    bass_sources = []
    if post_params.get("bass_gain", 0) > 3:
        bass_sources.append("Bass Gain")
    # Spatial proximity (applies when spatial audio is enabled)
    avg_dist = post_params.get("audio_8d_distance", 0.5)
    if avg_dist < 0.4 and post_params.get("audio_8d_proximity", True) is not False:
        if audio_8d_enabled:
            bass_sources.append("Spatial Proximity")
    
    if len(bass_sources) >= 2:
        warnings.append(f"🟡 Multiple bass boosts: {', '.join(bass_sources)} - reduce one to avoid muddiness")
    
    # Count high boost sources
    high_sources = []
    if asmr_enabled:
        if post_params.get("asmr_tingles", 0) > 50:
            high_sources.append("Tingles")
        if post_params.get("asmr_crispness", 0) > 50:
            high_sources.append("Crispness")
        if post_params.get("asmr_breathiness", 0) > 50:
            high_sources.append("Breathiness")
    if post_params.get("treble_gain", 0) > 3:
        high_sources.append("Treble")
    
    if len(high_sources) >= 3:
        warnings.append(f"🟡 Multiple high boosts: {', '.join(high_sources)} - may sound harsh")
    
    # Log warnings
    for warning in warnings:
        log_warn(f"[POST] {warning}")
    
    # Log good combinations
    if asmr_enabled and audio_8d_enabled:
        log_info(f"[POST] 🟢 ASMR tonal + spatial audio = optimal quality")


def post_process_voice(main_wav_path: str, post_params: Dict[str, Any]) -> str:
    """
    Post-process audio with all effects in optimal order.
    
    PROCESSING PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 1-6: FFmpeg Pass (tonal processing)                                  │
    │   • EQ (highpass, lowpass, bass, treble)                                   │
    │   • ASMR Tonal (tingles, crispness, breathiness, compression)              │
    │   • Enhancement (crystalizer, deesser)                                     │
    │   • Reverb                                                                  │
    │   • Limiter (safety, when ASMR enabled)                                    │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ STAGE 7: Python Spatial (unified binaural processing)                      │
    │   • ITD (interaural time difference - sub-sample accurate)                 │
    │   • Head Shadow (multi-band HRTF-like filtering)                           │
    │   • Proximity Effect (bass + presence boost for closeness)                 │
    │   • Crossfeed (natural inter-ear bleed)                                    │
    │   • Micro-movements (organic variation - prevents robotic feel)            │
    │   • Dynamic Panning (sweep/rotate/extreme modes)                           │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ STAGE 8: FFmpeg Pass (pitch shift, if enabled)                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    SPATIAL TRIGGER CONDITIONS:
    • audio_8d_enabled=true → Dynamic or static positioning with full spatial effects
    • asmr_enabled=true (no 8D) → Tonal shaping only (no spatial unless 8D also enabled)
    
    Note: All spatial processing now uses Python spatial_audio.py for maximum quality.
    """
    import time
    t_start = time.perf_counter()
    
    filters = []
    
    # Check what's enabled
    pitch_shift_enabled = post_params.get("pitch_shift_enabled", False) and post_params.get("pitch_shift_semitones", 0) != 0
    asmr_enabled = post_params.get("asmr_enabled", False)
    audio_8d_enabled = post_params.get("audio_8d_enabled", False)
    
    # Determine if we need stereo output (spatial processing always needs stereo)
    needs_stereo = audio_8d_enabled or asmr_enabled

    if needs_stereo:
        # Properly convert mono to stereo by duplicating channel 0 to both L/R
        # This is the starting point for stereo effects (8D, ASMR binaural, etc.) to manipulate
        filters.append("pan=stereo|c0=c0|c1=c0")
    
    # ===== STAGE 1: EQ (Subtractive first) =====
    # Highpass: Remove rumble/noise below speech frequencies
    # SAFEGUARD: Don't cut above 80Hz to preserve voice body and clarity
    highpass = post_params.get('highpass', 0.0)
    if highpass > 0:
        # Clamp to safe range: 20-80Hz (voice fundamentals start ~85Hz)
        safe_highpass = max(20.0, min(80.0, highpass))
        if safe_highpass != highpass:
            log_warn(f"[POST] Highpass clamped from {highpass:.1f}Hz to {safe_highpass:.1f}Hz to preserve voice clarity")
        filters.append(f"highpass=f={safe_highpass:.1f}")
    
    # Lowpass: Remove hiss/noise above speech frequencies
    # SAFEGUARD: Don't cut below 12kHz to preserve clarity and sibilance
    lowpass = post_params.get('lowpass', 0.0)
    if lowpass > 0:
        # Clamp to safe range: 8kHz-20kHz (preserve clarity up to 12kHz minimum)
        safe_lowpass = max(8000.0, min(20000.0, lowpass))
        if safe_lowpass != lowpass:
            log_warn(f"[POST] Lowpass clamped from {lowpass:.1f}Hz to {safe_lowpass:.1f}Hz to preserve voice clarity")
        filters.append(f"lowpass=f={safe_lowpass:.1f}")
    
    # ===== ASMR ENHANCEMENT SYSTEM =====
    # Comprehensive processing for maximum tingle-inducing quality
    if asmr_enabled:
        # Get ASMR parameters (0-100 scale, convert to 0-1)
        tingles = post_params.get("asmr_tingles", 60) / 100.0
        breathiness = post_params.get("asmr_breathiness", 65) / 100.0
        crispness = post_params.get("asmr_crispness", 55) / 100.0
        
        # Derive intimacy from Distance slider: close (0) = max intimacy, far (1) = no intimacy
        distance = post_params.get("audio_8d_distance", 0.3)
        intimacy = 1.0 - distance  # 0=far, 1=in ear
        
        log_info(f"[ASMR] Intimacy={intimacy:.0%} (from dist={distance}) Tingles={tingles:.0%} Breath={breathiness:.0%} Crisp={crispness:.0%}")
        
        # === INTIMACY: Compression for close, intimate feel ===
        # Light compression for "closeness" feel (gentler to avoid over-compression)
        # SAFEGUARD: Use gentler settings to preserve clarity and natural dynamics
        if intimacy > 0.1:
            # Softer compression - higher threshold, lower ratio for natural sound
            # Maximum knee width (8dB) for smoother, more transparent compression
            # NOTE: FFmpeg acompressor knee range is 1-8dB, so we use 8 (maximum)
            threshold = -24 + (1.0 - intimacy) * 10  # -24dB to -14dB
            ratio = 1.8 + intimacy * 1.4  # 1.8:1 to 3.2:1 (much gentler)
            attack = 30 - intimacy * 10  # 30ms to 20ms (slower attack preserves transients)
            release = 250 - intimacy * 50  # 250ms to 200ms (slower release preserves natural decay)
            makeup = intimacy * 1.5  # Up to 1.5dB makeup gain (reduced to avoid pumping)
            filters.append(f"acompressor=threshold={threshold:.0f}dB:ratio={ratio:.1f}:attack={attack:.0f}:release={release:.0f}:makeup={makeup:.1f}:knee=8")
        
        # NOTE: Warmth/bass is now handled by Spatial Proximity effect (uses Distance slider)
        
        # === TINGLE ZONE: 2-8kHz enhancement ===
        if tingles > 0.1:
            # Single presence band instead of multiple stacking EQs
            tingle_gain = tingles * 2.5  # Up to 2.5dB (was 5dB!)
            filters.append(f"equalizer=f=4000:width_type=o:width=1.5:g={tingle_gain:.1f}")
        
        # === CRISPNESS: High detail ===
        if crispness > 0.1:
            # Single high shelf instead of multiple bands
            crisp_gain = crispness * 2  # Up to 2dB (was 6dB!)
            filters.append(f"highshelf=f=8000:w=0.8:g={crisp_gain:.1f}")
        
        # === BREATHINESS: Airy quality ===
        if breathiness > 0.1:
            # Single air shelf (was multiple stacking highshelves!)
            breath_gain = breathiness * 3  # Up to 3dB (was 8dB!)
            filters.append(f"highshelf=f=12000:w=0.7:g={breath_gain:.1f}")
    
    # ===== STAGE 3: Additive EQ (non-ASMR) =====
    bass_gain = post_params.get('bass_gain', 0.0)
    if bass_gain != 0:
        bass_freq = post_params.get('bass_freq', 100.0)
        filters.append(f"equalizer=f={bass_freq}:width_type=o:width=2:g={bass_gain:.1f}")
    
    treble_gain = post_params.get('treble_gain', 0.0)
    if treble_gain != 0:
        treble_freq = post_params.get('treble_freq', 8000.0)
        filters.append(f"equalizer=f={treble_freq}:width_type=o:width=2:g={treble_gain:.1f}")
    
    # ===== STAGE 4: Enhancement =====
    # Crystalizer: Adds harmonics/excitement but can introduce distortion
    # SAFEGUARD: Limit intensity to preserve clarity
    crystal = post_params.get("crystalizer", 0.0)
    if crystal > 0:
        # Clamp to safe range: 0-3 (was unlimited, can go higher and cause distortion)
        safe_crystal = min(3.0, crystal)
        if safe_crystal != crystal:
            log_warn(f"[POST] Crystalizer clamped from {crystal:.1f} to {safe_crystal:.1f} to preserve clarity")
        filters.append(f"crystalizer=i={safe_crystal:.1f}")
    
    # Deesser: Reduces sibilance but can dull clarity if too aggressive
    # SAFEGUARD: Moderate intensity to preserve natural sibilance
    deess = post_params.get("deesser", 0.0)
    if deess > 0:
        # Clamp to safe range: 0-2 (was unlimited)
        safe_deess = min(2.0, deess)
        if safe_deess != deess:
            log_warn(f"[POST] Deesser clamped from {deess:.2f} to {safe_deess:.2f} to preserve clarity")
        filters.append(f"deesser=i={safe_deess:.2f}")
    
    # ===== STAGE 5: Reverb =====
    # Simplified: just delay + decay, gains are fixed to sensible values
    reverb_delay = post_params.get("reverb_delay", 0.0)
    reverb_decay = post_params.get("reverb_decay", 0.0)
    
    if reverb_decay > 0 and reverb_delay > 0:
        # Fixed gain values: 0.8 in, 0.9 out (preserves original while adding reverb)
        in_gain = 0.8
        out_gain = 0.9
        # Multi-tap echo for richer reverb
        delays = f"{reverb_delay:.1f}|{reverb_delay * 1.6:.1f}|{reverb_delay * 2.8:.1f}|{reverb_delay * 4.2:.1f}"
        decays = f"{reverb_decay:.2f}|{max(0.05, reverb_decay * 0.75):.2f}|{max(0.05, reverb_decay * 0.56):.2f}|{max(0.05, reverb_decay * 0.38):.2f}"
        filters.append(f"aecho={in_gain:.2f}:{out_gain:.2f}:{delays}:{decays}")
    
    # ===== STAGE 6: Stereo/Spatial =====
    # 
    # UNIFIED SPATIAL PROCESSING:
    # All spatial effects (ITD, proximity, head shadow, crossfeed) are handled by
    # Python spatial_audio.py for maximum quality. No more duplicate FFmpeg effects.
    #
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ ASMR Tonal (FFmpeg)          │ Spatial Positioning (Python)                │
    # ├─────────────────────────────────────────────────────────────────────────────┤
    # │ • Warmth (low shelf)         │ • ITD (sub-sample accurate timing)          │
    # │ • Tingles (presence boost)   │ • Head Shadow (multi-band HRTF-like)        │
    # │ • Crispness (high detail)    │ • Proximity Effect (bass + presence)        │
    # │ • Breathiness (air)          │ • Crossfeed (natural ear bleed)             │
    # │ • Intimacy (compression)     │ • Micro-movements (organic variation)       │
    # │                              │ • Dynamic panning (sweep/rotate/extreme)    │
    # └─────────────────────────────────────────────────────────────────────────────┘
    #
    # audio_8d_enabled=true → Full spatial processing (ITD, head shadow, proximity, panning)
    
    # Check for potential reverb stacking (the only remaining interaction concern)
    _log_effect_interactions(post_params, asmr_enabled, audio_8d_enabled)
    
    # UNIFIED SPATIAL AUDIO - All spatial processing through Python spatial_audio.py
    # This ensures consistent high-quality ITD, head shadow, proximity, and crossfeed
    use_python_spatial = False
    spatial_params = {}
    avg_distance = 0.5  # Default distance
    
    if audio_8d_enabled:
        # === UNIFIED SPATIAL AUDIO ===
        mode = post_params.get("audio_8d_mode", "rotate")
        speed = post_params.get("audio_8d_speed", 0.1)
        
        # Pan arc in degrees (20-360), convert to half-arc for ±range
        pan_arc_degrees = post_params.get("audio_8d_depth", 180.0)
        if pan_arc_degrees <= 2.0:  # Old 0-1 format
            pan_arc_degrees = pan_arc_degrees * 180.0
        half_arc = pan_arc_degrees / 2.0
        
        # Distance controls the proximity effect (bass + presence boost when close)
        # Uses single distance slider from UI (0 = touching ear, 1 = far)
        avg_distance = post_params.get("audio_8d_distance", 0.3)
        
        # Distance-based volume (closer = louder)
        volume_db = 2.0 - 4.0 * avg_distance
        if abs(volume_db) > 0.3:
            filters.append(f"volume={volume_db:.1f}dB")
        
        # ALL modes use Python spatial for consistency and quality
        use_python_spatial = True
        
        if mode == "center":
            # Center/Binaural: Stationary front position with full spatial effects
            spatial_params = {
                "mode": "static",
                "start_angle": 0.0,  # Front center
                "head_shadow": True,
                "head_shadow_intensity": 0.4,
            }
            log_info(f"[Spatial] Center (binaural): front position, distance={avg_distance:.2f}")
        elif mode == "static":
            spatial_params = {
                "mode": "static",
                "start_angle": -90.0,  # Full left
                "head_shadow": True,
                "head_shadow_intensity": 0.5,
            }
            log_info(f"[Spatial] Static left positioning")
        elif mode == "static_right":
            spatial_params = {
                "mode": "static",
                "start_angle": 90.0,  # Full right
                "head_shadow": True,
                "head_shadow_intensity": 0.5,
            }
            log_info(f"[Spatial] Static right positioning")
        else:
            # Dynamic modes (sweep/rotate/extreme)
            spatial_params = {
                "mode": mode,
                "speed_hz": speed,
                "start_angle": -half_arc,
                "end_angle": half_arc,
                "head_shadow": True,
                "head_shadow_intensity": 0.5 * (1.0 - avg_distance),
            }
            log_info(f"[Spatial] Dynamic: mode={mode}, speed={speed}Hz, arc={pan_arc_degrees:.0f}° (±{half_arc:.0f}°)")
    
    # ===== STAGE 7: Limiting (safety) =====
    # Add limiter when ASMR is enabled to prevent clipping from compression/EQ
    if asmr_enabled:
        filters.append("alimiter=limit=0.95:attack=5:release=50")
    
    # Build and run main filter chain
    filter_str = ",".join(filters) if filters else "anull"

    fd, tmp = tempfile.mkstemp(suffix="_post.wav")
    os.close(fd)
    output_channels = 2 if needs_stereo else None
    cmd = build_ffmpeg_base_cmd(main_wav_path, tmp, filters=filter_str, output_channels=output_channels)
    
    t_ffmpeg_start = time.perf_counter()
    subprocess.run(cmd, check=True)
    t_ffmpeg_end = time.perf_counter()
    log_info(f"[POST] FFmpeg main pass: {(t_ffmpeg_end - t_ffmpeg_start)*1000:.0f}ms, filters: {len(filters)}")
    
    current_file = tmp
    log_info(f"[POST] FFmpeg done, current_file exists: {os.path.exists(current_file)}, size: {os.path.getsize(current_file) if os.path.exists(current_file) else 0}")
    
    # ===== SEPARATE PASS: Pitch Shift (requires sample rate manipulation) =====
    if pitch_shift_enabled:
        semitones = post_params.get("pitch_shift_semitones", 0)
        fd_pitch, tmp_pitch = tempfile.mkstemp(suffix="_pitch.wav")
        os.close(fd_pitch)
        
        pitch_factor = 2 ** (semitones / 12.0)
        
        # Build pitch filter
        pitch_filters = [f"asetrate=44100*{pitch_factor:.6f}", "aresample=44100", "atempo=1"]
        pitch_filter = ",".join(pitch_filters)
        pitch_cmd = build_ffmpeg_base_cmd(current_file, tmp_pitch, filters=pitch_filter)
        
        t_pitch_start = time.perf_counter()
        subprocess.run(pitch_cmd, check=True)
        t_pitch_end = time.perf_counter()
        log_info(f"[POST] FFmpeg pitch pass: {(t_pitch_end - t_pitch_start)*1000:.0f}ms")
        
        try:
            os.unlink(current_file)
        except:
            pass
        current_file = tmp_pitch
    
    # ===== SEPARATE PASS: Python Spatial Audio (for immersive 8D/16D) =====
    if use_python_spatial and spatial_params:
        import numpy as np
        t_spatial_start = time.perf_counter()
        
        fd_spatial, tmp_spatial = tempfile.mkstemp(suffix="_spatial.wav")
        os.close(fd_spatial)
        
        try:
            # Read current audio
            log_info(f"[POST] Spatial: Reading audio file...")
            t_read = time.perf_counter()
            audio_data, sample_rate = sf.read(current_file)
            
            # Calculate file info for logging
            duration_sec = len(audio_data) / sample_rate
            file_size_mb = (audio_data.nbytes) / (1024 * 1024)
            log_info(f"[POST] Spatial: Loaded {file_size_mb:.1f}MB ({duration_sec:.1f}s @ {sample_rate}Hz) in {(time.perf_counter()-t_read)*1000:.0f}ms")
            
            # Get enhanced spatial parameters
            spatial_quality = post_params.get("audio_8d_quality", "balanced")  # fast/balanced/ultra
            spatial_distance = avg_distance if 'avg_distance' in dir() else 0.5
            
            # Warn about large files
            if duration_sec > 300:  # > 5 minutes
                log_warn(f"[POST] Spatial: Large file ({duration_sec/60:.1f} min) - processing may take a while...")
            
            # Get time offset for streaming continuity
            time_offset = post_params.get("spatial_time_offset", 0.0)
            if time_offset > 0:
                log_info(f"[POST] Spatial: Streaming mode, time_offset={time_offset:.2f}s")
            
            log_info(f"[POST] Spatial: Starting {spatial_quality} processing...")
            t_process = time.perf_counter()
            
            # Apply Python spatial audio processing with enhanced features
            # time_offset is for streaming: maintains panning phase across chunks
            stereo_output = process_spatial_audio_buffer(
                audio_data,
                sample_rate,
                mode=spatial_params.get("mode", "sweep"),
                speed_hz=spatial_params.get("speed_hz", 0.1),
                start_angle=spatial_params.get("start_angle", -90.0),
                end_angle=spatial_params.get("end_angle", 90.0),
                head_shadow=spatial_params.get("head_shadow", True),
                head_shadow_intensity=spatial_params.get("head_shadow_intensity", 0.4),
                # Enhanced parameters for ultra-realistic spatial audio
                quality=spatial_quality,
                distance=spatial_distance,
                # Let quality preset handle these, but allow explicit override
                itd_enabled=post_params.get("audio_8d_itd", None),
                proximity_enabled=post_params.get("audio_8d_proximity", None),
                crossfeed_enabled=post_params.get("audio_8d_crossfeed", None),
                micro_movements=post_params.get("audio_8d_micro_movements", None),
                speech_aware=post_params.get("audio_8d_speech_aware", True),
                time_offset=time_offset,  # For streaming continuity
            )
            
            log_info(f"[POST] Spatial: Processing done in {(time.perf_counter()-t_process)*1000:.0f}ms")
            
            # Write output
            log_info(f"[POST] Spatial: Writing output...")
            t_write = time.perf_counter()
            sf.write(tmp_spatial, stereo_output, sample_rate)
            log_info(f"[POST] Spatial: Write done in {(time.perf_counter()-t_write)*1000:.0f}ms")
            
            # Clean up previous temp file
            try:
                os.unlink(current_file)
            except:
                pass
            current_file = tmp_spatial
            
            t_spatial_end = time.perf_counter()
            log_info(f"[POST] Python spatial audio ({spatial_quality}): {(t_spatial_end - t_spatial_start)*1000:.0f}ms total")
        except Exception as e:
            import traceback
            log_error(f"[POST] Python spatial audio failed: {e}")
            log_error(f"[POST] Traceback: {traceback.format_exc()}")
            _safe_unlink(tmp_spatial)
    
    t_end = time.perf_counter()
    log_info(f"[POST] Total: {(t_end - t_start)*1000:.0f}ms")
    log_info(f"[POST] Returning file: {current_file}, exists: {os.path.exists(current_file)}, size: {os.path.getsize(current_file) if os.path.exists(current_file) else 0}")
    return current_file


def blend_with_background(
    main_wav_path: str,
    bg_files: List[str],
    bg_volumes: List[float],
    main_volume: float = 1.0,
    bg_delays: List[float] = None,
    bg_fade_ins: List[float] = None,
    bg_fade_outs: List[float] = None,
) -> str:
    """
    Blend main audio with background tracks using proper audio mixing.
    Uses overlay mixing instead of amix to preserve quality.
    Supports fade in/out for each background track.
    """
    # Get main audio info
    main_sample_rate = 44100
    main_len_s = 0.0
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", main_wav_path
        ], capture_output=True, text=True, check=True)
        lines = probe.stdout.strip().split("\n")
        if lines:
            main_len_s = float(lines[0])
        if len(lines) > 1:
            main_sample_rate = int(float(lines[1]))
    except Exception:
        try:
            with sf.SoundFile(main_wav_path) as f:
                main_len_s = len(f) / f.samplerate
                main_sample_rate = f.samplerate
        except:
            pass

    if main_len_s <= 0:
        fd, tmp = tempfile.mkstemp(suffix="_final.wav")
        os.close(fd)
        shutil.copyfile(main_wav_path, tmp)
        return tmp

    if bg_delays is None:
        bg_delays = [0.0] * len(bg_files)
    if bg_fade_ins is None:
        bg_fade_ins = [0.0] * len(bg_files)
    if bg_fade_outs is None:
        bg_fade_outs = [0.0] * len(bg_files)

    # Build ffmpeg command
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", main_wav_path]
    
    # Collect valid background tracks
    valid_bg = []
    for i, bg_path in enumerate(bg_files):
        try:
            vol = float(bg_volumes[i]) if i < len(bg_volumes) else 0.0
        except:
            continue
        if vol <= 0 or not os.path.exists(bg_path):
            continue
        delay = float(bg_delays[i]) if i < len(bg_delays) else 0.0
        fade_in = float(bg_fade_ins[i]) if i < len(bg_fade_ins) else 0.0
        fade_out = float(bg_fade_outs[i]) if i < len(bg_fade_outs) else 0.0
        valid_bg.append((bg_path, vol, max(0, delay), max(0, fade_in), max(0, fade_out)))
        args += ["-stream_loop", "-1", "-i", bg_path]

    if not valid_bg:
        # No background, just copy with volume adjustment if needed
        if main_volume == 1.0:
            fd, tmp = tempfile.mkstemp(suffix="_final.wav")
            os.close(fd)
            shutil.copyfile(main_wav_path, tmp)
            return tmp
        else:
            fd, tmp = tempfile.mkstemp(suffix="_final.wav")
            os.close(fd)
            vol_db = 20 * math.log10(max(0.0001, main_volume))
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", main_wav_path, "-af", f"volume={vol_db}dB",
                "-c:a", "pcm_f32le", tmp
            ], check=True)
            return tmp

    # =========================================================================
    # EXACT same pattern as working /v1/background/mix-chunk endpoint
    # =========================================================================
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", main_wav_path]
    
    # Add background tracks with -t to limit duration (like mix-chunk uses)
    for bg_path, vol, delay, fade_in, fade_out in valid_bg:
        # Use -t to limit bg duration to main length (no infinite loop needed)
        args += ["-t", str(main_len_s), "-i", bg_path]
    
    # Build filter - EXACT same pattern as mix-chunk
    filter_inputs = []
    mix_inputs = []
    
    # Main track
    if main_volume != 1.0:
        vol_db = 20 * math.log10(max(0.001, main_volume))
        filter_inputs.append(f"[0:a]volume={vol_db:.1f}dB[main]")
    else:
        filter_inputs.append("[0:a]anull[main]")
    mix_inputs.append("[main]")
    
    # Background tracks
    for i, (bg_path, vol, delay, fade_in, fade_out) in enumerate(valid_bg):
        input_idx = i + 1
        vol_db = 20 * math.log10(max(0.001, vol))
        chain = f"[{input_idx}:a]"
        
        # Apply delay if needed
        if delay > 0:
            chain += f"adelay={int(delay*1000)}|{int(delay*1000)},"
        
        # Apply fade in
        if fade_in > 0:
            chain += f"afade=t=in:st=0:d={fade_in:.2f},"
        
        # Apply fade out
        if fade_out > 0:
            fade_out_start = max(0, main_len_s - fade_out)
            chain += f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f},"
        
        # Apply volume
        chain += f"volume={vol_db:.1f}dB[bg{i}]"
        filter_inputs.append(chain)
        mix_inputs.append(f"[bg{i}]")
    
    # Mix all inputs - amix with normalize=0, duration=first to match main audio length
    filter_str = ";".join(filter_inputs) + ";" + "".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:duration=first:normalize=0[out]"
    
    fd, tmp = tempfile.mkstemp(suffix="_final.wav")
    os.close(fd)
    
    # Output args - EXACT same as mix-chunk
    args += ["-filter_complex", filter_str, "-map", "[out]", "-c:a", "pcm_f32le", tmp]
    
    subprocess.run(args, check=True)
    return tmp


def save_output(src_path: str, text: str) -> str:
    ensure_dir(OUTPUT_DIR)
    words = re.findall(r"\w+", text)[:4]
    base_name = "_".join(words).lower() if words else "output"
    output_path = os.path.join(OUTPUT_DIR, base_name + ".wav")
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{counter}.wav")
        counter += 1
    try:
        shutil.copyfile(src_path, output_path)
        log_info(f"Saved: {output_path}")
    except Exception as e:
        log_error(f"Error saving to {output_path}: {e}")
        return src_path
    return output_path


@app.post("/v1/resample")
async def api_resample(
    audio: UploadFile = File(...),
    sample_rate: int = Form(44100),
    volume: float = Form(1.0),
):
    """
    Resample audio to target sample rate and optionally adjust volume.
    
    Combined operation is faster than separate calls.
    Uses SoXr for high-quality resampling.
    
    Args:
        audio: Input audio file
        sample_rate: Target sample rate (default 44100)
        volume: Volume multiplier (1.0 = no change, 0.5 = 50%, 2.0 = 200%)
    
    Returns:
        Resampled WAV file
    """
    import soxr
    
    fd, tmp_input = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    try:
        content = await audio.read()
        with open(tmp_input, "wb") as f:
            f.write(content)
        
        # Read audio
        data, current_sr = sf.read(tmp_input, dtype='float32')
        
        needs_resample = current_sr != sample_rate
        needs_volume = abs(volume - 1.0) >= 0.01
        
        # If nothing to do, return original
        if not needs_resample and not needs_volume:
            return FileResponse(
                tmp_input,
                media_type="audio/wav",
                filename="resampled.wav",
            )
        
        ops = []
        if needs_resample:
            ops.append(f"resample {current_sr}→{sample_rate}Hz")
        if needs_volume:
            ops.append(f"volume {int(volume*100)}%")
        log_info(f"[Resample] Processing: {', '.join(ops)}")
        
        # Resample if needed using SoXr VHQ
        if needs_resample:
            data = soxr.resample(data, current_sr, sample_rate, quality='VHQ')
        
        # Apply volume if needed
        if needs_volume:
            data = data * volume
            data = np.clip(data, -1.0, 1.0)
        
        # Write output
        fd, tmp_output = tempfile.mkstemp(suffix="_resampled.wav")
        os.close(fd)
        sf.write(tmp_output, data.astype(np.float32), sample_rate)
        
        _safe_unlink(tmp_input)
        return FileResponse(
            tmp_output,
            media_type="audio/wav",
            filename="resampled.wav",
            background=BackgroundTask(_safe_unlink, tmp_output),
        )
        
    except Exception as e:
        _safe_unlink(tmp_input)
        log_error(f"Resample error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/postprocess")
async def api_postprocess(
    audio: UploadFile = File(...),
    highpass: float = Form(0.0),  # Default 0 = disabled
    lowpass: float = Form(12000.0),
    bass_freq: float = Form(60.0),
    bass_gain: float = Form(4.0),
    treble_freq: float = Form(8000.0),
    treble_gain: float = Form(2.0),
    reverb_delay: float = Form(0.0),
    reverb_decay: float = Form(0.0),
    crystalizer: float = Form(0.0),
    deesser: float = Form(0.0),
    audio_8d_enabled: bool = Form(False),
    audio_8d_mode: str = Form("rotate"),
    audio_8d_speed: float = Form(0.1),
    audio_8d_depth: float = Form(180.0),  # Arc in degrees
    audio_8d_distance: float = Form(0.3),  # 0=touching ear, 1=far
    audio_8d_quality: str = Form("balanced"),  # fast/balanced/ultra
    audio_8d_itd: bool = Form(None),  # Override: Interaural Time Difference
    audio_8d_proximity: bool = Form(None),  # Override: Near-field proximity effect
    audio_8d_crossfeed: bool = Form(None),  # Override: Natural crossfeed
    audio_8d_micro_movements: bool = Form(None),  # Override: Organic micro-movements
    audio_8d_speech_aware: bool = Form(True),  # Speech-aware transitions (snap to pauses)
    spatial_time_offset: float = Form(0.0),  # For streaming: accumulated time from previous chunks
    pitch_shift_enabled: bool = Form(False),
    pitch_shift_semitones: int = Form(0),
    asmr_enabled: bool = Form(False),
    asmr_tingles: int = Form(60),
    asmr_breathiness: int = Form(65),
    asmr_crispness: int = Form(55),
):
    log_info(f"[POST] Received request, reading audio file: {audio.filename}")
    
    fd, tmp_input = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        log_info(f"[POST] Reading upload content...")
        content = await audio.read()
        file_size_mb = len(content) / (1024 * 1024)
        log_info(f"[POST] Received {file_size_mb:.1f}MB, saving to temp file...")
        
        with open(tmp_input, "wb") as f:
            f.write(content)
        log_info(f"[POST] Saved, starting processing...")

        params = {
            "highpass": highpass,
            "lowpass": lowpass,
            "bass_freq": bass_freq,
            "bass_gain": bass_gain,
            "treble_freq": treble_freq,
            "treble_gain": treble_gain,
            "reverb_delay": reverb_delay,
            "reverb_decay": reverb_decay,
            "crystalizer": crystalizer,
            "deesser": deesser,
            "audio_8d_enabled": audio_8d_enabled,
            "audio_8d_mode": audio_8d_mode,
            "audio_8d_speed": audio_8d_speed,
            "audio_8d_depth": audio_8d_depth,
            "audio_8d_distance": audio_8d_distance,
            "audio_8d_quality": audio_8d_quality,
            "audio_8d_itd": audio_8d_itd,
            "audio_8d_proximity": audio_8d_proximity,
            "audio_8d_crossfeed": audio_8d_crossfeed,
            "audio_8d_micro_movements": audio_8d_micro_movements,
            "audio_8d_speech_aware": audio_8d_speech_aware,
            "spatial_time_offset": spatial_time_offset,
            "pitch_shift_enabled": pitch_shift_enabled,
            "pitch_shift_semitones": pitch_shift_semitones,
            "asmr_enabled": asmr_enabled,
            "asmr_tingles": asmr_tingles,
            "asmr_breathiness": asmr_breathiness,
            "asmr_crispness": asmr_crispness,
        }

        import time as _time
        t_proc_start = _time.perf_counter()
        log_info(f"[POST] Calling post_process_voice...")
        output_path = post_process_voice(tmp_input, params)
        t_proc_end = _time.perf_counter()
        log_info(f"[POST] post_process_voice completed in {(t_proc_end - t_proc_start)*1000:.0f}ms, output: {output_path}")
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail=f"Output file not found: {output_path}")
        
        output_size = os.path.getsize(output_path)
        log_info(f"[POST] Creating FileResponse for {output_size / (1024*1024):.1f}MB file...")
        
        response = FileResponse(
            output_path,
            media_type="audio/wav",
            filename="postprocessed.wav",
            background=BackgroundTask(_cleanup_many, [tmp_input, output_path]),
        )
        log_info(f"[POST] FileResponse created, returning to client")
        return response
    except Exception as e:
        import traceback
        log_error(f"Post-processing error: {type(e).__name__}: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        _cleanup_many([tmp_input])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/blend")
async def api_blend(
    audio: UploadFile = File(...),
    bg_files: str = Form("[]"),
    bg_volumes: str = Form("[]"),
    main_volume: float = Form(1.0),
    bg_delays: str = Form("[]"),
    bg_fade_ins: str = Form("[]"),
    bg_fade_outs: str = Form("[]"),
):
    fd, tmp_input = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        content = await audio.read()
        with open(tmp_input, "wb") as f:
            f.write(content)

        try:
            bg_file_list = json.loads(bg_files)
            bg_volume_list = json.loads(bg_volumes)
            bg_delay_list = json.loads(bg_delays)
            bg_fade_in_list = json.loads(bg_fade_ins)
            bg_fade_out_list = json.loads(bg_fade_outs)
        except Exception:
            bg_file_list = []
            bg_volume_list = []
            bg_delay_list = []
            bg_fade_in_list = []
            bg_fade_out_list = []

        output_path = blend_with_background(tmp_input, bg_file_list, bg_volume_list, main_volume, bg_delay_list, bg_fade_in_list, bg_fade_out_list)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="blended.wav",
            background=BackgroundTask(_cleanup_many, [tmp_input, output_path]),
        )
    except Exception as e:
        log_error(f"Blend error: {e}")
        _cleanup_many([tmp_input])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/save")
async def api_save(audio: UploadFile = File(...), text: str = Form("output")):
    fd, tmp_input = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        content = await audio.read()
        with open(tmp_input, "wb") as f:
            f.write(content)

        output_path = save_output(tmp_input, text)
        return JSONResponse({"success": True, "path": output_path})
    except Exception as e:
        log_error(f"Save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _safe_unlink(tmp_input)


@app.post("/v1/process-chunk")
async def api_process_chunk(
    audio: UploadFile = File(...),
    # JSON string containing all post-processing params (simpler than individual form fields)
    post_params_json: str = Form("{}"),
    # Final resample params
    target_sample_rate: int = Form(44100),
    output_volume: float = Form(1.0),
):
    """
    Combined endpoint: PostProcess + Resample in one HTTP call.
    Reduces round-trips for streaming chunk processing.
    
    post_params_json: JSON string with post-processing parameters, e.g.:
    {
        "highpass": 80, "lowpass": 12000, "bass_gain": 3.0,
        "audio_8d_enabled": true, "audio_8d_mode": "circular", ...
    }
    Empty dict or "{}" skips post-processing.
    """
    import time
    t_start = time.perf_counter()
    
    try:
        content = await audio.read()
        current_bytes = content
        
        # Parse post-process params
        try:
            post_params = json.loads(post_params_json) if post_params_json else {}
        except json.JSONDecodeError:
            post_params = {}
        
        # Step 1: Post-process (if any params provided)
        do_postprocess = bool(post_params) and any(
            post_params.get(k) for k in [
                'highpass', 'bass_gain', 'treble_gain', 'reverb_delay', 'crystalizer', 'deesser',
                'audio_8d_enabled', 'pitch_shift_enabled', 'asmr_enabled'
            ]
        )
        
        if do_postprocess:
            current_bytes = post_process_voice_bytes(current_bytes, post_params)
        
        # Step 2: Resample + volume (if needed) - inline for efficiency
        import soxr
        try:
            data, current_sr = sf.read(io.BytesIO(current_bytes), dtype='float32')
            needs_sr = current_sr != target_sample_rate
            needs_vol = abs(output_volume - 1.0) >= 0.01
            
            if needs_sr or needs_vol:
                if needs_sr:
                    data = soxr.resample(data, current_sr, target_sample_rate, quality='VHQ')
                if needs_vol:
                    data = data * output_volume
                    data = np.clip(data, -1.0, 1.0)
                
                out_buf = io.BytesIO()
                sf.write(out_buf, data.astype(np.float32), target_sample_rate, format='WAV')
                current_bytes = out_buf.getvalue()
            elif not do_postprocess:
                # Keep exact original bytes if absolutely no-op
                current_bytes = content
        except Exception as resample_err:
            log_error(f"Resample step error: {resample_err}")
        
        t_elapsed = (time.perf_counter() - t_start) * 1000
        log_info(f"[ProcessChunk] Completed in {t_elapsed:.0f}ms")

        return Response(content=current_bytes, media_type="audio/wav")
    except Exception as e:
        import traceback
        log_error(f"Process-chunk error: {type(e).__name__}: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse
    import uvicorn

    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(description="VoiceForge Audio Services Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8892)
    args = parser.parse_args()

    log_info(f"Starting Audio Services server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=os.getenv("VF_UVICORN_LOG_LEVEL", "warning").lower(),
        access_log=_env_flag("VF_ACCESS_LOGS", "0"),
    )

