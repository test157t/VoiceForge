"""
Audio Pipeline Service - Thin coordinator for TTS → RVC → PostProcess → Blend.

This module just coordinates calls to the microservices via their clients.
All actual processing happens in the servers:
- chatterbox_server: TTS generation
- rvc_server: Voice conversion 
- audio_services_server: Post-processing, blending, resampling, saving
"""

import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Set
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf  # For debug logging

from util.clients import (
    get_chatterbox_client,
    get_pocket_tts_client,
    get_kokoro_tts_client,
    get_omnivoice_tts_client,
    get_omnivoice_onnx_tts_client,
    get_omnivoice_onnx_gpu_tts_client,
    transcribe_audio,
    run_rvc,
    run_rvc_stream,
    run_postprocess,
    run_blend,
    run_save,
    run_resample,
    run_process_chunk,
)
from util.text_utils import split_text
from util.audio_utils import convert_to_format, get_mime_type, get_audio_info
from util.file_utils import resolve_audio_path
from config import get_config, is_rvc_enabled, is_post_enabled, is_background_enabled, get_bg_tracks
from servers.models.requests import TTSRequest

# Pipeline sample rate (44.1kHz CD quality)
PIPELINE_SAMPLE_RATE = 44100


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


VF_VERBOSE_LOGS = _env_flag("VF_VERBOSE_LOGS", "0")
VF_PIPELINE_DEBUG_LOGS = _env_flag("VF_PIPELINE_DEBUG_LOGS", "0")
VF_METRICS_LOGS = _env_flag("VF_METRICS_LOGS", "0")


def _log_verbose(message: str):
    if VF_VERBOSE_LOGS:
        print(message)


def _log_debug(message: str):
    if VF_PIPELINE_DEBUG_LOGS:
        print(message)


def _log_metrics(message: str):
    if VF_METRICS_LOGS:
        print(message)

# Shared executor for blocking operations
_executor: Optional[ThreadPoolExecutor] = None

def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", os.cpu_count() or 4)))
    return _executor


# Simple cancellation tracking
_cancelled: Set[str] = set()

def cancel_generation(request_id: str) -> bool:
    _cancelled.add(request_id)
    return True

def cancel_all_generations() -> int:
    count = len(_active)
    _cancelled.update(_active)
    return count

def is_cancelled(request_id: str) -> bool:
    return request_id in _cancelled

def get_active_requests() -> List[str]:
    return list(_active)

_active: Set[str] = set()

# Cache OmniVoice prompt transcripts across requests (important for call mode)
_omnivoice_ref_cache: dict = {}
_omnivoice_ref_cache_lock = threading.Lock()
_omnivoice_ref_inflight: dict = {}
_omnivoice_ref_inflight_lock = threading.Lock()
_omnivoice_ref_cache_loaded = False

OMNIVOICE_REF_ASR_MODEL = os.getenv("OMNIVOICE_REF_ASR_MODEL", "glm-asr-nano")
OMNIVOICE_REF_CACHE_FILE = os.getenv(
    "OMNIVOICE_REF_CACHE_FILE",
    os.path.join(tempfile.gettempdir(), "voiceforge_omnivoice_ref_cache.json"),
)


class CancelledException(Exception):
    pass


def _needs_resample(path: str, target_sr: int, volume: float) -> bool:
    info = get_audio_info(path)
    current_sr = info.get("samplerate")
    if abs(volume - 1.0) >= 0.01:
        return True
    if current_sr is None:
        return True
    return current_sr != target_sr


def _norm_voice_cache_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _load_omnivoice_ref_cache_once() -> None:
    global _omnivoice_ref_cache_loaded
    if _omnivoice_ref_cache_loaded:
        return
    with _omnivoice_ref_cache_lock:
        if _omnivoice_ref_cache_loaded:
            return
        try:
            if os.path.exists(OMNIVOICE_REF_CACHE_FILE):
                with open(OMNIVOICE_REF_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        _omnivoice_ref_cache.update(data)
        except Exception:
            pass
        _omnivoice_ref_cache_loaded = True


def _save_omnivoice_ref_cache() -> None:
    try:
        parent = os.path.dirname(OMNIVOICE_REF_CACHE_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = OMNIVOICE_REF_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_omnivoice_ref_cache, f, ensure_ascii=True)
        os.replace(tmp, OMNIVOICE_REF_CACHE_FILE)
    except Exception:
        pass


async def _get_omnivoice_ref_text_cached(
    voice_path: Optional[str],
    provided_ref_text: Optional[str],
    requested_asr_model: Optional[str],
    executor: ThreadPoolExecutor,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Resolve OmniVoice reference transcript once and cache across requests."""
    _load_omnivoice_ref_cache_once()

    if provided_ref_text and str(provided_ref_text).strip():
        return str(provided_ref_text).strip()

    if not voice_path or not os.path.isfile(voice_path):
        return None

    try:
        stat = os.stat(voice_path)
        cache_key = _norm_voice_cache_key(voice_path)
        with _omnivoice_ref_cache_lock:
            cache_entry = _omnivoice_ref_cache.get(cache_key)
        if cache_entry and cache_entry.get("mtime_ns") == stat.st_mtime_ns and cache_entry.get("size") == stat.st_size:
            cached_text = cache_entry.get("text")
            if cached_text:
                return cached_text
    except Exception:
        cache_key = _norm_voice_cache_key(voice_path)

    loop = asyncio.get_running_loop()

    # De-duplicate concurrent transcriptions for the same prompt file
    with _omnivoice_ref_inflight_lock:
        inflight = _omnivoice_ref_inflight.get(cache_key)
        if inflight is None:
            inflight = loop.create_future()
            _omnivoice_ref_inflight[cache_key] = inflight
            is_owner = True
        else:
            is_owner = False

    if is_owner:
        async def _compute_ref_text():
            ref_text: Optional[str] = None
            try:
                if status_callback:
                    status_callback("Transcribing OmniVoice prompt (ASR)...")
                asr_model = (str(requested_asr_model or "").strip() or OMNIVOICE_REF_ASR_MODEL)
                asr_result = await loop.run_in_executor(
                    executor,
                    lambda: transcribe_audio(
                        audio_path=voice_path,
                        language="auto",
                        response_format="json",
                        model=asr_model,
                    ),
                )
                ref_text = (asr_result.get("text") or "").strip() or None
                if ref_text:
                    try:
                        stat_local = os.stat(voice_path)
                        with _omnivoice_ref_cache_lock:
                            _omnivoice_ref_cache[cache_key] = {
                                "mtime_ns": stat_local.st_mtime_ns,
                                "size": stat_local.st_size,
                                "text": ref_text,
                            }
                            _save_omnivoice_ref_cache()
                    except Exception:
                        pass
            except Exception:
                ref_text = None
            finally:
                with _omnivoice_ref_inflight_lock:
                    fut = _omnivoice_ref_inflight.pop(cache_key, None)
                    if fut is not None and not fut.done():
                        fut.set_result(ref_text)

        # Detached task: request cancellation should not cancel shared transcript work.
        asyncio.create_task(_compute_ref_text())

    try:
        return await asyncio.shield(inflight)
    except asyncio.CancelledError:
        return None
    except Exception:
        return None


def _get_tts_backend(request: TTSRequest) -> str:
    """Resolve TTS backend with server-side guardrails.

    Some clients can accidentally send a stale default backend. If OmniVoice-specific
    fields are present, prefer OmniVoice to avoid misrouting to Chatterbox.
    """
    allowed = ('chatterbox', 'pocket_tts', 'kokoro', 'omnivoice', 'omnivoice_onnx', 'omnivoice_onnx_gpu')
    backend = getattr(request, 'tts_backend', 'chatterbox')
    if backend not in allowed:
        backend = 'chatterbox'

    # Strong OmniVoice signals: explicit ref-text/model or non-default voice value.
    omni_ref_text = getattr(request, 'omnivoice_ref_text', None)
    omni_ref_model = getattr(request, 'omnivoice_ref_asr_model', None)
    omni_voice = str(getattr(request, 'omnivoice_voice', '') or '').strip()
    has_omni_signal = bool(
        (omni_ref_text and str(omni_ref_text).strip())
        or (omni_ref_model and str(omni_ref_model).strip())
        or (omni_voice and omni_voice.lower() not in {'auto', 'default', 'random'})
    )

    # If client requested chatterbox but sent OmniVoice fields, trust OmniVoice fields.
    if backend == 'chatterbox' and has_omni_signal:
        return 'omnivoice'

    return backend


@dataclass
class AudioPipelineResult:
    """Result of audio pipeline execution."""
    success: bool
    audio_data: Optional[bytes] = None
    mime_type: str = "audio/mpeg"
    error: Optional[str] = None
    temp_files: List[str] = field(default_factory=list)
   
    def cleanup(self):
        for path in self.temp_files:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except:
                pass


async def generate_audio(
    request: TTSRequest,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    request_id: Optional[str] = None,
) -> AudioPipelineResult:
    """
    Pipeline: TTS → RVC → PostProcess → Blend → Format
   
    Each step calls the appropriate server via clients.py.
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    temp_files: List[str] = []
    _active.add(request_id)
    _cancelled.discard(request_id)
   
    def status(msg: str):
        if status_callback:
            status_callback(msg)
        _log_verbose(f"[{request_id}] {msg}")
   
    def progress(p: float):
        if progress_callback:
            progress_callback(p)
   
    def check():
        if is_cancelled(request_id):
            raise CancelledException()
   
    try:
        config = get_config()
        executor = get_executor()
        tts_backend = _get_tts_backend(request)
        _log_verbose(f"[{request_id}] Pipeline: tts_backend={tts_backend}, request.tts_backend={getattr(request, 'tts_backend', 'NOT SET')}")
       
        # Flags
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_bg = request.enable_background if request.enable_background is not None else is_background_enabled()
       
        # Resolve prompt audio (for Chatterbox) or voice-like input for TTS backends
        prompt = None
        pocket_voice = None
        omnivoice_voice = None
        omnivoice_ref_text = getattr(request, 'omnivoice_ref_text', None)
        omnivoice_ref_asr_model = getattr(request, 'omnivoice_ref_asr_model', None)
        if tts_backend == "chatterbox":
            prompt = resolve_audio_path(request.chatterbox_prompt_audio)
            if not prompt or not os.path.exists(prompt or ""):
                return AudioPipelineResult(success=False, error="Prompt audio required for Chatterbox", temp_files=temp_files)
        elif tts_backend == "pocket_tts":
            # Pocket TTS can use built-in voice names or file paths for cloning
            pocket_voice = getattr(request, 'pocket_tts_voice', 'alba')
            # If it looks like a path, try to resolve it
            if pocket_voice and ('/' in pocket_voice or '\\' in pocket_voice or pocket_voice.endswith('.wav')):
                resolved = resolve_audio_path(pocket_voice)
                if resolved and os.path.exists(resolved):
                    pocket_voice = resolved
        elif tts_backend == "kokoro":
            # Kokoro uses voice presets (af, am, bf, bm)
            kokoro_voice = getattr(request, 'kokoro_voice', 'af_sarah')
        elif tts_backend in {"omnivoice", "omnivoice_onnx", "omnivoice_onnx_gpu"}:
            # OmniVoice uses auto/instruct text/or ref audio path
            omnivoice_voice = getattr(request, 'omnivoice_voice', 'auto')
            if omnivoice_voice and ('/' in omnivoice_voice or '\\' in omnivoice_voice or omnivoice_voice.endswith('.wav')):
                resolved = resolve_audio_path(omnivoice_voice)
                if resolved and os.path.exists(resolved):
                    omnivoice_voice = resolved
            omnivoice_ref_text = await _get_omnivoice_ref_text_cached(
                voice_path=omnivoice_voice,
                provided_ref_text=omnivoice_ref_text,
                requested_asr_model=omnivoice_ref_asr_model,
                executor=executor,
                status_callback=None,
            )

        # === Step 1: TTS ===
        check()
        status(f"Generating TTS ({tts_backend})...")
        progress(0.1)

        if tts_backend == "pocket_tts":
            _log_verbose(f"[{request_id}] Using Pocket TTS with voice={pocket_voice}")
            pocket_tts = get_pocket_tts_client()
            _log_verbose(f"[{request_id}] Pocket TTS client URL: {pocket_tts.server_url}")
            status("Generating TTS (check Pocket TTS terminal for progress)...")

            # Use the regular generate endpoint - server logs per-sentence progress
            # The streaming endpoint has issues with very large audio (base64 encoding)
            max_tokens = request.tts_batch_tokens or 50 # Use UI chunk size setting
            tts_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: pocket_tts.generate(
                    text=request.input,
                    voice=pocket_voice,
                    speed=getattr(request, 'speed', 1.0),
                    max_tokens=max_tokens,
                )
            )
            _log_verbose(f"[{request_id}] Pocket TTS generated: {tts_path}")
        elif tts_backend == "kokoro":
            _log_verbose(f"[{request_id}] Using Kokoro TTS with voice={kokoro_voice}")
            kokoro_tts = get_kokoro_tts_client()
            _log_verbose(f"[{request_id}] Kokoro TTS client URL: {kokoro_tts.server_url}")
            status("Generating TTS with Kokoro...")
            max_tokens = request.tts_batch_tokens or 50
            token_method = request.tts_token_method or "tiktoken"

            tts_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: kokoro_tts.generate(
                    text=request.input,
                    voice=kokoro_voice,
                    speed=getattr(request, 'speed', 1.0),
                    max_tokens=max_tokens,
                    token_method=token_method,
                )
            )
            _log_verbose(f"[{request_id}] Kokoro TTS generated: {tts_path}")
        elif tts_backend in {"omnivoice", "omnivoice_onnx", "omnivoice_onnx_gpu"}:
            is_onnx_cpu = tts_backend == "omnivoice_onnx"
            is_onnx_gpu = tts_backend == "omnivoice_onnx_gpu"
            _log_verbose(
                f"[{request_id}] Using OmniVoice"
                f"{' ONNX GPU' if is_onnx_gpu else (' ONNX' if is_onnx_cpu else '')}"
                f" with voice={omnivoice_voice}"
            )
            if is_onnx_gpu:
                omnivoice_tts = get_omnivoice_onnx_gpu_tts_client()
            elif is_onnx_cpu:
                omnivoice_tts = get_omnivoice_onnx_tts_client()
            else:
                omnivoice_tts = get_omnivoice_tts_client()
            _log_verbose(f"[{request_id}] OmniVoice client URL: {omnivoice_tts.server_url}")
            status(
                "Generating TTS with OmniVoice ONNX GPU..."
                if is_onnx_gpu
                else ("Generating TTS with OmniVoice ONNX..." if is_onnx_cpu else "Generating TTS with OmniVoice...")
            )
            max_tokens = request.tts_batch_tokens or 50
            token_method = request.tts_token_method or "tiktoken"

            tts_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: omnivoice_tts.generate(
                    text=request.input,
                    voice=omnivoice_voice,
                    ref_text=omnivoice_ref_text,
                    speed=getattr(request, 'speed', 1.0),
                    max_tokens=max_tokens,
                    token_method=token_method,
                )
            )
            _log_verbose(f"[{request_id}] OmniVoice generated: {tts_path}")
        else: # chatterbox
            chatterbox = get_chatterbox_client()
            tts_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: chatterbox.generate(
                    text=request.input,
                    prompt_audio_path=prompt,
                    seed=request.chatterbox_seed or 0,
                    max_tokens=request.tts_batch_tokens or 200,
                )
            )
        temp_files.append(tts_path)
        current = tts_path
        progress(0.3)
       
        # === Step 2: RVC ===
        check()
        if do_rvc:
            status("Running RVC...")
            model = request.rvc_model or config.get("rvc_model", "Goddess_Nicole")
            rvc_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_rvc(current, model, {}, request_id=request_id)
            )
            temp_files.append(rvc_path)
            current = rvc_path
        progress(0.5)
       
        # === Step 2.5: Normalize to Pipeline Sample Rate ===
        # Post-processing effects (especially spatial audio) behave differently at different sample rates
        # Normalizing here ensures consistent behavior
        check()
        if do_post and _needs_resample(current, PIPELINE_SAMPLE_RATE, 1.0):
            status("Normalizing sample rate...")
            norm_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_resample(current, PIPELINE_SAMPLE_RATE, 1.0, request_id=request_id)
            )
            if norm_path != current:
                temp_files.append(norm_path)
                current = norm_path
       
        # === Step 3: Post-Processing ===
        check()
        if do_post:
            post_params = request.get_post_params()
            if post_params.needs_processing():
                status("Post-processing...")
                post_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: run_postprocess(current, post_params.to_dict(), request_id=request_id)
                )
                temp_files.append(post_path)
                current = post_path
        progress(0.7)
       
        # === Step 4: Background Blend ===
        check()
        if do_bg:
            bg_params = request.get_background_params()
            bg_files, bg_vols, bg_delays, bg_fins, bg_fouts = [], [], [], [], []
           
            tracks = get_bg_tracks() if bg_params.use_config_tracks else [
                {"file": f, "volume": bg_params.volumes[i] if i < len(bg_params.volumes) else 0.3,
                 "delay": bg_params.delays[i] if i < len(bg_params.delays) else 0,
                 "fade_in": bg_params.fade_ins[i] if i < len(bg_params.fade_ins) else 0,
                 "fade_out": bg_params.fade_outs[i] if i < len(bg_params.fade_outs) else 0}
                for i, f in enumerate(bg_params.files)
            ]
           
            for t in tracks:
                if not t or not t.get("file"):
                    continue
                resolved = resolve_audio_path(str(t["file"]))
                if resolved and os.path.exists(resolved) and float(t.get("volume", 0.3)) > 0:
                    bg_files.append(resolved)
                    bg_vols.append(float(t.get("volume", 0.3)))
                    bg_delays.append(float(t.get("delay", 0)))
                    bg_fins.append(float(t.get("fade_in", 0)))
                    bg_fouts.append(float(t.get("fade_out", 0)))
           
            if bg_files:
                status("Blending background...")
                blend_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: run_blend(current, bg_files, bg_vols, 1.0, bg_delays=bg_delays,
                                     bg_fade_ins=bg_fins, bg_fade_outs=bg_fouts, request_id=request_id)
                )
                temp_files.append(blend_path)
                current = blend_path
        progress(0.85)
       
        # === Step 5: Output Volume ===
        vol = getattr(request, 'output_volume', 1.0)
        if _needs_resample(current, PIPELINE_SAMPLE_RATE, vol):
            vol_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_resample(current, PIPELINE_SAMPLE_RATE, vol, request_id=request_id)
            )
            if vol_path != current:
                temp_files.append(vol_path)
                current = vol_path
        progress(0.9)
       
        # === Step 6: Save (optional) ===
        if getattr(request, "save_output", False):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: run_save(current, request.input, request_id=request_id)
                )
            except Exception as e:
                status(f"Warning: Save failed: {e}")
        progress(0.95)
       
        # === Step 7: Convert to output format ===
        audio_data = await asyncio.to_thread(
            convert_to_format, current, request.response_format, request.speed
        )
        progress(1.0)
        status("Complete!")
       
        return AudioPipelineResult(
            success=True,
            audio_data=audio_data,
            mime_type=get_mime_type(request.response_format),
            temp_files=temp_files,
        )
       
    except CancelledException:
        return AudioPipelineResult(success=False, error="Cancelled", temp_files=temp_files)
    except Exception as e:
        return AudioPipelineResult(success=False, error=str(e), temp_files=temp_files)
    finally:
        _active.discard(request_id)
        _cancelled.discard(request_id)


async def generate_audio_streaming(
    request: TTSRequest,
    status_callback: Optional[Callable[[str], None]] = None,
    request_id: Optional[str] = None,
):
    """
    Streaming pipeline: TTS → RVC → PostProcess per chunk, yielded as SSE events.
   
    Background blending info is included for client-side mixing.
    """
    import base64
    import io
    import tempfile
    import threading
    from pydub import AudioSegment
   
    request_id = request_id or str(uuid.uuid4())[:8]
    temp_files: List[str] = []
    processed_chunk_paths: Optional[List[str]] = None  # Initialize here so finally block can access
    _active.add(request_id)
    _cancelled.discard(request_id)
   
    def status(msg: str):
        if status_callback:
            status_callback(msg)
        _log_verbose(f"[{request_id}] {msg}")
   
    try:
        stream_start = time.perf_counter()
        first_chunk_emitted_at = None
        total_input_bytes = 0
        total_output_bytes = 0

        config = get_config()
        executor = get_executor()
       
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_bg = request.enable_background if request.enable_background is not None else is_background_enabled()
        tts_backend = _get_tts_backend(request)
       
        prompt = None
        pocket_voice = None
        kokoro_voice = None
        omnivoice_voice = None
        omnivoice_ref_text = getattr(request, 'omnivoice_ref_text', None)
        omnivoice_ref_asr_model = getattr(request, 'omnivoice_ref_asr_model', None)
        if tts_backend == "chatterbox":
            prompt = resolve_audio_path(request.chatterbox_prompt_audio)
            if not prompt or not os.path.exists(prompt or ""):
                yield {"type": "error", "message": "Prompt audio required"}
                return
        elif tts_backend == "pocket_tts":
            # Pocket TTS can use built-in voice names or file paths for cloning
            pocket_voice = getattr(request, 'pocket_tts_voice', 'alba')
            if pocket_voice and ('/' in pocket_voice or '\\' in pocket_voice or pocket_voice.endswith('.wav')):
                resolved = resolve_audio_path(pocket_voice)
                if resolved and os.path.exists(resolved):
                    pocket_voice = resolved
        elif tts_backend == "kokoro":
            # Kokoro uses voice presets (af, am, bf, bm)
            kokoro_voice = getattr(request, 'kokoro_voice', 'af_sarah')
        elif tts_backend in {"omnivoice", "omnivoice_onnx", "omnivoice_onnx_gpu"}:
            omnivoice_voice = getattr(request, 'omnivoice_voice', 'auto')
            if omnivoice_voice and ('/' in omnivoice_voice or '\\' in omnivoice_voice or omnivoice_voice.endswith('.wav')):
                resolved = resolve_audio_path(omnivoice_voice)
                if resolved and os.path.exists(resolved):
                    omnivoice_voice = resolved
            omnivoice_ref_text = await _get_omnivoice_ref_text_cached(
                voice_path=omnivoice_voice,
                provided_ref_text=omnivoice_ref_text,
                requested_asr_model=omnivoice_ref_asr_model,
                executor=executor,
                status_callback=status,
            )
       
        # Gather background track info for client
        bg_tracks = []
        if do_bg:
            bg_params = request.get_background_params()
            tracks = get_bg_tracks() if bg_params.use_config_tracks else [
                {"file": f, "volume": bg_params.volumes[i] if i < len(bg_params.volumes) else 0.3,
                 "delay": bg_params.delays[i] if i < len(bg_params.delays) else 0,
                 "fade_in": bg_params.fade_ins[i] if i < len(bg_params.fade_ins) else 0,
                 "fade_out": bg_params.fade_outs[i] if i < len(bg_params.fade_outs) else 0}
                for i, f in enumerate(bg_params.files)
            ]
            for t in tracks:
                if t and t.get("file"):
                    resolved = resolve_audio_path(str(t["file"]))
                    if resolved and os.path.exists(resolved):
                        bg_tracks.append({"file": resolved, "volume": float(t.get("volume", 0.3)),
                                         "delay": float(t.get("delay", 0)), "fade_in": float(t.get("fade_in", 0)),
                                         "fade_out": float(t.get("fade_out", 0))})
       
        rvc_model = request.rvc_model or config.get("rvc_model", "Goddess_Nicole")
        post_params = request.get_post_params().to_dict() if do_post and request.get_post_params().needs_processing() else None
        output_vol = getattr(request, 'output_volume', 1.0)
       
        # Track time offset for spatial audio panning continuity
        spatial_time_offset = 0.0
       
        # Track processed chunks for save_output
        save_output = getattr(request, "save_output", False)
        processed_chunk_paths = [] if save_output else None

        event_queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()
        loop = asyncio.get_event_loop()
       
        def reader():
            try:
                if tts_backend == "pocket_tts":
                    # Pocket TTS - use streaming endpoint for real-time chunk processing
                    import base64
                    import httpx

                    pocket_tts = get_pocket_tts_client()
                    max_tokens = request.tts_batch_tokens or 50

                    payload = {
                        "model": "pocket-tts",
                        "input": request.input,
                        "voice": pocket_voice,
                        "response_format": "wav",
                        "speed": getattr(request, 'speed', 1.0),
                        "max_tokens": max_tokens
                    }

                    _log_verbose(f"[{request_id}] PocketTTS streaming - voice={pocket_voice}, chunks~{len(request.input)//100}")

                    event_count = 0

                    # Use httpx for proper unbuffered SSE streaming
                    with httpx.Client(timeout=httpx.Timeout(3600.0, connect=30.0)) as client:
                        with client.stream("POST", f"{pocket_tts.server_url}/v1/audio/speech/stream", json=payload) as response:
                            if response.status_code != 200:
                                print(f"[{request_id}] PocketTTS error: {response.status_code}")
                                loop.call_soon_threadsafe(event_queue.put_nowait, {
                                    "type": "error",
                                    "message": f"Pocket TTS error: {response.status_code}"
                                })
                                return

                            _log_verbose(f"[{request_id}] SSE connected, streaming...")

                            for line in response.iter_lines():
                                if stop_event.is_set():
                                    _log_verbose(f"[{request_id}] Cancelled after {event_count} events")
                                    break

                                if not line or not line.startswith('data: '):
                                    continue

                                try:
                                    event = json.loads(line[6:])
                                    event_type = event.get('type')
                                    event_count += 1

                                    if event_type == 'start':
                                        _log_verbose(f"[{request_id}] START: {event.get('chunks')} chunks")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "start",
                                            "chunks": event.get("chunks", 1),
                                            "sample_rate": event.get("sample_rate", 48000)
                                        })

                                    elif event_type == 'chunk':
                                        audio_b64 = event.get('audio_bytes_b64', '')
                                        audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b''
                                        idx = event.get('index', 0)
                                        _log_verbose(f"[{request_id}] CHUNK {idx}: {len(audio_bytes)} bytes -> RVC")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "chunk",
                                            "index": idx,
                                            "audio_bytes": audio_bytes
                                        })

                                    elif event_type == 'complete':
                                        _log_verbose(f"[{request_id}] COMPLETE: {event.get('chunks_sent', 0)} chunks sent")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "complete"})

                                    elif event_type == 'error':
                                        print(f"[{request_id}] ERROR: {event.get('message')}")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "error",
                                            "message": event.get("message", "Unknown error")
                                        })

                                    elif event_type == 'warning':
                                        _log_verbose(f"[{request_id}] WARNING: {event.get('message')}")

                                except json.JSONDecodeError:
                                    continue

                            _log_verbose(f"[{request_id}] Stream done, {event_count} events")

                elif tts_backend == "kokoro":
                    # Kokoro TTS - use streaming endpoint
                    import base64
                    import httpx

                    kokoro_tts = get_kokoro_tts_client()
                    kokoro_voice = getattr(request, 'kokoro_voice', 'af_sarah')
                    max_tokens = request.tts_batch_tokens or 50
                    token_method = request.tts_token_method or "tiktoken"

                    payload = {
                        "model": "kokoro-tts",
                        "input": request.input,
                        "voice": kokoro_voice,
                        "response_format": "wav",
                        "speed": getattr(request, 'speed', 1.0),
                        "max_tokens": max_tokens,
                        "token_method": token_method,
                    }

                    _log_verbose(f"[{request_id}] KokoroTTS streaming - voice={kokoro_voice}")

                    event_count = 0

                    # Use httpx for proper unbuffered SSE streaming
                    with httpx.Client(timeout=httpx.Timeout(3600.0, connect=30.0)) as client:
                        with client.stream("POST", f"{kokoro_tts.server_url}/v1/audio/speech/stream", json=payload) as response:
                            if response.status_code != 200:
                                print(f"[{request_id}] KokoroTTS error: {response.status_code}")
                                loop.call_soon_threadsafe(event_queue.put_nowait, {
                                    "type": "error",
                                    "message": f"Kokoro TTS error: {response.status_code}"
                                })
                                return

                            _log_verbose(f"[{request_id}] SSE connected, streaming...")

                            for line in response.iter_lines():
                                if stop_event.is_set():
                                    _log_verbose(f"[{request_id}] Cancelled after {event_count} events")
                                    break

                                if not line or not line.startswith('data: '):
                                    continue

                                try:
                                    event = json.loads(line[6:])
                                    event_type = event.get('type')
                                    event_count += 1

                                    if event_type == 'start':
                                        _log_verbose(f"[{request_id}] START: {event.get('chunks')} chunks")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "start",
                                            "chunks": event.get("chunks", 1),
                                            "sample_rate": event.get("sample_rate", 24000)
                                        })

                                    elif event_type == 'chunk':
                                        # Kokoro sends 'audio', Pocket TTS sends 'audio_bytes_b64'
                                        audio_b64 = event.get('audio_bytes_b64', '') or event.get('audio', '')
                                        audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b''
                                        idx = event.get('index', 0)
                                        _log_verbose(f"[{request_id}] CHUNK {idx}: {len(audio_bytes)} bytes -> RVC")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "chunk",
                                            "index": idx,
                                            "audio_bytes": audio_bytes
                                        })

                                    elif event_type == 'complete':
                                        _log_verbose(f"[{request_id}] COMPLETE: {event.get('chunks_sent', 0)} chunks sent")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "complete"})

                                    elif event_type == 'error':
                                        print(f"[{request_id}] ERROR: {event.get('message')}")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "error",
                                            "message": event.get("message", "Unknown error")
                                        })

                                    elif event_type == 'warning':
                                        _log_verbose(f"[{request_id}] WARNING: {event.get('message')}")

                                except json.JSONDecodeError:
                                    continue

                    _log_verbose(f"[{request_id}] Stream done, {event_count} events")

                elif tts_backend in {"omnivoice", "omnivoice_onnx", "omnivoice_onnx_gpu"}:
                    is_onnx_cpu = tts_backend == "omnivoice_onnx"
                    is_onnx_gpu = tts_backend == "omnivoice_onnx_gpu"
                    if is_onnx_gpu:
                        omnivoice_tts = get_omnivoice_onnx_gpu_tts_client()
                    elif is_onnx_cpu:
                        omnivoice_tts = get_omnivoice_onnx_tts_client()
                    else:
                        omnivoice_tts = get_omnivoice_tts_client()
                    omnivoice_voice = getattr(request, 'omnivoice_voice', 'auto')
                    max_tokens = request.tts_batch_tokens or 50
                    token_method = request.tts_token_method or "tiktoken"

                    _log_verbose(
                        f"[{request_id}] OmniVoice"
                        f"{' ONNX GPU' if is_onnx_gpu else (' ONNX' if is_onnx_cpu else '')}"
                        f" streaming - voice={omnivoice_voice}"
                    )
                    text_chunks = split_text(
                        request.input,
                        max_tokens=max_tokens,
                        token_method=token_method,
                    )
                    text_chunks = [c for c in text_chunks if c and c.strip()]

                    total_chunks = len(text_chunks)
                    loop.call_soon_threadsafe(event_queue.put_nowait, {
                        "type": "start",
                        "chunks": total_chunks,
                        "sample_rate": 48000,
                    })

                    if total_chunks == 0:
                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                            "type": "error",
                            "message": "No valid text chunks for OmniVoice",
                        })
                        return

                    for idx, text_chunk in enumerate(text_chunks):
                        if stop_event.is_set():
                            _log_verbose(f"[{request_id}] Reader thread: stop_event set, breaking at chunk {idx}")
                            break

                        out_path = None
                        try:
                            out_path = omnivoice_tts.generate(
                                text=text_chunk,
                                voice=omnivoice_voice,
                                ref_text=omnivoice_ref_text,
                                speed=getattr(request, 'speed', 1.0),
                                max_tokens=max_tokens,
                                token_method=token_method,
                                prechunked=True,
                                request_id=request_id,
                            )

                            with open(out_path, "rb") as f:
                                audio_bytes = f.read()

                            loop.call_soon_threadsafe(event_queue.put_nowait, {
                                "type": "chunk",
                                "index": idx,
                                "audio_bytes": audio_bytes,
                                "text": text_chunk,
                            })
                        except Exception as e:
                            loop.call_soon_threadsafe(event_queue.put_nowait, {
                                "type": "error",
                                "message": str(e),
                            })
                            return
                        finally:
                            try:
                                if out_path and os.path.exists(out_path):
                                    os.remove(out_path)
                            except Exception:
                                pass

                    loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "complete"})
                    _log_verbose(f"[{request_id}] OmniVoice stream iteration complete, generated {total_chunks} chunks")

                else: # chatterbox
                    chatterbox = get_chatterbox_client()
                    stream = chatterbox.stream_events(
                        text=request.input,
                        prompt_audio_path=prompt,
                        seed=request.chatterbox_seed or 0,
                        max_tokens=request.tts_batch_tokens or 200,
                        request_id=request_id,
                        stop_event=stop_event
                    )
                    event_count = 0
                    for event in stream:
                        # Check if cancelled BEFORE processing event
                        if stop_event.is_set():
                            _log_verbose(f"[{request_id}] Reader thread: stop_event set, breaking after {event_count} events")
                            break
                       
                        event_count += 1
                        event_type = event.get("type", "unknown")
                        _log_verbose(f"[{request_id}] Received event #{event_count}: type={event_type}")
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    _log_verbose(f"[{request_id}] Stream iteration complete, received {event_count} events")
            except Exception as e:
                print(f"[{request_id}] Stream reader exception: {e}")
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    {"type": "error", "message": str(e)}
                )
            finally:
                _log_verbose(f"[{request_id}] Reader thread finishing")
                loop.call_soon_threadsafe(event_queue.put_nowait, None)
       
        threading.Thread(target=reader, daemon=True).start()
       
        total_duration = 0.0
        total_chunks = None
        chunks_sent = 0
       
        while True:
            event = await event_queue.get()
            if event is None:
                break
           
            if is_cancelled(request_id):
                stop_event.set()
                yield {"type": "cancelled", "message": "Cancelled", "chunks_sent": chunks_sent}
                return
           
            event_type = event.get("type")
           
            if event_type == "start":
                total_chunks = event.get("chunks")
                yield {
                    "type": "start",
                    "request_id": request_id,
                    "chunks": total_chunks or 1,
                    "rvc_enabled": do_rvc,
                    "post_enabled": bool(post_params),
                    "background_enabled": bool(bg_tracks),
                    "background_tracks": bg_tracks,
                    "sample_rate": PIPELINE_SAMPLE_RATE,
                    "tts_backend": tts_backend,
                }
                status(f"Streaming {total_chunks or 1} chunks")
                continue
           
            if event_type == "error":
                yield {"type": "error", "message": event.get("message", "Unknown error")}
                return
           
            if event_type == "complete":
                # Save combined audio if requested
                if save_output and processed_chunk_paths:
                    try:
                        status(f"Saving merged audio ({len(processed_chunk_paths)} chunks)...")
                        # Combine all processed chunks
                        combined_seg = None
                        for chunk_path in processed_chunk_paths:
                            try:
                                seg = AudioSegment.from_file(chunk_path)
                                if combined_seg is None:
                                    combined_seg = seg
                                else:
                                    combined_seg += seg
                            except Exception as e:
                                print(f"[{request_id}] Warning: Failed to load chunk {chunk_path}: {e}")
                       
                        if combined_seg is not None:
                            # Export to temp file and save
                            fd, merged_path = tempfile.mkstemp(suffix="_merged.wav")
                            os.close(fd)
                            combined_seg.export(merged_path, format="wav")
                           
                            try:
                                run_save(merged_path, request.input, request_id=request_id)
                                status("Merged audio saved!")
                            except Exception as e:
                                print(f"[{request_id}] Warning: Save failed: {e}")
                            finally:
                                try:
                                    os.remove(merged_path)
                                except:
                                    pass
                    except Exception as e:
                        print(f"[{request_id}] Warning: Failed to save merged audio: {e}")
                    finally:
                        # Cleanup processed chunk files
                        for pf in processed_chunk_paths:
                            try:
                                if pf and os.path.exists(pf):
                                    os.remove(pf)
                            except:
                                pass
               
                yield {"type": "complete", "chunks_sent": chunks_sent, "total_duration": round(total_duration, 2)}
                elapsed = time.perf_counter() - stream_start
                ttfa = first_chunk_emitted_at if first_chunk_emitted_at is not None else elapsed
                _log_metrics(
                    f"[{request_id}] STREAM METRICS backend={tts_backend} chunks={chunks_sent} "
                    f"ttfa_ms={ttfa*1000:.0f} total_ms={elapsed*1000:.0f} "
                    f"in_kb={total_input_bytes/1024:.1f} out_kb={total_output_bytes/1024:.1f}"
                )
                return
           
            if event_type != "chunk":
                continue
           
            chunk_temps = []
            chunk_index = event.get("index", chunks_sent)
            chunk_text = event.get("text", "")
           
            # Write raw TTS chunk to temp file
            audio_bytes = event.get("audio_bytes")
            if not audio_bytes:
                continue
            total_input_bytes += len(audio_bytes)

            # Fast path: if no extra processing is required, avoid temp file I/O.
            # Keep save_output on the existing path to preserve merged-save behavior.
            fast_path_used = False
            if (not do_rvc) and (post_params is None) and (not save_output):
                try:
                    info_mem = sf.info(io.BytesIO(audio_bytes))
                    mem_duration = (info_mem.frames / info_mem.samplerate) if info_mem.samplerate else 0.0
                    mem_needs_resample = (
                        info_mem.samplerate != PIPELINE_SAMPLE_RATE
                        or abs(output_vol - 1.0) >= 0.01
                    )
                    if not mem_needs_resample:
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        total_duration += mem_duration
                        chunks_sent += 1
                        total_output_bytes += len(audio_bytes)
                        if first_chunk_emitted_at is None:
                            first_chunk_emitted_at = time.perf_counter() - stream_start

                        yield {
                            "type": "chunk",
                            "index": chunk_index,
                            "total": total_chunks or 1,
                            "audio": audio_b64,
                            "duration": round(mem_duration, 2),
                            "text": (chunk_text or "")[:100],
                        }
                        fast_path_used = True
                except Exception:
                    fast_path_used = False

            if fast_path_used:
                continue
           
            fd, tts_path = tempfile.mkstemp(suffix="_tts_chunk.wav")
            os.close(fd)
            with open(tts_path, "wb") as f:
                f.write(audio_bytes)
            chunk_temps.append(tts_path)
           
            # Debug: log audio properties from TTS
            try:
                import soundfile as sf
                _info = sf.info(tts_path)
                _log_debug(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: sr={_info.samplerate}, channels={_info.channels}, frames={_info.frames}, duration={_info.frames/_info.samplerate:.2f}s, bytes={len(audio_bytes)}")
            except Exception as e:
                _log_debug(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: failed to get info: {e}")
           
            # Standard path - works for Chatterbox chunks
            current = tts_path
           
            # RVC (blocking for small chunks is fine)
            if do_rvc:
                status(f"RVC chunk {chunk_index + 1}")
                rvc_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda p=current: run_rvc_stream(p, rvc_model, {}, request_id=request_id)
                )
                chunk_temps.append(rvc_path)
                current = rvc_path
                # Debug: log after RVC
                try:
                    _info = sf.info(current)
                    _log_debug(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} after RVC: sr={_info.samplerate}, frames={_info.frames}")
                except:
                    pass
           
            # Normalize sample rate before post-processing
            # Ensures consistent spatial audio behavior
            if post_params and _needs_resample(current, PIPELINE_SAMPLE_RATE, 1.0):
                norm_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda p=current: run_resample(p, PIPELINE_SAMPLE_RATE, 1.0, request_id=request_id)
                )
                if norm_path != current:
                    chunk_temps.append(norm_path)
                    current = norm_path
                    _log_debug(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} after resample: sr=44100")
           
            # Combined PostProcess + Resample in one HTTP call (optimized)
            needs_post = post_params is not None
            needs_resample = _needs_resample(current, PIPELINE_SAMPLE_RATE, output_vol)
           
            if needs_post or needs_resample:
                status(f"Processing chunk {chunk_index + 1}")
               
                # Prepare post params with spatial offset
                chunk_post_params = None
                if needs_post:
                    chunk_post_params = post_params.copy()
                    chunk_post_params["spatial_time_offset"] = spatial_time_offset
               
                # Single HTTP call for PostProcess + Resample
                processed_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda p=current, pp=chunk_post_params: run_process_chunk(
                        p, pp, PIPELINE_SAMPLE_RATE, output_vol, request_id=request_id
                    )
                )
                if processed_path and processed_path != current:
                    chunk_temps.append(processed_path)
                    current = processed_path
           
            # Read and encode
            with open(current, "rb") as f:
                final_audio_bytes = f.read()
                audio_b64 = base64.b64encode(final_audio_bytes).decode('utf-8')
                total_output_bytes += len(final_audio_bytes)
           
            try:
                import soundfile as sf
                info = sf.info(current)
                duration = (info.frames / info.samplerate) if info.samplerate else 0
            except Exception:
                try:
                    seg = AudioSegment.from_file(current)
                    duration = len(seg) / 1000.0
                except Exception:
                    duration = 0
           
            total_duration += duration
            spatial_time_offset += duration  # Track time for panning continuity
            chunks_sent += 1
            if first_chunk_emitted_at is None:
                first_chunk_emitted_at = time.perf_counter() - stream_start
           
            yield {
                "type": "chunk",
                "index": chunk_index,
                "total": total_chunks or 1,
                "audio": audio_b64,
                "duration": round(duration, 2),
                "text": (chunk_text or "")[:100],
            }
           
            # Keep final processed file for save_output, cleanup others
            if save_output and processed_chunk_paths is not None:
                # Keep the final processed file (current) for later merging
                processed_chunk_paths.append(current)
                # Remove current from chunk_temps so it doesn't get deleted
                if current in chunk_temps:
                    chunk_temps.remove(current)
           
            # Cleanup chunk temps (but not the final processed file if save_output)
            for tf in chunk_temps:
                try:
                    if tf and os.path.exists(tf):
                        os.remove(tf)
                except:
                    pass
       
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}
    finally:
        _active.discard(request_id)
        _cancelled.discard(request_id)
        for tf in temp_files:
            try:
                if tf and os.path.exists(tf):
                    os.remove(tf)
            except:
                pass
        # Cleanup any remaining processed chunk files (in case of error/cancellation)
        if processed_chunk_paths:
            for pf in processed_chunk_paths:
                try:
                    if pf and os.path.exists(pf):
                        os.remove(pf)
                except:
                    pass
