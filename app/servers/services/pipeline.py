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
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Set
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf  # For debug logging

from util.clients import (
    get_chatterbox_client,
    get_pocket_tts_client,
    get_kokoro_tts_client,
    run_rvc,
    run_rvc_stream,
    run_postprocess,
    run_blend,
    run_save,
    run_resample,
    run_process_chunk,
)
from util.audio_utils import convert_to_format, get_mime_type, get_audio_info
from util.file_utils import resolve_audio_path
from config import get_config, is_rvc_enabled, is_post_enabled, is_background_enabled, get_bg_tracks
from servers.models.requests import TTSRequest

# Pipeline sample rate (44.1kHz CD quality)
PIPELINE_SAMPLE_RATE = 44100

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


def _get_tts_backend(request: TTSRequest) -> str:
    """Get TTS backend from request. Supports chatterbox, pocket_tts, and kokoro."""
    backend = getattr(request, 'tts_backend', 'chatterbox')
    if backend in ('chatterbox', 'pocket_tts', 'kokoro'):
        return backend
    return 'chatterbox'


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
        print(f"[{request_id}] {msg}")
   
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
        print(f"[{request_id}] Pipeline: tts_backend={tts_backend}, request.tts_backend={getattr(request, 'tts_backend', 'NOT SET')}")
       
        # Flags
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_bg = request.enable_background if request.enable_background is not None else is_background_enabled()
       
        # Resolve prompt audio (for Chatterbox) or voice (for Pocket TTS)
        prompt = None
        pocket_voice = None
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

        # === Step 1: TTS ===
        check()
        status(f"Generating TTS ({tts_backend})...")
        progress(0.1)

        if tts_backend == "pocket_tts":
            print(f"[{request_id}] Using Pocket TTS with voice={pocket_voice}")
            pocket_tts = get_pocket_tts_client()
            print(f"[{request_id}] Pocket TTS client URL: {pocket_tts.server_url}")
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
            print(f"[{request_id}] Pocket TTS generated: {tts_path}")
        elif tts_backend == "kokoro":
            print(f"[{request_id}] Using Kokoro TTS with voice={kokoro_voice}")
            kokoro_tts = get_kokoro_tts_client()
            print(f"[{request_id}] Kokoro TTS client URL: {kokoro_tts.server_url}")
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
            print(f"[{request_id}] Kokoro TTS generated: {tts_path}")
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
        print(f"[{request_id}] {msg}")
   
    try:
        config = get_config()
        executor = get_executor()
       
        do_rvc = request.enable_rvc if request.enable_rvc is not None else is_rvc_enabled()
        do_post = request.enable_post if request.enable_post is not None else is_post_enabled()
        do_bg = request.enable_background if request.enable_background is not None else is_background_enabled()
        tts_backend = _get_tts_backend(request)
       
        prompt = None
        pocket_voice = None
        kokoro_voice = None
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

                    print(f"[{request_id}] PocketTTS streaming - voice={pocket_voice}, chunks~{len(request.input)//100}")

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

                            print(f"[{request_id}] SSE connected, streaming...")

                            for line in response.iter_lines():
                                if stop_event.is_set():
                                    print(f"[{request_id}] Cancelled after {event_count} events")
                                    break

                                if not line or not line.startswith('data: '):
                                    continue

                                try:
                                    event = json.loads(line[6:])
                                    event_type = event.get('type')
                                    event_count += 1

                                    if event_type == 'start':
                                        print(f"[{request_id}] START: {event.get('chunks')} chunks")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "start",
                                            "chunks": event.get("chunks", 1),
                                            "sample_rate": event.get("sample_rate", 48000)
                                        })

                                    elif event_type == 'chunk':
                                        audio_b64 = event.get('audio_bytes_b64', '')
                                        audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b''
                                        idx = event.get('index', 0)
                                        print(f"[{request_id}] CHUNK {idx}: {len(audio_bytes)} bytes -> RVC")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "chunk",
                                            "index": idx,
                                            "audio_bytes": audio_bytes
                                        })

                                    elif event_type == 'complete':
                                        print(f"[{request_id}] COMPLETE: {event.get('chunks_sent', 0)} chunks sent")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "complete"})

                                    elif event_type == 'error':
                                        print(f"[{request_id}] ERROR: {event.get('message')}")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "error",
                                            "message": event.get("message", "Unknown error")
                                        })

                                    elif event_type == 'warning':
                                        print(f"[{request_id}] WARNING: {event.get('message')}")

                                except json.JSONDecodeError:
                                    continue

                            print(f"[{request_id}] Stream done, {event_count} events")

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

                    print(f"[{request_id}] KokoroTTS streaming - voice={kokoro_voice}")

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

                            print(f"[{request_id}] SSE connected, streaming...")

                            for line in response.iter_lines():
                                if stop_event.is_set():
                                    print(f"[{request_id}] Cancelled after {event_count} events")
                                    break

                                if not line or not line.startswith('data: '):
                                    continue

                                try:
                                    event = json.loads(line[6:])
                                    event_type = event.get('type')
                                    event_count += 1

                                    if event_type == 'start':
                                        print(f"[{request_id}] START: {event.get('chunks')} chunks")
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
                                        print(f"[{request_id}] CHUNK {idx}: {len(audio_bytes)} bytes -> RVC")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "chunk",
                                            "index": idx,
                                            "audio_bytes": audio_bytes
                                        })

                                    elif event_type == 'complete':
                                        print(f"[{request_id}] COMPLETE: {event.get('chunks_sent', 0)} chunks sent")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "complete"})

                                    elif event_type == 'error':
                                        print(f"[{request_id}] ERROR: {event.get('message')}")
                                        loop.call_soon_threadsafe(event_queue.put_nowait, {
                                            "type": "error",
                                            "message": event.get("message", "Unknown error")
                                        })

                                    elif event_type == 'warning':
                                        print(f"[{request_id}] WARNING: {event.get('message')}")

                                except json.JSONDecodeError:
                                    continue

                    print(f"[{request_id}] Stream done, {event_count} events")

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
                            print(f"[{request_id}] Reader thread: stop_event set, breaking after {event_count} events")
                            break
                       
                        event_count += 1
                        event_type = event.get("type", "unknown")
                        print(f"[{request_id}] Received event #{event_count}: type={event_type}")
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    print(f"[{request_id}] Stream iteration complete, received {event_count} events")
            except Exception as e:
                print(f"[{request_id}] Stream reader exception: {e}")
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    {"type": "error", "message": str(e)}
                )
            finally:
                print(f"[{request_id}] Reader thread finishing")
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
           
            fd, tts_path = tempfile.mkstemp(suffix="_tts_chunk.wav")
            os.close(fd)
            with open(tts_path, "wb") as f:
                f.write(audio_bytes)
            chunk_temps.append(tts_path)
           
            # Debug: log audio properties from TTS
            try:
                import soundfile as sf
                _info = sf.info(tts_path)
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: sr={_info.samplerate}, channels={_info.channels}, frames={_info.frames}, duration={_info.frames/_info.samplerate:.2f}s, bytes={len(audio_bytes)}")
            except Exception as e:
                print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index}: failed to get info: {e}")
           
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
                    print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} after RVC: sr={_info.samplerate}, frames={_info.frames}")
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
                    print(f"[PIPELINE-DEBUG] {tts_backend} chunk {chunk_index} after resample: sr=44100")
           
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
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
           
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
