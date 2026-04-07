"""
Shared utilities for handling audio file uploads and processing.

Eliminates duplicate patterns for:
- Reading uploaded files
- Converting to WAV
- Processing audio
- Returning responses
- Cleanup
"""

import os
import asyncio
from typing import Optional, Callable, Dict, Any
from fastapi import UploadFile, HTTPException
from fastapi.responses import Response

from util.temp_file_utils import TempFileManager, ensure_wav_format
from util.executor_utils import get_shared_executor


UPLOAD_CHUNK_SIZE = 1024 * 1024


async def _write_upload_to_path(upload_file: UploadFile, dest_path: str, chunk_size: int = UPLOAD_CHUNK_SIZE) -> int:
    """Write uploaded content to disk in chunks to avoid large memory spikes."""
    total_bytes = 0
    await upload_file.seek(0)
    with open(dest_path, "wb") as f:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            total_bytes += len(chunk)
    return total_bytes


async def process_audio_upload(
    upload_file: UploadFile,
    processor: Callable[[str, Dict[str, Any]], str],
    params: Optional[Dict[str, Any]] = None,
    convert_to_wav: bool = True,
    sample_rate: int = 44100,
    channels: int = 2,
    output_filename: str = "processed_audio.wav"
) -> Response:
    """
    Common pattern for processing uploaded audio files.
    
    Args:
        upload_file: FastAPI UploadFile object
        processor: Function that takes (audio_path, params) and returns output path
        params: Optional parameters dict to pass to processor
        convert_to_wav: Whether to convert input to WAV format
        sample_rate: Target sample rate for conversion
        channels: Target channel count for conversion
        output_filename: Filename for the response
    
    Returns:
        FastAPI Response with processed audio
    
    Raises:
        HTTPException on processing errors
    """
    manager = TempFileManager()
    params = params or {}
    
    try:
        # Save uploaded file
        file_ext = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".tmp"
        input_path = manager.create_temp_file(suffix=file_ext)
        
        await _write_upload_to_path(upload_file, input_path)
        
        # Convert to WAV if needed
        if convert_to_wav:
            wav_path = ensure_wav_format(input_path, sample_rate=sample_rate, channels=channels)
            if wav_path != input_path:
                manager.files.append(wav_path)
            audio_path = wav_path
        else:
            audio_path = input_path
        
        # Process audio (run in executor if blocking)
        executor = get_shared_executor()
        output_path = await asyncio.get_event_loop().run_in_executor(
            executor, processor, audio_path, params
        )
        manager.files.append(output_path)
        
        # Read and return result
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{output_filename}"'}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        manager.cleanup()


async def process_audio_upload_async(
    upload_file: UploadFile,
    processor: Callable[[str, Dict[str, Any]], str],
    params: Optional[Dict[str, Any]] = None,
    convert_to_wav: bool = True,
    sample_rate: int = 44100,
    channels: int = 2,
    output_filename: str = "processed_audio.wav"
) -> Response:
    """
    Same as process_audio_upload but processor is async.
    
    Args:
        upload_file: FastAPI UploadFile object
        processor: Async function that takes (audio_path, params) and returns output path
        params: Optional parameters dict to pass to processor
        convert_to_wav: Whether to convert input to WAV format
        sample_rate: Target sample rate for conversion
        channels: Target channel count for conversion
        output_filename: Filename for the response
    
    Returns:
        FastAPI Response with processed audio
    
    Raises:
        HTTPException on processing errors
    """
    manager = TempFileManager()
    params = params or {}
    
    try:
        # Save uploaded file
        file_ext = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".tmp"
        input_path = manager.create_temp_file(suffix=file_ext)
        
        await _write_upload_to_path(upload_file, input_path)
        
        # Convert to WAV if needed
        if convert_to_wav:
            wav_path = ensure_wav_format(input_path, sample_rate=sample_rate, channels=channels)
            if wav_path != input_path:
                manager.files.append(wav_path)
            audio_path = wav_path
        else:
            audio_path = input_path
        
        # Process audio (async)
        output_path = await processor(audio_path, params)
        manager.files.append(output_path)
        
        # Read and return result
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{output_filename}"'}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        manager.cleanup()

