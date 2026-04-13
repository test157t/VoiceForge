# clients.py
"""
Consolidated HTTP service clients for VoiceForge microservices.

This module provides client classes for communicating with:
- ASR Server (speech-to-text)
- RVC Server (voice conversion)
- Chatterbox Server (text-to-speech with voice cloning)

Also provides a ServiceProxy class for unified access to all services.

Performance optimizations:
- Shared HTTP session with connection pooling (keep-alive)
- Availability caching to avoid repeated health checks
- Configurable timeouts per service
"""
import os
import json
import tempfile
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from typing import Optional, Callable, Literal, Dict, Any


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


VF_VERBOSE_LOGS = _env_flag("VF_VERBOSE_LOGS", "0")


def _log_verbose(message: str):
    if VF_VERBOSE_LOGS:
        print(message)


def normalize_base_url(url: str) -> str:
    """Ensure a base URL always includes http:// or https://"""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url.rstrip("/")


# ============================================
# SHARED HTTP SESSION WITH CONNECTION POOLING
# ============================================

_session_lock = threading.Lock()
_shared_session: Optional[requests.Session] = None


def get_shared_session() -> requests.Session:
    """Get or create a shared requests.Session with connection pooling.
    
    Benefits:
    - Connection keep-alive (reuses TCP connections)
    - Connection pooling (up to 10 connections per host)
    - Automatic retries on connection errors
    - Thread-safe
    """
    global _shared_session
    
    if _shared_session is not None:
        return _shared_session
    
    with _session_lock:
        if _shared_session is not None:
            return _shared_session
        
        session = requests.Session()
        
        # Configure retry strategy for transient failures
        retry_strategy = Retry(
            total=2,  # Max 2 retries
            backoff_factor=0.1,  # 0.1s, 0.2s between retries
            status_forcelist=[502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "POST"],  # Retry both GET and POST
        )
        
        # Configure connection pooling adapter
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,  # Max connections per pool
            max_retries=retry_strategy,
            pool_block=False,  # Don't block when pool is full
        )
        
        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers for keep-alive
        session.headers.update({
            "Connection": "keep-alive",
        })
        
        _shared_session = session
    
    return _shared_session


def reset_shared_session():
    """Reset the shared session (useful for testing or after config changes)."""
    global _shared_session
    with _session_lock:
        if _shared_session is not None:
            try:
                _shared_session.close()
            except Exception:
                pass
            _shared_session = None


# ============================================
# SERVER CONFIGURATION
# ============================================

# Unified ASR Server (supports Whisper + GLM-ASR, routes by model name)
ASR_SERVER_URL = normalize_base_url(os.getenv("ASR_SERVER_URL", "127.0.0.1:8889"))
WHISPERASR_SERVER_URL = normalize_base_url(os.getenv("WHISPERASR_SERVER_URL", ASR_SERVER_URL))
GLMASR_SERVER_URL = normalize_base_url(os.getenv("GLMASR_SERVER_URL", ASR_SERVER_URL))
RVC_SERVER_URL = normalize_base_url(os.getenv("RVC_SERVER_URL", "127.0.0.1:8891"))
CHATTERBOX_SERVER_URL = normalize_base_url(os.getenv("CHATTERBOX_SERVER_URL", "127.0.0.1:8893"))
POCKET_TTS_SERVER_URL = normalize_base_url(os.getenv("POCKET_TTS_SERVER_URL", "127.0.0.1:8894"))
KOKORO_TTS_SERVER_URL = normalize_base_url(os.getenv("KOKORO_TTS_SERVER_URL", "127.0.0.1:8897"))
OMNIVOICE_TTS_SERVER_URL = normalize_base_url(os.getenv("OMNIVOICE_TTS_SERVER_URL", "127.0.0.1:8898"))
OMNIVOICE_ONNX_TTS_SERVER_URL = normalize_base_url(os.getenv("OMNIVOICE_ONNX_TTS_SERVER_URL", "127.0.0.1:8899"))

AUDIO_SERVICES_SERVER_URL = normalize_base_url(os.getenv("AUDIO_SERVICES_SERVER_URL", "127.0.0.1:8892"))
POSTPROCESS_SERVER_URL = normalize_base_url(os.getenv("POSTPROCESS_SERVER_URL", AUDIO_SERVICES_SERVER_URL))
PREPROCESS_SERVER_URL = normalize_base_url(os.getenv("PREPROCESS_SERVER_URL", AUDIO_SERVICES_SERVER_URL))

RVC_SERVER_TIMEOUT = int(os.getenv("RVC_SERVER_TIMEOUT", "300"))  # 5 minutes default
POSTPROCESS_SERVER_TIMEOUT = int(os.getenv("POSTPROCESS_SERVER_TIMEOUT", "120"))  # 2 minutes default
PREPROCESS_SERVER_TIMEOUT = int(os.getenv("PREPROCESS_SERVER_TIMEOUT", "300"))  # 5 minutes default


# ============================================
# BASE CLIENT
# ============================================

class BaseServiceClient:
    """Base client for HTTP microservices.
    
    Provides common functionality for service clients:
    - Health checking with connection reuse
    - Status retrieval
    - File upload helpers
    - Shared session with connection pooling
    """
    
    def __init__(self, server_url: str, health_endpoint: str = "/health"):
        """
        Initialize the client.
        
        Args:
            server_url: Base URL of the service (e.g., "http://127.0.0.1:8889")
            health_endpoint: Path to health check endpoint (default: "/health")
        """
        self.server_url = server_url
        self.health_endpoint = health_endpoint
    
    @property
    def session(self) -> requests.Session:
        """Get the shared HTTP session with connection pooling."""
        return get_shared_session()
    
    def is_available(self) -> bool:
        """Check if server is available.
        
        Returns:
            True if server responds with 200 OK on health endpoint
        """
        try:
            response = self.session.get(
                f"{self.server_url}{self.health_endpoint}", 
                timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_status(self) -> dict:
        """Get server status.
        
        Returns:
            Status dictionary from health endpoint, or error status
        """
        try:
            response = self.session.get(
                f"{self.server_url}{self.health_endpoint}", 
                timeout=2
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    def _post_with_file(
        self, 
        endpoint: str, 
        file_path: str, 
        file_key: str = "file",
        data: Optional[dict] = None, 
        timeout: int = 120,
        mime_type: str = "audio/wav"
    ) -> requests.Response:
        """POST request with file upload using shared session.
        
        Args:
            endpoint: API endpoint path (e.g., "/v1/audio/transcriptions")
            file_path: Path to file to upload
            file_key: Form field name for the file (default: "file")
            data: Additional form data to send
            timeout: Request timeout in seconds
            mime_type: MIME type of the file
        
        Returns:
            Response object from requests
            
        Raises:
            requests.exceptions.ConnectionError: If cannot connect to server
        """
        with open(file_path, "rb") as f:
            files = {file_key: (Path(file_path).name, f, mime_type)}
            return self.session.post(
                f"{self.server_url}{endpoint}",
                files=files,
                data=data or {},
                timeout=timeout
            )
    
    def _handle_connection_error(self, service_name: str, start_instruction: str) -> RuntimeError:
        """Create a user-friendly connection error.
        
        Args:
            service_name: Name of the service for error message
            start_instruction: How to start the service
            
        Returns:
            RuntimeError with helpful message
        """
        return RuntimeError(
            f"Cannot connect to {service_name} server at {self.server_url}. "
            f"{start_instruction}"
        )


# ============================================
# ASR CLIENT
# ============================================

class WhisperASRClient(BaseServiceClient):
    """Client for the Whisper ASR microservice (whisperasr_server)."""
    
    def __init__(self, server_url: str = None):
        super().__init__(server_url or WHISPERASR_SERVER_URL)
    
    def is_available(self) -> bool:
        """Check if ASR server is available and Whisper is loaded."""
        status = self.get_status()
        return status.get("whisper_available", False)
    
    def get_status(self) -> dict:
        """Get ASR server status."""
        status = super().get_status()
        # Add default values for ASR-specific fields if not present
        if "whisper_available" not in status:
            status["whisper_available"] = False
        if "model_loaded" not in status:
            status["model_loaded"] = False
        return status
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        response_format: Literal["json", "text", "verbose_json"] = "json",
        clean_vocals: bool = False,
        skip_existing_vocals: bool = True,
        postprocess_audio: bool = False,
        device: str = "gpu",
        model: str = None
    ) -> dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            response_format: Response format (json, text, verbose_json)
            clean_vocals: If True, use UVR5 to remove background music/noise before transcription
            skip_existing_vocals: If True, reuse cached vocals if available
            postprocess_audio: If True, apply audio enhancement before transcription
            device: "gpu" or "cpu"
            model: ASR model name
        
        Returns:
            Transcription result dict with "text" key
        
        Raises:
            RuntimeError: If transcription fails
        """
        try:
            data = {
                "language": language,
                "response_format": response_format,
                "clean_vocals": str(clean_vocals).lower(),
                "skip_existing_vocals": str(skip_existing_vocals).lower(),
                "postprocess_audio": str(postprocess_audio).lower(),
                "device": device
            }
            if model:
                data["model"] = model
            
            response = self._post_with_file(
                endpoint="/v1/audio/transcriptions",
                file_path=audio_path,
                file_key="file",
                data=data,
                timeout=300  # Longer timeout for long audio
            )
            
            # Handle specific error codes
            if response.status_code == 507:
                # GPU OOM error - include helpful message
                try:
                    detail = response.json().get("detail", response.text)
                except:
                    detail = response.text
                raise RuntimeError(f"GPU out of memory: {detail}")
            
            if response.status_code != 200:
                raise RuntimeError(f"ASR server error: {response.status_code} - {response.text}")
            
            if response_format == "text":
                return {"text": response.text}
            return response.json()
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "ASR",
                "Please start the ASR server using option [4] in the launcher."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"ASR transcription failed: {e}")
    
    def transcribe_stream(
        self,
        audio_path: str,
        language: str = "en",
        clean_vocals: bool = False
    ):
        """
        Streaming transcription - yields updates as Server-Sent Events.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            clean_vocals: If True, use UVR5 to remove background music/noise
        
        Yields:
            dict: Event updates with type, message, progress, text etc.
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'file': (os.path.basename(audio_path), f)}
                data = {
                    'language': language,
                    'clean_vocals': str(clean_vocals).lower()
                }
                
                response = self.session.post(
                    f"{self.server_url}/v1/audio/transcriptions/stream",
                    files=files,
                    data=data,
                    stream=True,
                    timeout=300
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"ASR server error: {response.status_code}")
                
                # Parse SSE stream
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])
                                yield event_data
                            except json.JSONDecodeError:
                                continue
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "ASR",
                "Please start the ASR server using option [4] in the launcher."
            )
        except Exception as e:
            raise RuntimeError(f"ASR streaming transcription failed: {e}")
    
    def get_gpu_memory(self) -> dict:
        """Get GPU memory usage from ASR server."""
        try:
            response = self.session.get(f"{self.server_url}/gpu_memory", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"available": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"available": False, "error": "ASR server not running"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def clear_gpu_cache(self) -> dict:
        """Clear GPU memory cache on ASR server."""
        try:
            response = self.session.post(f"{self.server_url}/clear_cache", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "ASR server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def unload(self) -> dict:
        """Unload the ASR model to free GPU memory."""
        try:
            response = self.session.post(f"{self.server_url}/unload", timeout=30)
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "ASR server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_model_info(self) -> dict:
        """Get information about the loaded ASR model."""
        try:
            response = self.session.get(f"{self.server_url}/model_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"loaded": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"loaded": False, "error": "ASR server not running"}
        except Exception as e:
            return {"loaded": False, "error": str(e)}


# ============================================
# GLM-ASR CLIENT
# ============================================

class GLMASRClient(BaseServiceClient):
    """Client for the GLM-ASR microservice (glmasr_server).
    
    GLM-ASR-Nano-2512 is a 1.5B parameter ASR model that:
    - Outperforms Whisper V3 on multiple benchmarks
    - Has excellent dialect support (Mandarin, Cantonese, English)
    - Excels at low-volume/whisper speech recognition
    """
    
    def __init__(self, server_url: str = None):
        super().__init__(server_url or GLMASR_SERVER_URL)
    
    def is_available(self) -> bool:
        """Check if GLM-ASR server is available."""
        status = self.get_status()
        # Unified ASR server can expose GLM backends in multiple forms
        return (
            status.get("whisper_available", False)
            or status.get("glm_asr_available", False)
        )
    
    def get_status(self) -> dict:
        """Get GLM-ASR server status."""
        status = super().get_status()
        # Add default values for ASR-specific fields if not present
        if "glm_asr_available" not in status:
            status["glm_asr_available"] = False
        if "glm_model_loaded" not in status:
            status["glm_model_loaded"] = False
        return status
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        response_format: Literal["json", "text", "verbose_json"] = "json",
        clean_vocals: bool = False,
        skip_existing_vocals: bool = True,
        postprocess_audio: bool = False,
        device: str = "gpu",
        model: str = None
    ) -> dict:
        """
        Transcribe audio file using GLM-ASR.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            response_format: Response format (json, text, verbose_json)
            clean_vocals: If True, use UVR5 to remove background music/noise before transcription
            skip_existing_vocals: If True, reuse cached vocals if available
            postprocess_audio: If True, apply audio enhancement before transcription
            device: "gpu" or "cpu"
            model: ASR model name (ignored for GLM-ASR, only one model available)
        
        Returns:
            Transcription result dict with "text" key
        
        Raises:
            RuntimeError: If transcription fails
        """
        try:
            data = {
                "language": language,
                "response_format": response_format,
                "clean_vocals": str(clean_vocals).lower(),
                "skip_existing_vocals": str(skip_existing_vocals).lower(),
                "postprocess_audio": str(postprocess_audio).lower(),
                "device": device
            }
            if model:
                data["model"] = model
            
            response = self._post_with_file(
                endpoint="/v1/audio/transcriptions",
                file_path=audio_path,
                file_key="file",
                data=data,
                timeout=300  # Longer timeout for long audio
            )
            
            # Handle specific error codes
            if response.status_code == 507:
                # GPU OOM error - include helpful message
                try:
                    detail = response.json().get("detail", response.text)
                except:
                    detail = response.text
                raise RuntimeError(f"GPU out of memory: {detail}")
            
            if response.status_code != 200:
                raise RuntimeError(f"GLM-ASR server error: {response.status_code} - {response.text}")
            
            if response_format == "text":
                return {"text": response.text}
            return response.json()
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "GLM-ASR",
                "Please start the GLM-ASR server using the launch_glmasr_server.bat script."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"GLM-ASR transcription failed: {e}")
    
    def transcribe_stream(
        self,
        audio_path: str,
        language: str = "en",
        clean_vocals: bool = False
    ):
        """
        Streaming transcription - yields updates as Server-Sent Events.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            clean_vocals: If True, use UVR5 to remove background music/noise
        
        Yields:
            dict: Event updates with type, message, progress, text etc.
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'file': (os.path.basename(audio_path), f)}
                data = {
                    'language': language,
                    'clean_vocals': str(clean_vocals).lower()
                }
                
                response = self.session.post(
                    f"{self.server_url}/v1/audio/transcriptions/stream",
                    files=files,
                    data=data,
                    stream=True,
                    timeout=300
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"GLM-ASR server error: {response.status_code}")
                
                # Parse SSE stream
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])
                                yield event_data
                            except json.JSONDecodeError:
                                continue
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "GLM-ASR",
                "Please start the GLM-ASR server using the launch_glmasr_server.bat script."
            )
        except Exception as e:
            raise RuntimeError(f"GLM-ASR streaming transcription failed: {e}")
    
    def get_gpu_memory(self) -> dict:
        """Get GPU memory usage from GLM-ASR server."""
        try:
            response = self.session.get(f"{self.server_url}/gpu_memory", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"available": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"available": False, "error": "GLM-ASR server not running"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def clear_gpu_cache(self) -> dict:
        """Clear GPU memory cache on GLM-ASR server."""
        try:
            response = self.session.post(f"{self.server_url}/clear_cache", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "GLM-ASR server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def unload(self) -> dict:
        """Unload the GLM-ASR model to free GPU memory."""
        try:
            response = self.session.post(f"{self.server_url}/unload", timeout=30)
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "GLM-ASR server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_model_info(self) -> dict:
        """Get information about the loaded GLM-ASR model."""
        try:
            response = self.session.get(f"{self.server_url}/model_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"loaded": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"loaded": False, "error": "GLM-ASR server not running"}
        except Exception as e:
            return {"loaded": False, "error": str(e)}


# ============================================
# RVC CLIENT
# ============================================

class RVCClient(BaseServiceClient):
    """Client for the RVC voice conversion microservice."""
    
    def __init__(self, server_url: str = None, timeout: int = None):
        super().__init__(server_url or RVC_SERVER_URL)
        self.timeout = timeout or RVC_SERVER_TIMEOUT
        self._server_available = None  # Cache for availability check
    
    def is_available(self) -> bool:
        """Check if RVC server is available (with caching).
        
        Only caches positive results - if server is down, will re-check next time.
        """
        if self._server_available:
            return True
        
        result = super().is_available()
        if result:
            self._server_available = True
        return result
    
    def reset_availability_cache(self):
        """Reset the availability cache (call after server might have stopped)."""
        self._server_available = None
    
    def get_status(self) -> dict:
        """Get RVC server status."""
        status = super().get_status()
        if "workers" not in status:
            status["workers"] = {"configured": 0, "loaded": 0}
        return status
    
    def load_model(self, model_name: str) -> dict:
        """Pre-load an RVC model to warm up the worker.
        
        This triggers model loading without doing any conversion,
        so subsequent conversions will be faster.
        
        Args:
            model_name: Name of the RVC model to load
            
        Returns:
            dict with status info
        """
        try:
            response = self.session.post(
                f"{self.server_url}/load_model",
                data={"model_name": model_name},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def convert(
        self,
        audio_path: str,
        model_name: str,
        pitch_algo: str = None,  # None = use server config
        pitch_lvl: int = None,
        index_influence: float = None,
        respiration_median_filtering: int = None,
        envelope_ratio: float = None,
        consonant_breath_protection: float = None
    ) -> str:
        """
        Convert audio using RVC voice conversion.
        
        Args:
            audio_path: Path to input audio file
            model_name: Name of the RVC model to use
            pitch_algo: Pitch detection algorithm (rmvpe, pm, harvest)
            pitch_lvl: Pitch shift in semitones
            index_influence: Index file influence (0.0-1.0)
            respiration_median_filtering: Median filter for breathing sounds
            envelope_ratio: Volume envelope ratio
            consonant_breath_protection: Protect consonants and breaths
        
        Returns:
            Path to output WAV file
        
        Raises:
            RuntimeError: If conversion fails
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                # Only include non-None params - server will use config for missing ones
                data = {'model_name': model_name}
                if pitch_algo is not None:
                    data['pitch_algo'] = pitch_algo
                if pitch_lvl is not None:
                    data['pitch_lvl'] = pitch_lvl
                if index_influence is not None:
                    data['index_influence'] = index_influence
                if respiration_median_filtering is not None:
                    data['respiration_median_filtering'] = respiration_median_filtering
                if envelope_ratio is not None:
                    data['envelope_ratio'] = envelope_ratio
                if consonant_breath_protection is not None:
                    data['consonant_breath_protection'] = consonant_breath_protection
                
                response = self.session.post(
                    f"{self.server_url}/v1/rvc",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"RVC server error: {response.status_code} - {response.text}")
            
            # Save the response audio
            fd, tmp = tempfile.mkstemp(suffix="_rvc.wav")
            os.close(fd)
            with open(tmp, 'wb') as f:
                f.write(response.content)
            
            return tmp
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "RVC",
                "Please start the RVC server using option [3] in the launcher."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"RVC conversion failed: {e}")
    
    def convert_chunked(
        self,
        audio_path: str,
        model_name: str,
        pitch_algo: str = None,  # None = use server config
        pitch_lvl: int = None,
        index_influence: float = None,
        respiration_median_filtering: int = None,
        envelope_ratio: float = None,
        consonant_breath_protection: float = None,
        chunk_duration: int = 60,
        request_id: str = None
    ) -> str:
        """
        Convert audio using chunked processing (better for long audio).
        
        Args:
            audio_path: Path to input audio file
            model_name: Name of the RVC model to use
            chunk_duration: Duration of each chunk in seconds
            request_id: Optional request ID for unified logging
            (other args same as convert())
        
        Returns:
            Path to output WAV file
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                # Only include non-None params - server will use config for missing ones
                data = {'model_name': model_name, 'chunk_duration': chunk_duration}
                if pitch_algo is not None:
                    data['pitch_algo'] = pitch_algo
                if pitch_lvl is not None:
                    data['pitch_lvl'] = pitch_lvl
                if index_influence is not None:
                    data['index_influence'] = index_influence
                if respiration_median_filtering is not None:
                    data['respiration_median_filtering'] = respiration_median_filtering
                if envelope_ratio is not None:
                    data['envelope_ratio'] = envelope_ratio
                if consonant_breath_protection is not None:
                    data['consonant_breath_protection'] = consonant_breath_protection
                if request_id:
                    data['request_id'] = request_id
                
                response = self.session.post(
                    f"{self.server_url}/v1/rvc/chunked",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"RVC server error: {response.status_code} - {response.text}")
            
            fd, tmp = tempfile.mkstemp(suffix="_rvc.wav")
            os.close(fd)
            with open(tmp, 'wb') as f:
                f.write(response.content)
            
            return tmp
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "RVC",
                "Please start the RVC server using option [3] in the launcher."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"RVC chunked conversion failed: {e}")
    
    def convert_stream(
        self,
        audio_path: str,
        model_name: str,
        pitch_algo: str = None,
        pitch_lvl: int = None,
        index_influence: float = None,
        respiration_median_filtering: int = None,
        envelope_ratio: float = None,
        consonant_breath_protection: float = None,
        chunk_duration: int = 60,
        on_chunk: callable = None,
        on_progress: callable = None,
        request_id: str = None
    ) -> str:
        """
        Convert audio using streaming - chunks are returned via SSE as they're processed.
        
        Args:
            audio_path: Path to input audio file
            model_name: Name of the RVC model to use
            chunk_duration: Duration of each chunk in seconds
            on_chunk: Callback(audio_bytes, chunk_index, total_chunks) for each chunk
            on_progress: Callback(progress_float) for progress updates
            request_id: Optional request ID for unified logging
            (other args same as convert())
        
        Returns:
            Path to combined output WAV file (all chunks concatenated)
        """
        import base64
        import json
        
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                data = {'model_name': model_name, 'chunk_duration': chunk_duration}
                if pitch_algo is not None:
                    data['pitch_algo'] = pitch_algo
                if pitch_lvl is not None:
                    data['pitch_lvl'] = pitch_lvl
                if index_influence is not None:
                    data['index_influence'] = index_influence
                if respiration_median_filtering is not None:
                    data['respiration_median_filtering'] = respiration_median_filtering
                if envelope_ratio is not None:
                    data['envelope_ratio'] = envelope_ratio
                if consonant_breath_protection is not None:
                    data['consonant_breath_protection'] = consonant_breath_protection
                if request_id:
                    data['request_id'] = request_id
                
                response = self.session.post(
                    f"{self.server_url}/v1/rvc/stream",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                    stream=True
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"RVC server error: {response.status_code} - {response.text}")
            
            # Process SSE events - collect RAW WAV bytes
            wav_chunks = []
            total_chunks = 1
            sample_rate = None
            
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue
                
                try:
                    event = json.loads(line[6:])
                    
                    if event['type'] == 'start':
                        total_chunks = event.get('chunks', 1)
                        sample_rate = event.get('sample_rate', 48000)
                    
                    elif event['type'] == 'chunk':
                        audio_bytes = base64.b64decode(event.get('audio', ''))
                        wav_chunks.append(audio_bytes)
                        
                        if on_chunk:
                            on_chunk(audio_bytes, event.get('index', 0), total_chunks)
                        if on_progress:
                            on_progress((event.get('index', 0) + 1) / total_chunks)
                    
                    elif event['type'] == 'error':
                        raise RuntimeError(f"RVC stream error: {event.get('message', 'Unknown error')}")
                    
                    elif event['type'] == 'complete':
                        break
                        
                except json.JSONDecodeError:
                    continue
            
            if not wav_chunks:
                raise RuntimeError("No audio chunks received from RVC stream")
            
            # Save output
            fd, output_path = tempfile.mkstemp(suffix="_rvc_stream.wav")
            os.close(fd)
            
            if len(wav_chunks) == 1:
                with open(output_path, 'wb') as f:
                    f.write(wav_chunks[0])
            else:
                import soundfile as sf
                import numpy as np
                import io
                
                all_audio = []
                for wav_bytes in wav_chunks:
                    audio_data, sr = sf.read(io.BytesIO(wav_bytes))
                    all_audio.append(audio_data)
                    if sample_rate is None:
                        sample_rate = sr
                
                sf.write(output_path, np.concatenate(all_audio), sample_rate)
            
            return output_path
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "RVC",
                "Please start the RVC server using option [3] in the launcher."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"RVC streaming conversion failed: {e}")

    def convert_stream_iter(
        self,
        audio_path: str,
        model_name: str,
        pitch_algo: str = None,
        pitch_lvl: int = None,
        index_influence: float = None,
        respiration_median_filtering: int = None,
        envelope_ratio: float = None,
        consonant_breath_protection: float = None,
        chunk_duration: int = 60,
        request_id: str = None
    ):
        """
        Convert audio using streaming - yields each chunk as it's processed.
        
        Yields:
            dict with keys: type, index, total, audio_bytes, sample_rate
        """
        import base64
        import json
        
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                data = {'model_name': model_name, 'chunk_duration': chunk_duration}
                if pitch_algo is not None:
                    data['pitch_algo'] = pitch_algo
                if pitch_lvl is not None:
                    data['pitch_lvl'] = pitch_lvl
                if index_influence is not None:
                    data['index_influence'] = index_influence
                if respiration_median_filtering is not None:
                    data['respiration_median_filtering'] = respiration_median_filtering
                if envelope_ratio is not None:
                    data['envelope_ratio'] = envelope_ratio
                if consonant_breath_protection is not None:
                    data['consonant_breath_protection'] = consonant_breath_protection
                if request_id:
                    data['request_id'] = request_id
                
                response = self.session.post(
                    f"{self.server_url}/v1/rvc/stream",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                    stream=True
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"RVC server error: {response.status_code} - {response.text}")
            
            total_chunks = 1
            sample_rate = 48000
            
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue
                
                try:
                    event = json.loads(line[6:])
                    
                    if event['type'] == 'start':
                        total_chunks = event.get('chunks', 1)
                        sample_rate = event.get('sample_rate', 48000)
                        yield {'type': 'start', 'chunks': total_chunks, 'sample_rate': sample_rate}
                    
                    elif event['type'] == 'chunk':
                        audio_bytes = base64.b64decode(event.get('audio', ''))
                        yield {
                            'type': 'chunk',
                            'index': event.get('index', 0),
                            'total': total_chunks,
                            'audio_bytes': audio_bytes,
                            'sample_rate': sample_rate
                        }
                    
                    elif event['type'] == 'error':
                        yield {'type': 'error', 'message': event.get('message', 'Unknown error')}
                        return
                    
                    elif event['type'] == 'complete':
                        yield {'type': 'complete'}
                        return
                        
                except json.JSONDecodeError:
                    continue
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "RVC",
                "Please start the RVC server using option [3] in the launcher."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"RVC streaming conversion failed: {e}")
    
    def get_models(self) -> dict:
        """Get list of available RVC models."""
        try:
            response = self.session.get(f"{self.server_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"models": [], "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"models": [], "error": "RVC server not running"}
        except Exception as e:
            return {"models": [], "error": str(e)}
    
    def load_model(self, model_name: str) -> dict:
        """Pre-load an RVC model to reduce first conversion latency."""
        try:
            response = self.session.post(
                f"{self.server_url}/load_model",
                data={'model_name': model_name},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "RVC server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def unload(self) -> dict:
        """Unload RVC models to free GPU memory."""
        try:
            response = self.session.post(f"{self.server_url}/unload", timeout=30)
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "RVC server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_gpu_memory(self) -> dict:
        """Get GPU memory usage from RVC server."""
        try:
            response = self.session.get(f"{self.server_url}/gpu_memory", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"available": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"available": False, "error": "RVC server not running"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def get_model_info(self) -> dict:
        """Get information about loaded RVC models."""
        try:
            response = self.session.get(f"{self.server_url}/model_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"loaded": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"loaded": False, "error": "RVC server not running"}
        except Exception as e:
            return {"loaded": False, "error": str(e)}
    
    def get_workers(self) -> dict:
        """Get worker pool status."""
        try:
            response = self.session.get(f"{self.server_url}/workers", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"num_workers": 0, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"num_workers": 0, "error": "RVC server not running"}
        except Exception as e:
            return {"num_workers": 0, "error": str(e)}
    
    def set_workers(self, num_workers: int) -> dict:
        """Set number of RVC workers."""
        try:
            response = self.session.post(
                f"{self.server_url}/workers",
                data={'num_workers': num_workers},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "RVC server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================
# POSTPROCESS CLIENT
# ============================================

class PostProcessClient(BaseServiceClient):
    """Client for the post-processing microservice."""
    
    def __init__(self, server_url: str = None, timeout: int = None):
        super().__init__(server_url or POSTPROCESS_SERVER_URL)
        self.timeout = timeout or POSTPROCESS_SERVER_TIMEOUT
        self._server_available = None  # Cache for availability check
    
    def is_available(self) -> bool:
        """Check if post-processing server is available (with caching).
        
        Only caches positive results - if server is down, will re-check next time.
        """
        if self._server_available:
            return True
        
        result = super().is_available()
        if result:
            self._server_available = True
        return result
    
    def reset_availability_cache(self):
        """Reset the availability cache (call after server might have stopped)."""
        self._server_available = None
    
    def postprocess(
        self,
        audio_path: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply post-processing effects to audio.
        
        Args:
            audio_path: Path to input audio file
            params: Post-processing parameters dict
        
        Returns:
            Path to processed audio file
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                
                response = self.session.post(
                    f"{self.server_url}/v1/postprocess",
                    files=files,
                    data=params,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"Postprocess server error: {response.status_code} - {response.text}")
            
            fd, tmp = tempfile.mkstemp(suffix="_post.wav")
            os.close(fd)
            with open(tmp, 'wb') as f:
                f.write(response.content)
            
            return tmp
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Post-processing server is not available. "
                "Please start the post-processing server."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Post-processing failed: {e}")
    
    def blend(
        self,
        audio_path: str,
        bg_files: list,
        bg_volumes: list,
        main_volume: float = 1.0,
        bg_delays: list = None,
        bg_fade_ins: list = None,
        bg_fade_outs: list = None
    ) -> str:
        """
        Blend main audio with background tracks.
        
        Args:
            audio_path: Path to main audio file
            bg_files: List of background file paths
            bg_volumes: List of background volumes
            main_volume: Main audio volume
            bg_delays: List of delays in seconds for each background track
            bg_fade_ins: List of fade in durations in seconds for each background track
            bg_fade_outs: List of fade out durations in seconds for each background track
        
        Returns:
            Path to blended audio file
        """
        import json
        
        if bg_delays is None:
            bg_delays = [0.0] * len(bg_files)
        if bg_fade_ins is None:
            bg_fade_ins = [0.0] * len(bg_files)
        if bg_fade_outs is None:
            bg_fade_outs = [0.0] * len(bg_files)
        
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                data = {
                    'bg_files': json.dumps(bg_files),
                    'bg_volumes': json.dumps(bg_volumes),
                    'main_volume': main_volume,
                    'bg_delays': json.dumps(bg_delays),
                    'bg_fade_ins': json.dumps(bg_fade_ins),
                    'bg_fade_outs': json.dumps(bg_fade_outs)
                }
                
                response = self.session.post(
                    f"{self.server_url}/v1/blend",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"Blend server error: {response.status_code} - {response.text}")
            
            fd, tmp = tempfile.mkstemp(suffix="_blend.wav")
            os.close(fd)
            with open(tmp, 'wb') as f:
                f.write(response.content)
            
            return tmp
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Post-processing server is not available.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Blend failed: {e}")
    
    def master(self, audio_path: str) -> str:
        """
        Apply final mastering to audio.
        
        Args:
            audio_path: Path to input audio file
        
        Returns:
            Path to mastered audio file
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                
                response = self.session.post(
                    f"{self.server_url}/v1/master",
                    files=files,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"Master server error: {response.status_code} - {response.text}")
            
            fd, tmp = tempfile.mkstemp(suffix="_master.wav")
            os.close(fd)
            with open(tmp, 'wb') as f:
                f.write(response.content)
            
            return tmp
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Post-processing server is not available.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Mastering failed: {e}")
    
    def save(self, audio_path: str, text: str = "output") -> str:
        """
        Save audio to output directory on server.
        
        Args:
            audio_path: Path to input audio file
            text: Text to use for filename generation
        
        Returns:
            Path where file was saved
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                data = {'text': text}
                
                response = self.session.post(
                    f"{self.server_url}/v1/save",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"Save server error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result.get("path", audio_path)
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Post-processing server is not available.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Save failed: {e}")
    
    def resample(
        self,
        audio_path: str,
        sample_rate: int = 44100,
        volume: float = 1.0
    ) -> str:
        """
        Resample audio to target sample rate and optionally adjust volume.
        
        Args:
            audio_path: Path to input audio file
            sample_rate: Target sample rate (default 44100)
            volume: Volume multiplier (1.0 = no change)
        
        Returns:
            Path to resampled audio file
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                data = {
                    'sample_rate': sample_rate,
                    'volume': volume
                }
                
                response = self.session.post(
                    f"{self.server_url}/v1/resample",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"Resample server error: {response.status_code} - {response.text}")
            
            fd, tmp = tempfile.mkstemp(suffix="_resampled.wav")
            os.close(fd)
            with open(tmp, 'wb') as f:
                f.write(response.content)
            
            return tmp
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Post-processing server is not available.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Resample failed: {e}")
    
    def process_chunk(
        self,
        audio_path: str,
        post_params: Optional[Dict[str, Any]] = None,
        target_sample_rate: int = 44100,
        output_volume: float = 1.0
    ) -> str:
        """
        Combined PostProcess + Resample in one HTTP call (optimized for streaming).
        
        Args:
            audio_path: Path to input audio file
            post_params: Post-processing parameters dict (None = skip post-process)
            target_sample_rate: Target sample rate for output
            output_volume: Volume multiplier for output
        
        Returns:
            Path to processed audio file
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('input.wav', f, 'audio/wav')}
                data = {
                    'post_params_json': json.dumps(post_params) if post_params else '{}',
                    'target_sample_rate': target_sample_rate,
                    'output_volume': output_volume,
                }
                
                response = self.session.post(
                    f"{self.server_url}/v1/process-chunk",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"Process-chunk server error: {response.status_code} - {response.text}")
            
            fd, tmp = tempfile.mkstemp(suffix="_processed.wav")
            os.close(fd)
            with open(tmp, 'wb') as f:
                f.write(response.content)
            
            return tmp
        
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Post-processing server is not available.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Process-chunk failed: {e}")


# ============================================
# PREPROCESS CLIENT
# ============================================

class PreprocessClient(BaseServiceClient):
    """Client for the preprocessing microservice (UVR5, etc.)."""

    def __init__(self, server_url: str = None, timeout: int = None):
        super().__init__(server_url or PREPROCESS_SERVER_URL)
        self.timeout = timeout or PREPROCESS_SERVER_TIMEOUT

    def clean_vocals(
        self,
        audio_path: str,
        aggression: int = 10,
        device: Optional[str] = None,
        skip_if_cached: bool = True,
        original_filename: str = None,
        model_key: str = "hp5_vocals",
    ) -> str:
        """
        Process audio using UVR5 models (server-side).

        Args:
            audio_path: Path to audio file
            aggression: UVR5 aggression level (0-20)
            device: "cuda" or "cpu"
            skip_if_cached: If True, reuse cached result if available
            original_filename: Original filename for cache key (uses basename if not provided)
            model_key: UVR5 model to use (hp5_vocals, deecho_normal, deecho_aggressive, deecho_dereverb)

        Returns:
            Path to processed WAV file (temp file).
        """
        try:
            # Use original filename or extract from path
            filename = original_filename or os.path.basename(audio_path)
            
            with open(audio_path, "rb") as f:
                files = {"audio": (filename, f, "audio/wav")}
                data = {
                    "aggression": str(int(aggression)),
                    "skip_if_cached": str(skip_if_cached).lower(),
                    "model_key": model_key,
                }
                if device:
                    data["device"] = device

                response = self.session.post(
                    f"{self.server_url}/v1/preprocess/uvr5/clean-vocals",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )

            if response.status_code != 200:
                raise RuntimeError(f"Preprocess server error: {response.status_code} - {response.text}")

            fd, tmp = tempfile.mkstemp(suffix="_vocals.wav")
            os.close(fd)
            with open(tmp, "wb") as out:
                out.write(response.content)
            return tmp

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Preprocess server at {self.server_url}. "
                "Please start the preprocess server."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"UVR5 clean vocals failed: {e}")



# ============================================
# CHATTERBOX CLIENT (Resemble AI TTS)
# ============================================

class ChatterboxClient(BaseServiceClient):
    """Client for the Chatterbox-Turbo TTS server (Resemble AI).
    
    Chatterbox-Turbo is a 350M parameter TTS model that supports:
    - Zero-shot voice cloning
    - Paralinguistic tags ([laugh], [chuckle], [cough], etc.)
    - Low latency generation
    """
    
    # Supported paralinguistic tags
    PARALINGUISTIC_TAGS = [
        "[laugh]", "[chuckle]", "[cough]", "[sigh]", 
        "[gasp]", "[groan]", "[yawn]", "[clear throat]"
    ]
    
    def __init__(self, server_url: str = None):
        super().__init__(server_url or CHATTERBOX_SERVER_URL)
        self._server_available = None
    
    def is_available(self) -> bool:
        """Check if Chatterbox server is available (with caching)."""
        if self._server_available:
            return True
        
        result = super().is_available()
        if result:
            self._server_available = True
        return result
    
    def reset_availability_cache(self):
        """Reset the availability cache."""
        self._server_available = None
    
    def get_status(self) -> dict:
        """Get server status."""
        status = super().get_status()
        if "model_loaded" not in status:
            status["model_loaded"] = False
        return status
    
    def warmup(self, prompt_audio_path: str = None) -> dict:
        """Trigger model warmup on server.
        
        Args:
            prompt_audio_path: Optional path to audio prompt for warmup.
                              Server will use config prompt if not provided.
        
        Returns:
            dict with status and message
        """
        try:
            payload = {}
            if prompt_audio_path:
                payload["prompt_audio_path"] = prompt_audio_path
            
            response = self.session.post(
                f"{self.server_url}/warmup",
                json=payload if payload else None,
                timeout=120
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "message": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def unload(self) -> dict:
        """Unload Chatterbox model to free GPU memory."""
        try:
            response = self.session.post(f"{self.server_url}/unload", timeout=30)
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Chatterbox server not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_model_info(self) -> dict:
        """Get information about the loaded Chatterbox model."""
        try:
            response = self.session.get(f"{self.server_url}/model_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"loaded": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"loaded": False, "error": "Chatterbox server not running"}
        except Exception as e:
            return {"loaded": False, "error": str(e)}
    
    def get_gpu_memory(self) -> dict:
        """Get GPU memory usage from Chatterbox server."""
        try:
            response = self.session.get(f"{self.server_url}/gpu_memory", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"available": False, "error": response.text}
        except requests.exceptions.ConnectionError:
            return {"available": False, "error": "Chatterbox server not running"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def generate(
        self,
        text: str,
        prompt_audio_path: str,
        seed: int = 0,
        max_tokens: int = 200,
        request_id: str = None
    ) -> str:
        """
        Generate TTS audio with voice cloning using Chatterbox-Turbo.
        
        Args:
            text: Text to synthesize. Supports paralinguistic tags like [laugh], [chuckle], etc.
            prompt_audio_path: Path to reference audio file (5+ seconds required, 10+ recommended)
            seed: Random seed for reproducibility (0 = random)
            max_tokens: Max tokens parameter (passed to server for compatibility)
            request_id: Optional request ID for logging/cancellation tracking
        
        Note: Chatterbox-Turbo doesn't support exaggeration/cfg_weight controls.
        
        Returns:
            Path to output WAV file
        
        Raises:
            RuntimeError: If generation fails
        """
        try:
            data = {
                "text": text,
                "seed": seed,
                "max_tokens": max_tokens
            }
            if request_id:
                data["request_id"] = request_id
            
            response = self._post_with_file(
                endpoint="/v1/tts/chunked",
                file_path=prompt_audio_path,
                file_key="prompt_audio",
                data=data,
                timeout=1800  # 30 minute timeout for long content
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Chatterbox server error: {response.status_code} - {response.text}")
            
            # Save response audio to temp file
            fd, output_path = tempfile.mkstemp(suffix="_chatterbox.wav")
            os.close(fd)
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "Chatterbox",
                "Please start the Chatterbox server using option [3] in the launcher."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Chatterbox generation failed: {e}")
    
    def generate_stream(
        self,
        text: str,
        prompt_audio_path: str,
        seed: int = 0,
        max_tokens: int = 200,
        on_chunk: callable = None,
        on_progress: callable = None,
        request_id: str = None
    ) -> str:
        """
        Streaming TTS - generates audio in chunks and streams them via SSE.
        
        Each chunk is generated and yielded immediately for real-time playback.
        
        Args:
            text: Text to synthesize
            prompt_audio_path: Path to reference audio file (5+ seconds required)
            seed: Random seed for reproducibility (0 = random)
            max_tokens: Max tokens per chunk
            on_chunk: Callback(audio_bytes, chunk_index, total_chunks) called for each chunk
            on_progress: Callback(progress_0_to_1) called for progress updates
            request_id: Optional request ID for logging/cancellation tracking
        
        Returns:
            Path to combined output WAV file (all chunks concatenated)
        
        Raises:
            RuntimeError: If generation fails
        """
        import base64
        import json
        
        try:
            with open(prompt_audio_path, "rb") as f:
                files = {"prompt_audio": (os.path.basename(prompt_audio_path), f, "audio/wav")}
                form_data = {
                    "text": text,
                    "seed": seed,
                    "max_tokens": max_tokens,
                }
                if request_id:
                    form_data["request_id"] = request_id
                
                response = self.session.post(
                    f"{self.server_url}/v1/tts/stream",
                    files=files,
                    data=form_data,
                    stream=True,
                    timeout=1800
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Chatterbox stream error: {response.status_code} - {response.text}")
                
                # Collect RAW WAV bytes
                wav_chunks = []
                total_chunks = 1
                sample_rate = None
                stream_complete = False
                
                # Parse SSE stream
                buffer = ""
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if stream_complete:
                        break  # Exit immediately after complete event
                    if chunk:
                        buffer += chunk
                        while "\n\n" in buffer:
                            event, buffer = buffer.split("\n\n", 1)
                            if event.startswith("data: "):
                                try:
                                    event_data = json.loads(event[6:])
                                    event_type = event_data.get("type")
                                    
                                    if event_type == "start":
                                        total_chunks = event_data.get("chunks", 1)
                                        sample_rate = event_data.get("sample_rate", 24000)
                                    
                                    elif event_type == "chunk":
                                        audio_b64 = event_data.get("audio")
                                        chunk_idx = event_data.get("index", 0)
                                        
                                        if audio_b64:
                                            wav_chunks.append(base64.b64decode(audio_b64))
                                            
                                            if on_chunk:
                                                on_chunk(wav_chunks[-1], chunk_idx, total_chunks)
                                            if on_progress:
                                                on_progress((chunk_idx + 1) / total_chunks)
                                    
                                    elif event_type == "complete":
                                        stream_complete = True
                                        break  # Exit inner loop immediately
                                    
                                    elif event_type == "error":
                                        raise RuntimeError(f"Stream error: {event_data.get('message')}")
                                    
                                except json.JSONDecodeError:
                                    pass
                
                if not wav_chunks:
                    raise RuntimeError("No audio chunks received from stream")
                
                # Save output
                fd, output_path = tempfile.mkstemp(suffix="_chatterbox_stream.wav")
                os.close(fd)
                
                if len(wav_chunks) == 1:
                    with open(output_path, 'wb') as f:
                        f.write(wav_chunks[0])
                else:
                    import soundfile as sf
                    import numpy as np
                    import io
                    
                    all_audio = []
                    for wav_bytes in wav_chunks:
                        audio_data, sr = sf.read(io.BytesIO(wav_bytes))
                        all_audio.append(audio_data)
                        if sample_rate is None:
                            sample_rate = sr
                    
                    sf.write(output_path, np.concatenate(all_audio), sample_rate)
                
                return output_path
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "Chatterbox",
                "Please start the Chatterbox server using option [3] in the launcher."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Chatterbox streaming failed: {e}")

    def stream_events(
        self,
        text: str,
        prompt_audio_path: str,
        seed: int = 0,
        max_tokens: int = 200,
        request_id: str = None,
        stop_event: Optional[threading.Event] = None
    ):
        """
        Stream Chatterbox SSE events as they arrive.
        
        Yields:
            dict events with:
            - type: "start" | "chunk" | "complete"
            - audio_bytes (for chunk events)
        """
        import base64
        import json
        
        response = None
        try:
            with open(prompt_audio_path, "rb") as f:
                files = {"prompt_audio": (os.path.basename(prompt_audio_path), f, "audio/wav")}
                form_data = {
                    "text": text,
                    "seed": seed,
                    "max_tokens": max_tokens,
                }
                if request_id:
                    form_data["request_id"] = request_id
                
                response = self.session.post(
                    f"{self.server_url}/v1/tts/stream",
                    files=files,
                    data=form_data,
                    stream=True,
                    timeout=1800
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Chatterbox stream error: {response.status_code} - {response.text}")
                
                buffer = ""
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if stop_event and stop_event.is_set():
                        break
                    if not chunk:
                        continue
                    buffer += chunk
                    while "\n\n" in buffer:
                        event, buffer = buffer.split("\n\n", 1)
                        if not event.startswith("data: "):
                            continue
                        try:
                            event_data = json.loads(event[6:])
                        except json.JSONDecodeError:
                            continue
                        
                        event_type = event_data.get("type")
                        if event_type == "chunk":
                            audio_b64 = event_data.get("audio")
                            if not audio_b64:
                                continue
                            event_data["audio_bytes"] = base64.b64decode(audio_b64)
                        
                        if event_type == "error":
                            raise RuntimeError(f"Stream error: {event_data.get('message')}")
                        
                        yield event_data
                        
                        if event_type == "complete":
                            return
        finally:
            try:
                if response is not None:
                    response.close()
            except Exception:
                pass


# ============================================
# POCKET TTS CLIENT (Kyutai Moshi-based TTS)
# ============================================

class PocketTTSClient(BaseServiceClient):
    """Client for the Pocket TTS server (Kyutai Moshi-based).
    
    Pocket TTS is a lightweight TTS model that supports:
    - Zero-shot voice cloning via audio prompt
    - Built-in voice presets (alba, marius, javert, etc.)
    - OpenAI-compatible /v1/audio/speech API
    """
    
    # Built-in voice presets
    BUILTIN_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
    
    def __init__(self, server_url: str = None):
        super().__init__(server_url or POCKET_TTS_SERVER_URL)
        self._server_available = None
    
    def is_available(self) -> bool:
        """Check if Pocket TTS server is available (with caching)."""
        if self._server_available:
            return True
        
        result = super().is_available()
        if result:
            self._server_available = True
        return result
    
    def reset_availability_cache(self):
        """Reset the availability cache."""
        self._server_available = None
    
    def get_status(self) -> dict:
        """Get server status."""
        status = super().get_status()
        if "model_loaded" not in status:
            status["model_loaded"] = False
        return status
    
    def get_voices(self) -> list:
        """Get list of available built-in voices."""
        try:
            response = self.session.get(f"{self.server_url}/v1/voices", timeout=5)
            if response.status_code == 200:
                return response.json().get("voices", self.BUILTIN_VOICES)
            return self.BUILTIN_VOICES
        except Exception:
            return self.BUILTIN_VOICES
    
    def generate(
        self,
        text: str,
        voice: str = "alba",
        speed: float = 1.0,
        max_tokens: int = 50,
        request_id: str = None
    ) -> str:
        """
        Generate TTS audio using Pocket TTS.
        
        Args:
            text: Text to synthesize
            voice: Voice name OR path to audio prompt for cloning.
                   Built-in voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
                   For cloning: pass the path to a reference audio file
            speed: Playback speed (0.25-4.0)
            max_tokens: Max tokens per chunk (5-25=fast, 50=balanced, 75-100=quality)
            request_id: Optional request ID for logging
        
        Returns:
            Path to output WAV file
        
        Raises:
            RuntimeError: If generation fails
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            payload = {
                "model": "pocket-tts",
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "speed": speed,
                "max_tokens": max_tokens
            }
            
            logger.info(f"[PocketTTS] Sending request to {self.server_url}/v1/audio/speech")
            logger.info(f"[PocketTTS] Payload: voice={voice}, text_len={len(text)}")
            print(f"[PocketTTS] Sending request to {self.server_url}/v1/audio/speech")
            
            response = self.session.post(
                f"{self.server_url}/v1/audio/speech",
                json=payload,
                timeout=3600  # 60 minute timeout for long audio
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Pocket TTS server error: {response.status_code} - {response.text}")
            
            # Save response audio to temp file
            fd, output_path = tempfile.mkstemp(suffix="_pocket_tts.wav")
            os.close(fd)
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "Pocket TTS",
                "Please start the Pocket TTS server."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Pocket TTS generation failed: {e}")
    
    def generate_stream(
        self,
        text: str,
        voice: str = "alba",
        speed: float = 1.0,
        request_id: str = None,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> str:
        """
        Generate TTS audio using Pocket TTS with progress streaming.
        
        Args:
            text: Text to synthesize
            voice: Voice name OR path to audio prompt for cloning
            speed: Playback speed (0.25-4.0)
            request_id: Optional request ID for logging
            on_progress: Optional callback for progress events
        
        Returns:
            Path to output WAV file
        
        Raises:
            RuntimeError: If generation fails
        """
        import logging
        import base64
        logger = logging.getLogger(__name__)
        
        try:
            payload = {
                "model": "pocket-tts",
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "speed": speed
            }
            
            logger.info(f"[PocketTTS] Streaming request to {self.server_url}/v1/audio/speech/stream")
            print(f"[PocketTTS] Streaming request - voice={voice}, text_len={len(text)}")
            
            response = self.session.post(
                f"{self.server_url}/v1/audio/speech/stream",
                json=payload,
                stream=True,
                timeout=3600  # 60 minute timeout for long audio
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Pocket TTS server error: {response.status_code} - {response.text}")
            
            audio_data = None
            
            # Parse SSE stream
            for line in response.iter_lines():
                if not line:
                    continue
                
                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue
                
                try:
                    event = json.loads(line[6:])
                    event_type = event.get('type')
                    
                    if event_type == 'progress':
                        logger.info(f"[PocketTTS] Progress: {event.get('sentence')}/{event.get('total')} ({event.get('progress')}%)")
                        if on_progress:
                            on_progress(event)
                    
                    elif event_type == 'sentence_complete':
                        logger.info(f"[PocketTTS] Sentence {event.get('sentence')} done: {event.get('audio_duration')}s in {event.get('generation_time')}s ({event.get('rtf')}x RT)")
                        if on_progress:
                            on_progress(event)
                    
                    elif event_type == 'start':
                        logger.info(f"[PocketTTS] Starting: {event.get('total_sentences')} sentences, {event.get('total_chars')} chars")
                        if on_progress:
                            on_progress(event)
                    
                    elif event_type == 'audio':
                        logger.info(f"[PocketTTS] Received audio: {event.get('duration')}s, {event.get('size_bytes')} bytes")
                        audio_data = base64.b64decode(event.get('data', ''))
                        if on_progress:
                            on_progress({'type': 'audio_received', 'duration': event.get('duration'), 'size_bytes': event.get('size_bytes')})
                    
                    elif event_type == 'complete':
                        logger.info(f"[PocketTTS] Complete: {event.get('audio_duration')}s in {event.get('total_time')}s ({event.get('rtf')}x RT)")
                        if on_progress:
                            on_progress(event)
                    
                    elif event_type == 'error':
                        raise RuntimeError(f"Pocket TTS error: {event.get('message')}")
                    
                    elif event_type == 'warning':
                        logger.warning(f"[PocketTTS] Warning: {event.get('message')}")
                        if on_progress:
                            on_progress(event)
                
                except json.JSONDecodeError:
                    continue
            
            if not audio_data:
                raise RuntimeError("No audio data received from Pocket TTS")
            
            # Save audio to temp file
            fd, output_path = tempfile.mkstemp(suffix="_pocket_tts.wav")
            os.close(fd)
            with open(output_path, "wb") as f:
                f.write(audio_data)
            
            return output_path
        
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "Pocket TTS",
                "Please start the Pocket TTS server."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Pocket TTS streaming generation failed: {e}")


class KokoroTTSClient(BaseServiceClient):
    """Client for the Kokoro TTS server (ONNX-based TTS).
    
    Kokoro TTS is a lightweight, high-quality TTS model using ONNX runtime.
    - Multiple voice presets (af, am, bf, bm)
    - Fast inference with ONNX runtime
    - OpenAI-compatible /v1/audio/speech API
    """
    
    # Available voice presets
    AVAILABLE_VOICES = ["af", "am", "bf", "bm"]
    
    def __init__(self, server_url: str = None):
        super().__init__(server_url or KOKORO_TTS_SERVER_URL)
        self._server_available = None
    
    def is_available(self) -> bool:
        """Check if Kokoro TTS server is available (with caching)."""
        if self._server_available:
            return True
        
        result = super().is_available()
        if result:
            self._server_available = True
        return result
    
    def reset_availability_cache(self):
        """Reset the availability cache."""
        self._server_available = None
    
    def get_status(self) -> dict:
        """Get server status."""
        status = super().get_status()
        if "model_loaded" not in status:
            status["model_loaded"] = False
        return status
    
    def get_voices(self) -> list:
        """Get list of available voices."""
        try:
            response = self.session.get(f"{self.server_url}/v1/voices", timeout=5)
            if response.status_code == 200:
                voices = response.json().get("voices", [])
                return [v["id"] for v in voices] if voices else self.AVAILABLE_VOICES
            return self.AVAILABLE_VOICES
        except Exception:
            return self.AVAILABLE_VOICES
    
    def generate(
        self,
        text: str,
        voice: str = "af",
        speed: float = 1.0,
        max_tokens: int = 50,
        token_method: str = "tiktoken",
        request_id: str = None
    ) -> str:
        """
        Generate TTS audio using Kokoro.
        
        Args:
            text: Text to synthesize
            voice: Voice preset (af, am, bf, bm)
            speed: Playback speed (0.25-4.0)
            max_tokens: Max tokens per text chunk
            token_method: Token counting method (tiktoken or words)
            request_id: Optional request ID for logging
            
        Returns:
            Path to output WAV file
            
        Raises:
            RuntimeError: If generation fails
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            payload = {
                "model": "kokoro-tts",
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "speed": speed,
                "max_tokens": max_tokens,
                "token_method": token_method,
            }
            
            logger.info(f"[KokoroTTS] Sending request to {self.server_url}/v1/audio/speech")
            logger.info(f"[KokoroTTS] Payload: voice={voice}, text_len={len(text)}")
            print(f"[KokoroTTS] Sending request to {self.server_url}/v1/audio/speech")
            
            response = self.session.post(
                f"{self.server_url}/v1/audio/speech",
                json=payload,
                timeout=3600  # 60 minute timeout for long audio
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Kokoro TTS server error: {response.status_code} - {response.text}")
            
            # Save response audio to temp file
            fd, output_path = tempfile.mkstemp(suffix="_kokoro_tts.wav")
            os.close(fd)
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
            
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "Kokoro TTS",
                "Please start the Kokoro TTS server."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Kokoro TTS generation failed: {e}")
    
    def generate_stream(
        self,
        text: str,
        voice: str = "af",
        speed: float = 1.0,
        max_tokens: int = 50,
        token_method: str = "tiktoken",
        request_id: str = None,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> str:
        """
        Generate TTS audio using Kokoro with progress streaming.
        
        Args:
            text: Text to synthesize
            voice: Voice preset (af, am, bf, bm)
            speed: Playback speed (0.25-4.0)
            max_tokens: Max tokens per text chunk
            token_method: Token counting method (tiktoken or words)
            request_id: Optional request ID for logging
            on_progress: Optional callback for progress events
            
        Returns:
            Path to output WAV file
            
        Raises:
            RuntimeError: If generation fails
        """
        import logging
        import base64
        logger = logging.getLogger(__name__)
        
        try:
            payload = {
                "model": "kokoro-tts",
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "speed": speed,
                "max_tokens": max_tokens,
                "token_method": token_method,
            }
            
            logger.info(f"[KokoroTTS] Streaming request to {self.server_url}/v1/audio/speech/stream")
            print(f"[KokoroTTS] Streaming request to {self.server_url}/v1/audio/speech/stream")
            
            response = self.session.post(
                f"{self.server_url}/v1/audio/speech/stream",
                json=payload,
                stream=True,
                timeout=3600
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Kokoro TTS stream error: {response.status_code} - {response.text}")
            
            audio_data = b""
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                
                try:
                    event = json.loads(line[6:])
                    event_type = event.get("type")
                    
                    if event_type == "chunk":
                        chunk_b64 = event.get("audio", "") or event.get("audio_bytes_b64", "")
                        chunk_audio = base64.b64decode(chunk_b64) if chunk_b64 else b""
                        audio_data += chunk_audio
                        if on_progress:
                            on_progress(event)
                    elif event_type == "complete":
                        if on_progress:
                            on_progress(event)
                    elif event_type == "error":
                        raise RuntimeError(f"Kokoro TTS error: {event.get('message')}")
                        
                except json.JSONDecodeError:
                    continue
            
            if not audio_data:
                raise RuntimeError("No audio data received from Kokoro TTS")
            
            # Save audio to temp file
            fd, output_path = tempfile.mkstemp(suffix="_kokoro_tts.wav")
            os.close(fd)
            with open(output_path, "wb") as f:
                f.write(audio_data)
            
            return output_path
            
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "Kokoro TTS",
                "Please start the Kokoro TTS server."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Kokoro TTS streaming generation failed: {e}")


class OmniVoiceTTSClient(BaseServiceClient):
    """Client for the OmniVoice TTS server.

    OmniVoice supports:
    - Auto voice generation
    - Voice design via instruct text
    - Voice cloning via reference audio path
    - OpenAI-compatible /v1/audio/speech API
    """

    def __init__(self, server_url: str = None):
        super().__init__(server_url or OMNIVOICE_TTS_SERVER_URL)
        self._server_available = None

    def is_available(self) -> bool:
        """Check if OmniVoice server is available (with caching)."""
        if self._server_available:
            return True

        result = super().is_available()
        if result:
            self._server_available = True
        return result

    def reset_availability_cache(self):
        """Reset the availability cache."""
        self._server_available = None

    def get_status(self) -> dict:
        """Get server status."""
        status = super().get_status()
        if "model_loaded" not in status:
            status["model_loaded"] = False
        return status

    def generate(
        self,
        text: str,
        voice: str = "auto",
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        max_tokens: int = 50,
        token_method: str = "tiktoken",
        prechunked: bool = False,
        request_id: str = None,
    ) -> str:
        """Generate TTS audio using OmniVoice."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            payload = {
                "model": "omnivoice-tts",
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "speed": speed,
                "max_tokens": max_tokens,
                "token_method": token_method,
                "prechunked": bool(prechunked),
            }
            if ref_text and ref_text.strip():
                payload["ref_text"] = ref_text.strip()

            logger.info(f"[OmniVoice] Sending request to {self.server_url}/v1/audio/speech")
            logger.info(f"[OmniVoice] Payload: voice={voice}, text_len={len(text)}")
            print(f"[OmniVoice] Sending request to {self.server_url}/v1/audio/speech")

            response = self.session.post(
                f"{self.server_url}/v1/audio/speech",
                json=payload,
                timeout=3600,
            )

            if response.status_code != 200:
                raise RuntimeError(f"OmniVoice server error: {response.status_code} - {response.text}")

            fd, output_path = tempfile.mkstemp(suffix="_omnivoice_tts.wav")
            os.close(fd)
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path

        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "OmniVoice TTS",
                "Please start the OmniVoice server.",
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"OmniVoice TTS generation failed: {e}")

    def stream_events(
        self,
        text: str,
        voice: str = "auto",
        speed: float = 1.0,
        max_tokens: int = 50,
        token_method: str = "tiktoken",
        request_id: str = None,
        stop_event: Optional[threading.Event] = None,
    ):
        """
        Stream OmniVoice SSE events as they arrive.

        Yields:
            dict events with:
            - type: "start" | "chunk" | "complete"
            - audio_bytes (for chunk events)
        """
        import base64
        import json

        payload = {
            "model": "omnivoice-tts",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": speed,
            "max_tokens": max_tokens,
            "token_method": token_method,
        }
        if request_id:
            payload["request_id"] = request_id

        response = None
        try:
            response = self.session.post(
                f"{self.server_url}/v1/audio/speech/stream",
                json=payload,
                stream=True,
                timeout=3600,
            )

            if response.status_code != 200:
                raise RuntimeError(f"OmniVoice stream error: {response.status_code} - {response.text}")

            buffer = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if stop_event and stop_event.is_set():
                    break
                if not chunk:
                    continue

                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    if not event.startswith("data: "):
                        continue

                    try:
                        event_data = json.loads(event[6:])
                    except json.JSONDecodeError:
                        continue

                    event_type = event_data.get("type")
                    if event_type == "chunk":
                        audio_b64 = event_data.get("audio_bytes_b64", "") or event_data.get("audio", "")
                        if not audio_b64:
                            continue
                        event_data["audio_bytes"] = base64.b64decode(audio_b64)

                    if event_type == "error":
                        raise RuntimeError(f"Stream error: {event_data.get('message')}")

                    yield event_data

                    if event_type == "complete":
                        return
        except requests.exceptions.ConnectionError:
            raise self._handle_connection_error(
                "OmniVoice TTS",
                "Please start the OmniVoice server.",
            )
        finally:
            try:
                if response is not None:
                    response.close()
            except Exception:
                pass


# ============================================
# GLOBAL INSTANCES AND HELPER FUNCTIONS
# ============================================

# Global client instances (lazy initialized)
_asr_client = None
_glmasr_client = None
_rvc_client = None
_chatterbox_client = None
_pocket_tts_client = None
_kokoro_tts_client = None
_omnivoice_tts_client = None
_omnivoice_onnx_tts_client = None


# ASR helpers (Whisper - default)
def get_asr_client() -> WhisperASRClient:
    """Get or create the global ASR client (Whisper)."""
    global _asr_client
    if _asr_client is None:
        _asr_client = WhisperASRClient()
    return _asr_client


def is_asr_available() -> bool:
    """Check if ASR (Whisper) is available."""
    return get_asr_client().is_available()


# WhisperASR helpers (preferred naming)
def get_whisperasr_client() -> WhisperASRClient:
    return get_asr_client()


def is_whisperasr_available() -> bool:
    return is_asr_available()


# GLM-ASR helpers
def get_glmasr_client() -> GLMASRClient:
    """Get or create the global GLM-ASR client."""
    global _glmasr_client
    if _glmasr_client is None:
        _glmasr_client = GLMASRClient()
    return _glmasr_client


def is_glmasr_available() -> bool:
    """Check if GLM-ASR server is available."""
    return get_glmasr_client().is_available()


def transcribe_audio(
    audio_path: str,
    language: str = "en",
    response_format: Literal["json", "text", "verbose_json"] = "json",
    clean_vocals: bool = False,
    skip_existing_vocals: bool = True,
    postprocess_audio: bool = False,
    device: str = "gpu",
    model: str = None
) -> dict:
    """Transcribe audio using the appropriate ASR server.
    
    Automatically routes to GLM-ASR or Whisper based on model name:
    - Models starting with 'glm' use GLM-ASR server
    - All other models use Whisper server
    
    Args:
        audio_path: Path to audio file
        language: Language code (default: "en")
        response_format: Response format (json, text, verbose_json)
        clean_vocals: If True, use UVR5 to remove background before transcription
        skip_existing_vocals: If True, reuse cached vocals if available
        postprocess_audio: If True, apply audio enhancement before transcription
        device: "gpu" or "cpu"
        model: ASR model name. Use "glm-asr-nano" for GLM-ASR, 
               "whisper-large-v3-turbo" etc. for Whisper
    
    Returns:
        Transcription result dict with "text" key
    """
    # Route to appropriate backend based on model name
    if model and model.lower().startswith("glm"):
        return get_glmasr_client().transcribe(
            audio_path, language, response_format, 
            clean_vocals, skip_existing_vocals, postprocess_audio, 
            device, model
        )
    else:
        return get_asr_client().transcribe(
            audio_path, language, response_format, 
            clean_vocals, skip_existing_vocals, postprocess_audio, 
            device, model
        )


# RVC helpers
def get_rvc_client() -> RVCClient:
    """Get or create the global RVC client."""
    global _rvc_client
    if _rvc_client is None:
        _rvc_client = RVCClient()
    return _rvc_client


def is_rvc_server_available() -> bool:
    """Check if RVC server is available."""
    return get_rvc_client().is_available()


def run_rvc(
    audio_path: str,
    model_name: str,
    rvc_params: dict,
    status_update: Callable[[str], None] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    chunk_duration: int = 60,
    request_id: str = None
) -> str:
    """
    Run RVC voice conversion via the RVC server.
    
    Args:
        audio_path: Path to input audio file
        model_name: Name of the RVC model to use
        rvc_params: RVC parameters dict with keys:
            - pitch_algo: Pitch detection algorithm (rmvpe, pm, harvest)
            - pitch_lvl: Pitch shift in semitones
            - index_influence: Index file influence (0.0-1.0)
            - respiration_median_filtering: Median filter for breathing
            - envelope_ratio: Volume envelope ratio
            - consonant_breath_protection: Protect consonants/breaths
        status_update: Callback for status updates
        progress_callback: Callback for progress updates (0.0-1.0)
        chunk_duration: Duration of each chunk in seconds
        request_id: Optional request ID for unified logging across services
    
    Returns:
        Path to output WAV file
    
    Raises:
        RuntimeError: If RVC server is not available or conversion fails
    """
    if status_update is None:
        status_update = lambda s: None
    if progress_callback is None:
        progress_callback = lambda f: None
    
    client = get_rvc_client()
    
    if not client.is_available():
        raise RuntimeError(
            "RVC server is not available. "
            "Please start the RVC server using option [3] in the launcher."
        )
    
    progress_callback(0.0)
    status_update(f"Sending to RVC server: {model_name}...")
    
    progress_callback(0.1)
    status_update("Processing with RVC server...")
    
    # Pass params as-is - None values will use server config
    result = client.convert_chunked(
        audio_path=audio_path,
        model_name=model_name,
        pitch_algo=rvc_params.get('pitch_algo'),
        pitch_lvl=rvc_params.get('pitch_lvl'),
        index_influence=rvc_params.get('index_influence'),
        respiration_median_filtering=rvc_params.get('respiration_median_filtering'),
        envelope_ratio=rvc_params.get('envelope_ratio'),
        consonant_breath_protection=rvc_params.get('consonant_breath_protection'),
        chunk_duration=chunk_duration,
        request_id=request_id
    )
    
    progress_callback(1.0)
    status_update("RVC conversion complete")
    return result


def run_rvc_stream(
    audio_path: str,
    model_name: str,
    rvc_params: dict,
    status_update: Callable[[str], None] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    chunk_duration: int = 60,
    request_id: str = None
) -> str:
    """
    Run RVC voice conversion via the RVC server using STREAMING endpoint.
    
    Uses SSE streaming transport - server streams chunks as they're processed.
    
    Args:
        Same as run_rvc()
    
    Returns:
        Path to output WAV file
    
    Raises:
        RuntimeError: If RVC server is not available or conversion fails
    """
    if status_update is None:
        status_update = lambda s: None
    if progress_callback is None:
        progress_callback = lambda f: None
    
    client = get_rvc_client()
    
    if not client.is_available():
        raise RuntimeError(
            "RVC server is not available. "
            "Please start the RVC server using option [3] in the launcher."
        )
    
    progress_callback(0.0)
    status_update(f"Sending to RVC server (streaming): {model_name}...")
    
    progress_callback(0.1)
    status_update("Processing with RVC server (streaming)...")
    
    # Use STREAMING endpoint
    result = client.convert_stream(
        audio_path=audio_path,
        model_name=model_name,
        pitch_algo=rvc_params.get('pitch_algo'),
        pitch_lvl=rvc_params.get('pitch_lvl'),
        index_influence=rvc_params.get('index_influence'),
        respiration_median_filtering=rvc_params.get('respiration_median_filtering'),
        envelope_ratio=rvc_params.get('envelope_ratio'),
        consonant_breath_protection=rvc_params.get('consonant_breath_protection'),
        chunk_duration=chunk_duration,
        on_progress=progress_callback,
        request_id=request_id
    )
    
    progress_callback(1.0)
    status_update("RVC conversion complete (streamed)")
    return result


# PostProcess helpers
_postprocess_client: Optional[PostProcessClient] = None

# Preprocess helpers
_preprocess_client: Optional[PreprocessClient] = None


def get_postprocess_client() -> PostProcessClient:
    """Get or create the global PostProcess client."""
    global _postprocess_client
    if _postprocess_client is None:
        _postprocess_client = PostProcessClient()
    return _postprocess_client


def get_preprocess_client() -> PreprocessClient:
    """Get or create the global Preprocess client."""
    global _preprocess_client
    if _preprocess_client is None:
        _preprocess_client = PreprocessClient()
    return _preprocess_client


def is_preprocess_server_available() -> bool:
    """Check if preprocess server is available."""
    return get_preprocess_client().is_available()


def clean_vocals_uvr5(
    audio_path: str,
    aggression: int = 10,
    device: Optional[str] = None,
    skip_if_cached: bool = True,
    original_filename: str = None,
) -> str:
    """Clean vocals using UVR5 via preprocess server. Returns vocals-only wav path."""
    return get_preprocess_client().clean_vocals(
        audio_path, 
        aggression=aggression, 
        device=device,
        skip_if_cached=skip_if_cached,
        original_filename=original_filename,
        model_key="hp5_vocals",
    )


def process_uvr5(
    audio_path: str,
    model_key: str = "hp5_vocals",
    aggression: int = 10,
    device: Optional[str] = None,
    skip_if_cached: bool = True,
    original_filename: str = None,
) -> str:
    """
    Process audio using UVR5 via preprocess server.
    
    Args:
        audio_path: Path to input audio
        model_key: Model to use (hp5_vocals, deecho_normal, deecho_aggressive, deecho_dereverb)
        aggression: Processing aggressiveness (0-20)
        device: "cuda" or "cpu"
        skip_if_cached: Reuse cached result if available
        original_filename: Original filename for cache key
    
    Returns:
        Path to processed wav file
    """
    return get_preprocess_client().clean_vocals(
        audio_path, 
        aggression=aggression, 
        device=device,
        skip_if_cached=skip_if_cached,
        original_filename=original_filename,
        model_key=model_key,
    )


def is_postprocess_server_available() -> bool:
    """Check if post-processing server is available."""
    return get_postprocess_client().is_available()


def run_postprocess(
    audio_path: str,
    params: Dict[str, Any],
    status_update: Callable[[str], None] = None,
    request_id: str = None
) -> str:
    """
    Run post-processing via the server.
    
    Args:
        audio_path: Path to input audio file
        params: Post-processing parameters
        status_update: Callback for status updates
        request_id: Optional request ID for unified logging
    
    Returns:
        Path to processed audio file
    """
    import time
    t_start = time.perf_counter()
    
    if status_update is None:
        status_update = lambda s: None
    
    req_tag = f"[{request_id}] " if request_id else ""
    
    client = get_postprocess_client()
    
    if not client.is_available():
        raise RuntimeError(
            "Post-processing server is not available. "
            "Please start the post-processing server."
        )
    
    status_update("Applying post-processing effects...")
    return client.postprocess(audio_path, params)


def run_blend(
    audio_path: str,
    bg_files: list,
    bg_volumes: list,
    main_volume: float = 1.0,
    status_update: Callable[[str], None] = None,
    bg_delays: list = None,
    bg_fade_ins: list = None,
    bg_fade_outs: list = None,
    request_id: str = None
) -> str:
    """
    Blend audio with background tracks via the server.
    
    Args:
        audio_path: Path to main audio file
        bg_files: List of background file paths
        bg_volumes: List of background volumes
        main_volume: Main audio volume
        status_update: Callback for status updates
        bg_delays: List of delays in seconds for each background track
        bg_fade_ins: List of fade in durations in seconds for each background track
        bg_fade_outs: List of fade out durations in seconds for each background track
        request_id: Optional request ID for unified logging
    
    Returns:
        Path to blended audio file
    """
    if status_update is None:
        status_update = lambda s: None
    
    if request_id:
        print(f"[{request_id}] Blending with {len(bg_files)} background tracks...")
    
    if bg_delays is None:
        bg_delays = [0.0] * len(bg_files)
    if bg_fade_ins is None:
        bg_fade_ins = [0.0] * len(bg_files)
    if bg_fade_outs is None:
        bg_fade_outs = [0.0] * len(bg_files)
    
    client = get_postprocess_client()
    
    if not client.is_available():
        raise RuntimeError("Post-processing server is not available.")
    
    status_update("Blending with background audio...")
    return client.blend(audio_path, bg_files, bg_volumes, main_volume, bg_delays, bg_fade_ins, bg_fade_outs)


# =============================================================================
# Streaming Background Mixer Client Functions
# =============================================================================

def preload_background_audio(files: list, sample_rate: int = 24000, request_id: str = None) -> dict:
    """
    Pre-load background audio files into cache on audio_services server.
    
    Args:
        files: List of file paths to preload
        sample_rate: Target sample rate
        request_id: Optional request ID for logging
    
    Returns:
        Cache info dict
    """
    req_tag = f"[{request_id}] " if request_id else ""
    
    try:
        session = get_shared_session()
        resp = session.post(
            f"{AUDIO_SERVICES_SERVER_URL}/v1/background/preload",
            json={"files": files, "sample_rate": sample_rate},
            timeout=300
        )
        resp.raise_for_status()
        info = resp.json()
        print(f"{req_tag}[BG] Preloaded {len(files)} files, cache: {info.get('memory_mb', 0):.1f}MB")
        return info
    except Exception as e:
        print(f"{req_tag}[BG] Preload failed: {e}")
        return {}


def mix_chunk_with_background(
    chunk_path: str,
    session_id: str,
    tracks: list,
    sample_rate: int = 24000,
    request_id: str = None
) -> str:
    """
    Mix background audio into a streaming chunk via audio_services server.
    
    Args:
        chunk_path: Path to the chunk WAV file
        session_id: Session ID to maintain state across chunks
        tracks: List of {"file": path, "volume": float, "delay": float}
        sample_rate: Sample rate
        request_id: Optional request ID for logging
    
    Returns:
        Path to mixed chunk file
    """
    import json
    import tempfile
    
    req_tag = f"[{request_id}] " if request_id else ""
    
    try:
        session = get_shared_session()
        
        with open(chunk_path, 'rb') as f:
            files = {'audio': (os.path.basename(chunk_path), f, 'audio/wav')}
            data = {
                'session_id': session_id,
                'tracks_json': json.dumps(tracks),
                'sample_rate': str(sample_rate),
            }
            
            resp = session.post(
                f"{AUDIO_SERVICES_SERVER_URL}/v1/background/mix-chunk",
                files=files,
                data=data,
                timeout=60
            )
            resp.raise_for_status()
        
        # Save the mixed audio to a temp file
        fd, output_path = tempfile.mkstemp(suffix="_bgmix.wav")
        os.close(fd)
        with open(output_path, 'wb') as f:
            f.write(resp.content)
        
        return output_path
        
    except Exception as e:
        print(f"{req_tag}[BG] Mix chunk failed: {e}")
        return chunk_path  # Return original on error


def end_background_session(session_id: str, request_id: str = None):
    """End a streaming background mix session."""
    req_tag = f"[{request_id}] " if request_id else ""
    
    try:
        session = get_shared_session()
        resp = session.post(
            f"{AUDIO_SERVICES_SERVER_URL}/v1/background/end-session",
            data={'session_id': session_id},
            timeout=10
        )
        resp.raise_for_status()
        print(f"{req_tag}[BG] Session {session_id} ended")
    except Exception as e:
        print(f"{req_tag}[BG] End session failed: {e}")


def run_master(
    audio_path: str,
    status_update: Callable[[str], None] = None,
    request_id: str = None
) -> str:
    """
    Apply final mastering via the server.
    
    Args:
        audio_path: Path to input audio file
        status_update: Callback for status updates
        request_id: Optional request ID for unified logging
    
    Returns:
        Path to mastered audio file
    """
    if status_update is None:
        status_update = lambda s: None
    
    if request_id:
        print(f"[{request_id}] Mastering audio...")
    
    client = get_postprocess_client()
    
    if not client.is_available():
        raise RuntimeError("Post-processing server is not available.")
    
    status_update("Finalizing and mastering audio...")
    return client.master(audio_path)


def run_save(
    audio_path: str,
    text: str = "output",
    status_update: Callable[[str], None] = None,
    request_id: str = None
) -> str:
    """
    Save audio to output directory via the server.
    
    Args:
        audio_path: Path to input audio file
        text: Text to use for filename generation
        status_update: Callback for status updates
        request_id: Optional request ID for unified logging
    
    Returns:
        Path where file was saved
    """
    if status_update is None:
        status_update = lambda s: None
    
    if request_id:
        print(f"[{request_id}] Saving output...")
    
    client = get_postprocess_client()
    
    if not client.is_available():
        raise RuntimeError("Post-processing server is not available.")
    
    status_update("Saving to output folder...")
    return client.save(audio_path, text)


def run_resample(
    audio_path: str,
    sample_rate: int = 44100,
    volume: float = 1.0,
    request_id: str = None
) -> str:
    """
    Resample audio and optionally adjust volume via server.
    
    Args:
        audio_path: Path to input audio file
        sample_rate: Target sample rate (default 44100)
        volume: Volume multiplier (1.0 = no change)
        request_id: Optional request ID for logging
    
    Returns:
        Path to processed audio file
    """
    if request_id:
        _log_verbose(f"[{request_id}] Resampling to {sample_rate}Hz, volume={volume}")
    
    client = get_postprocess_client()
    
    if not client.is_available():
        raise RuntimeError("Post-processing server is not available.")
    
    return client.resample(audio_path, sample_rate, volume)


def run_process_chunk(
    audio_path: str,
    post_params: Optional[Dict[str, Any]] = None,
    target_sample_rate: int = 44100,
    output_volume: float = 1.0,
    request_id: str = None
) -> str:
    """
    Combined PostProcess + Resample in one HTTP call (optimized for streaming).
    
    Args:
        audio_path: Path to input audio file
        post_params: Post-processing parameters dict (None = skip post-process)
        target_sample_rate: Target sample rate for output
        output_volume: Volume multiplier for output
        request_id: Optional request ID for logging
    
    Returns:
        Path to processed audio file
    """
    if request_id:
        _log_verbose(f"[{request_id}] Processing chunk: post={post_params is not None}, sr={target_sample_rate}, vol={output_volume}")
    
    client = get_postprocess_client()
    
    if not client.is_available():
        raise RuntimeError("Post-processing server is not available.")
    
    return client.process_chunk(audio_path, post_params, target_sample_rate, output_volume)


# Chatterbox helpers
def get_chatterbox_client() -> ChatterboxClient:
    """Get or create the global Chatterbox client."""
    global _chatterbox_client
    if _chatterbox_client is None:
        _chatterbox_client = ChatterboxClient()
    return _chatterbox_client


def is_chatterbox_server_available() -> bool:
    """Check if Chatterbox server is available."""
    return get_chatterbox_client().is_available()


# Pocket TTS helpers
def get_pocket_tts_client() -> PocketTTSClient:
    """Get or create the global Pocket TTS client."""
    global _pocket_tts_client
    if _pocket_tts_client is None:
        _pocket_tts_client = PocketTTSClient()
    return _pocket_tts_client


def is_pocket_tts_server_available() -> bool:
    """Check if Pocket TTS server is available."""
    return get_pocket_tts_client().is_available()


# Kokoro TTS helpers
def get_kokoro_tts_client() -> KokoroTTSClient:
    """Get or create the global Kokoro TTS client."""
    global _kokoro_tts_client
    if _kokoro_tts_client is None:
        _kokoro_tts_client = KokoroTTSClient()
    return _kokoro_tts_client


def is_kokoro_tts_server_available() -> bool:
    """Check if Kokoro TTS server is available."""
    return get_kokoro_tts_client().is_available()


# OmniVoice TTS helpers
def get_omnivoice_tts_client() -> OmniVoiceTTSClient:
    """Get or create the global OmniVoice TTS client."""
    global _omnivoice_tts_client
    if _omnivoice_tts_client is None:
        _omnivoice_tts_client = OmniVoiceTTSClient()
    return _omnivoice_tts_client


def is_omnivoice_tts_server_available() -> bool:
    """Check if OmniVoice TTS server is available."""
    return get_omnivoice_tts_client().is_available()


def get_omnivoice_onnx_tts_client() -> OmniVoiceTTSClient:
    """Get or create the global OmniVoice ONNX TTS client."""
    global _omnivoice_onnx_tts_client
    if _omnivoice_onnx_tts_client is None:
        _omnivoice_onnx_tts_client = OmniVoiceTTSClient(server_url=OMNIVOICE_ONNX_TTS_SERVER_URL)
    return _omnivoice_onnx_tts_client


def is_omnivoice_onnx_tts_server_available() -> bool:
    """Check if OmniVoice ONNX TTS server is available."""
    return get_omnivoice_onnx_tts_client().is_available()


# Export all public symbols
__all__ = [
    # Base
    'BaseServiceClient',
    # Clients
    'WhisperASRClient',
    'GLMASRClient',
    'RVCClient',
    'ChatterboxClient',
    'PocketTTSClient',
    'KokoroTTSClient',
    'OmniVoiceTTSClient',
    'PostProcessClient',
    'PreprocessClient',
    # ASR helpers (Whisper)
    'get_whisperasr_client',
    'is_whisperasr_available',
    'transcribe_audio',
    # GLM-ASR helpers
    'get_glmasr_client',
    'is_glmasr_available',
    # RVC helpers
    'get_rvc_client',
    'is_rvc_server_available',
    'run_rvc',
    # PostProcess helpers
    'get_postprocess_client',
    'is_postprocess_server_available',
    'run_postprocess',
    'run_blend',
    'run_master',
    'run_resample',
    'run_save',
    'preload_background_audio',
    'mix_chunk_with_background',
    'end_background_session',
    # Preprocess helpers
    'get_preprocess_client',
    'is_preprocess_server_available',
    'clean_vocals_uvr5',
    'process_uvr5',
    # Chatterbox helpers
    'get_chatterbox_client',
    'is_chatterbox_server_available',
    # Pocket TTS helpers
    'get_pocket_tts_client',
    'is_pocket_tts_server_available',
    # Kokoro TTS helpers
    'get_kokoro_tts_client',
    'is_kokoro_tts_server_available',
    # OmniVoice TTS helpers
    'get_omnivoice_tts_client',
    'is_omnivoice_tts_server_available',
    'get_omnivoice_onnx_tts_client',
    'is_omnivoice_onnx_tts_server_available',
    # Server URLs
    'get_shared_session',
    'WHISPERASR_SERVER_URL',
    'GLMASR_SERVER_URL',
    'RVC_SERVER_URL',
    'CHATTERBOX_SERVER_URL',
    'POCKET_TTS_SERVER_URL',
    'KOKORO_TTS_SERVER_URL',
    'OMNIVOICE_TTS_SERVER_URL',
    'OMNIVOICE_ONNX_TTS_SERVER_URL',
    'AUDIO_SERVICES_SERVER_URL',
]
