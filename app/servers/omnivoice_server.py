from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import io
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

omnivoice_models: Dict[str, Any] = {}
_model_details: Dict[str, str] = {}
_model_load_lock = threading.Lock()
_model_load_failures: Dict[str, Dict[str, Any]] = {}
DEFAULT_OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_OMNIVOICE_ONNX_MODEL_ID = "gluschenko/omnivoice-onnx"
OMNIVOICE_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 48000
ASR_SERVER_URL = os.getenv("ASR_SERVER_URL", "http://127.0.0.1:8889").rstrip("/")
OMNIVOICE_USE_EXTERNAL_ASR = os.getenv("OMNIVOICE_USE_EXTERNAL_ASR", "1").strip().lower() in {"1", "true", "yes", "on"}
OMNIVOICE_EXTERNAL_ASR_MODEL = os.getenv("OMNIVOICE_EXTERNAL_ASR_MODEL", "glm-asr-nano")


def _normalize_runtime(value: str) -> str:
    runtime = (value or "torch").strip().lower()
    if runtime in {"onnx-gpu", "onnx_gpu", "onnxcuda", "onnx-cuda", "cuda"}:
        return "onnx-gpu"
    if runtime in {"onnx", "onnx-cpu", "onnx_cpu", "onnxcpu"}:
        return "onnx-cpu"
    return "torch"


OMNIVOICE_DEFAULT_RUNTIME = _normalize_runtime(os.getenv("OMNIVOICE_RUNTIME", "torch"))
_model_from_env = os.getenv("OMNIVOICE_MODEL_ID", "").strip()
OMNIVOICE_TORCH_MODEL_ID = os.getenv("OMNIVOICE_TORCH_MODEL_ID", "").strip()
if not OMNIVOICE_TORCH_MODEL_ID:
    if _model_from_env and OMNIVOICE_DEFAULT_RUNTIME == "torch":
        OMNIVOICE_TORCH_MODEL_ID = _model_from_env
    else:
        OMNIVOICE_TORCH_MODEL_ID = DEFAULT_OMNIVOICE_MODEL_ID

OMNIVOICE_ONNX_MODEL_ID = os.getenv("OMNIVOICE_ONNX_MODEL_ID", "").strip()
if not OMNIVOICE_ONNX_MODEL_ID:
    if _model_from_env and OMNIVOICE_DEFAULT_RUNTIME == "onnx-cpu":
        OMNIVOICE_ONNX_MODEL_ID = _model_from_env
    else:
        OMNIVOICE_ONNX_MODEL_ID = DEFAULT_OMNIVOICE_ONNX_MODEL_ID

OMNIVOICE_ONNX_MODEL_FILE = os.getenv("OMNIVOICE_ONNX_MODEL_FILE", "onnx/omnivoice.qint8.onnx").strip()
OMNIVOICE_ONNX_PROVIDERS = [
    provider.strip()
    for provider in os.getenv("OMNIVOICE_ONNX_PROVIDERS", "CPUExecutionProvider").split(",")
    if provider.strip()
]
OMNIVOICE_ONNX_GPU_PROVIDERS = [
    provider.strip()
    for provider in os.getenv(
        "OMNIVOICE_ONNX_GPU_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider",
    ).split(",")
    if provider.strip()
]
OMNIVOICE_ONNX_GPU_REQUIRE_CUDA = os.getenv("OMNIVOICE_ONNX_GPU_REQUIRE_CUDA", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OMNIVOICE_ONNX_NUM_STEP = int(os.getenv("OMNIVOICE_ONNX_NUM_STEP", "24"))
OMNIVOICE_ONNX_INTRA_OP_THREADS = int(os.getenv("OMNIVOICE_ONNX_INTRA_OP_THREADS", "0"))
OMNIVOICE_ONNX_INTER_OP_THREADS = int(os.getenv("OMNIVOICE_ONNX_INTER_OP_THREADS", "0"))
OMNIVOICE_ONNX_GRAPH_OPT = os.getenv("OMNIVOICE_ONNX_GRAPH_OPT", "all").strip().lower()
OMNIVOICE_ONNX_REPO_ID = os.getenv("OMNIVOICE_ONNX_REPO_ID", DEFAULT_OMNIVOICE_ONNX_MODEL_ID).strip() or DEFAULT_OMNIVOICE_ONNX_MODEL_ID
OMNIVOICE_ONNX_REMOTE_PATH = os.getenv("OMNIVOICE_ONNX_REMOTE_PATH", "onnx/omnivoice.qint8.onnx").strip() or "onnx/omnivoice.qint8.onnx"
APP_DIR = os.path.dirname(os.path.dirname(__file__))
OMNIVOICE_MODEL_CACHE_DIR = os.path.join(APP_DIR, "models", "omnivoice")
OMNIVOICE_PRELOAD_DEFAULT_RUNTIME = os.getenv("OMNIVOICE_PRELOAD_DEFAULT_RUNTIME", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OMNIVOICE_MODEL_LOAD_RETRY_SECONDS = int(os.getenv("OMNIVOICE_MODEL_LOAD_RETRY_SECONDS", "30"))

_resolved_onnx_model_file: Optional[str] = None


def _ensure_onnx_sidecar_if_needed(local_onnx_path: str) -> None:
    if not local_onnx_path.lower().endswith(".onnx"):
        return

    remote_sidecar = OMNIVOICE_ONNX_REMOTE_PATH + "_data"
    local_sidecar = local_onnx_path + "_data"
    _download_hf_file(OMNIVOICE_ONNX_REPO_ID, remote_sidecar, local_sidecar, required=False)


def _download_hf_file(repo_id: str, remote_path: str, local_target: str, required: bool = True) -> bool:
    os.makedirs(os.path.dirname(local_target), exist_ok=True)
    if os.path.isfile(local_target):
        return True

    url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_path}"
    tmp_target = local_target + ".part"
    logger.info(f"Downloading {remote_path} from {repo_id}")
    try:
        with httpx.stream("GET", url, timeout=httpx.Timeout(1800.0, connect=10.0), follow_redirects=True) as resp:
            if resp.status_code == 404:
                if required:
                    raise RuntimeError(f"Required model file not found on Hugging Face: {remote_path}")
                logger.warning(f"Optional model sidecar not found on Hugging Face: {remote_path}")
                return False
            resp.raise_for_status()
            with open(tmp_target, "wb") as out:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        os.replace(tmp_target, local_target)
        logger.info(f"Saved {remote_path} to {local_target}")
        return True
    except Exception:
        try:
            if os.path.exists(tmp_target):
                os.remove(tmp_target)
        except Exception:
            pass
        if required:
            raise
        return False


def _ensure_onnx_model_file() -> str:
    global _resolved_onnx_model_file

    if _resolved_onnx_model_file and os.path.isfile(_resolved_onnx_model_file):
        return _resolved_onnx_model_file

    configured = (OMNIVOICE_ONNX_MODEL_FILE or "").strip().strip('"')
    if configured and os.path.isfile(configured):
        _ensure_onnx_sidecar_if_needed(configured)
        _resolved_onnx_model_file = configured
        return _resolved_onnx_model_file

    if configured and not os.path.isabs(configured):
        repo_relative_local = os.path.join(OMNIVOICE_MODEL_CACHE_DIR, configured.replace("/", os.sep).replace("\\", os.sep))
        if os.path.isfile(repo_relative_local):
            _ensure_onnx_sidecar_if_needed(repo_relative_local)
            _resolved_onnx_model_file = repo_relative_local
            return _resolved_onnx_model_file

    local_target = os.path.join(
        OMNIVOICE_MODEL_CACHE_DIR,
        OMNIVOICE_ONNX_REMOTE_PATH.replace("/", os.sep).replace("\\", os.sep),
    )
    os.makedirs(os.path.dirname(local_target), exist_ok=True)

    if os.path.isfile(local_target):
        _ensure_onnx_sidecar_if_needed(local_target)
        _resolved_onnx_model_file = local_target
        return _resolved_onnx_model_file

    _download_hf_file(OMNIVOICE_ONNX_REPO_ID, OMNIVOICE_ONNX_REMOTE_PATH, local_target, required=True)
    _ensure_onnx_sidecar_if_needed(local_target)

    _resolved_onnx_model_file = local_target
    return _resolved_onnx_model_file

VOICE_GUIDE = [
    "auto",
    "instruct:female, low pitch, british accent",
    r"C:\path\to\reference.wav",
]


class AudioSpeechRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "model": "omnivoice-tts",
                "input": "Hello, world!",
                "voice": "auto",
                "response_format": "wav",
                "speed": 1.0,
                "max_tokens": 50,
            }
        },
    )

    model: str = Field(default="omnivoice-tts", description="Model to use")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(
        default="auto",
        description="auto, instruct:<attributes>, or path to reference audio for cloning",
    )
    ref_text: Optional[str] = Field(default=None, description="Optional transcript for reference audio")
    response_format: str = Field(default="wav", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    num_step: Optional[int] = Field(default=None, ge=4, le=64, description="Decoding steps (lower = faster, lower quality)")
    max_tokens: int = Field(default=50, ge=5, le=200, description="Max tokens per chunk")
    token_method: str = Field(default="tiktoken", description="Token counting method: tiktoken or words")
    prechunked: bool = Field(default=False, description="If true, skip server-side text splitting")
    runtime: Optional[str] = Field(
        default=None,
        description="Runtime override: torch, onnx-cpu, onnx-gpu, or auto (default server runtime)",
    )


def _resolve_runtime_override(runtime: Optional[str]) -> str:
    requested = (runtime or "").strip().lower()
    if requested in {"", "auto", "default"}:
        return OMNIVOICE_DEFAULT_RUNTIME
    return _normalize_runtime(requested)


def _model_id_for_runtime(runtime: str) -> str:
    return OMNIVOICE_ONNX_MODEL_ID if runtime in {"onnx-cpu", "onnx-gpu"} else OMNIVOICE_TORCH_MODEL_ID


def _providers_for_runtime(runtime: str) -> List[str]:
    if runtime == "onnx-gpu":
        return OMNIVOICE_ONNX_GPU_PROVIDERS
    return OMNIVOICE_ONNX_PROVIDERS


def _resolve_audio_path(path_or_empty: str) -> Optional[str]:
    candidate = (path_or_empty or "").strip().strip('"')
    if not candidate:
        return None
    if candidate.startswith("file://"):
        candidate = candidate[len("file://") :]
    if os.path.isfile(candidate):
        return candidate
    return None


def _split_text_into_chunks(text: str, max_tokens: int = 50, token_method: str = "tiktoken") -> List[str]:
    import sys

    util_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "util")
    if util_path not in sys.path:
        sys.path.insert(0, util_path)

    from text_utils import split_text

    method = (token_method or "tiktoken").strip().lower()
    if method not in {"tiktoken", "words"}:
        method = "tiktoken"

    return [chunk for chunk in split_text(text, max_tokens=max_tokens, token_method=method) if chunk.strip()]


def _extract_waveform(result: Any) -> np.ndarray:
    """Normalize OmniVoice output into a mono float32 numpy array."""
    tensor = None

    if isinstance(result, list) and result:
        tensor = result[0]
    elif isinstance(result, torch.Tensor):
        tensor = result

    if tensor is None:
        raise RuntimeError("OmniVoice returned empty audio")

    if isinstance(tensor, torch.Tensor):
        wav = tensor.detach().cpu().float().numpy()
    else:
        wav = np.asarray(tensor, dtype=np.float32)

    if wav.ndim == 2:
        wav = wav[0]
    if wav.ndim != 1:
        raise RuntimeError(f"Unexpected OmniVoice output shape: {wav.shape}")

    return wav.astype(np.float32)


def _voice_kwargs(voice: str, ref_text: Optional[str]) -> Dict[str, Any]:
    voice = (voice or "auto").strip()
    lower = voice.lower()
    kwargs: Dict[str, Any] = {}

    looks_like_path = ("/" in voice) or ("\\" in voice) or lower.endswith(".wav")

    ref_audio = _resolve_audio_path(voice)
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
        if ref_text and ref_text.strip():
            kwargs["ref_text"] = ref_text.strip()
        return kwargs
    if looks_like_path:
        raise HTTPException(status_code=400, detail=f"Reference audio file not found: {voice}")

    if lower.startswith("instruct:"):
        instruct = voice.split(":", 1)[1].strip()
        if instruct:
            kwargs["instruct"] = instruct
        return kwargs

    if lower not in {"", "auto", "default", "random"}:
        kwargs["instruct"] = voice

    return kwargs


def _transcribe_reference_with_asr_server(ref_audio_path: str) -> Optional[str]:
    """Use VoiceForge ASR server once per request to avoid per-chunk re-transcription."""
    if not OMNIVOICE_USE_EXTERNAL_ASR:
        return None

    try:
        t0 = time.perf_counter()
        with open(ref_audio_path, "rb") as f:
            files = {"file": (os.path.basename(ref_audio_path), f, "audio/wav")}
            data = {
                "language": "auto",
                "response_format": "json",
                "model": OMNIVOICE_EXTERNAL_ASR_MODEL,
                "clean_vocals": "false",
                "skip_existing_vocals": "true",
                "postprocess_audio": "false",
                "device": "gpu",
            }
            with httpx.Client(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
                response = client.post(f"{ASR_SERVER_URL}/v1/audio/transcriptions", files=files, data=data)

        if response.status_code != 200:
            logger.warning(
                f"External ASR transcription failed ({response.status_code}). "
                "Falling back to OmniVoice internal transcription."
            )
            return None

        payload = response.json()
        text = (payload.get("text") or "").strip()
        if not text:
            logger.warning("External ASR returned empty text; falling back to OmniVoice internal transcription.")
            return None

        asr_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Reference transcript obtained from ASR server ({len(text)} chars, {asr_ms:.0f} ms)")
        return text
    except Exception as e:
        logger.warning(f"External ASR unavailable; falling back to OmniVoice internal transcription: {e}")
        return None


def _prepare_reference_text_once(voice: str, ref_text: Optional[str]) -> Optional[str]:
    """Prepare ref_text once per request so chunked generation does not retranscribe each chunk."""
    if ref_text and ref_text.strip():
        logger.info(f"Using caller-provided ref_text ({len(ref_text.strip())} chars)")
        return ref_text.strip()

    ref_audio = _resolve_audio_path(voice)
    if not ref_audio:
        return None

    logger.info(f"Preparing reference transcript once for: {os.path.basename(ref_audio)}")
    return _transcribe_reference_with_asr_server(ref_audio)


def _generate_chunk_audio(
    model: Any,
    text: str,
    voice: str,
    ref_text: Optional[str],
    speed: float,
    num_step: Optional[int] = None,
) -> np.ndarray:
    kwargs = _voice_kwargs(voice, ref_text)
    if num_step is not None:
        kwargs["num_step"] = int(num_step)
    result = model.generate(text=text, speed=speed, **kwargs)
    audio = _extract_waveform(result)

    if OMNIVOICE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
        audio = scipy.signal.resample_poly(audio, TARGET_SAMPLE_RATE, OMNIVOICE_SAMPLE_RATE)

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _load_torch_runtime_model(model_id: str):
    from omnivoice import OmniVoice

    if torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device_map = "mps"
        dtype = torch.float32
    else:
        device_map = "cpu"
        dtype = torch.float32

    model = OmniVoice.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype,
    )
    return model, f"torch runtime on {device_map} (dtype={dtype}, model={model_id})"


class OmniVoiceONNXRuntimeModel:
    def __init__(self, model, ort_session):
        self._model = model
        self._ort_session = ort_session
        self.is_onnx_runtime = True
        self.providers = list(ort_session.get_providers())

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)


def _build_onnx_runtime_model(model_id: str, onnx_model_file: str, providers: List[str]):
    import onnxruntime as ort
    import torch.nn as nn
    from huggingface_hub import snapshot_download
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig, OmniVoiceModelOutput, _get_time_steps, _gumbel_sample
    from omnivoice.models.omnivoice import RuleDurationEstimator
    from transformers import AutoFeatureExtractor, AutoTokenizer, HiggsAudioV2TokenizerModel

    class OmniVoiceONNX(OmniVoice):
        def __init__(self, config, ort_session):
            super().__init__(config, llm=nn.Identity())
            self._ort_session = ort_session

        def forward(
            self,
            input_ids,
            audio_mask,
            attention_mask=None,
            position_ids=None,
            document_ids=None,
            labels=None,
        ):
            del document_ids, labels

            seq_len = input_ids.size(-1)
            batch = input_ids.size(0)

            if attention_mask is None:
                attn_2d = torch.ones((batch, seq_len), dtype=torch.int64, device=input_ids.device)
            else:
                attn = attention_mask
                if attn.dim() == 4:
                    attn = attn[:, 0, :, :]
                if attn.dim() == 3:
                    attn = attn.any(dim=-1)
                attn_2d = attn.to(dtype=torch.int64)

            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.int64, device=input_ids.device).unsqueeze(0).expand(batch, -1)
            else:
                position_ids = position_ids.to(dtype=torch.int64)

            feeds = {
                "input_ids": input_ids.detach().to(torch.int64).cpu().numpy(),
                "audio_mask": audio_mask.detach().to(torch.bool).cpu().numpy(),
                "attention_mask": attn_2d.detach().cpu().numpy(),
                "position_ids": position_ids.detach().cpu().numpy(),
            }
            logits = self._ort_session.run(None, feeds)[0]
            logits_t = torch.from_numpy(logits).to(input_ids.device)
            return OmniVoiceModelOutput(logits=logits_t)

        def _generate_iterative(self, task, gen_config):
            B = task.batch_size

            inputs_list = [
                self._prepare_inference_inputs(
                    task.texts[i],
                    task.target_lens[i],
                    task.ref_texts[i],
                    task.ref_audio_tokens[i],
                    task.langs[i],
                    task.instructs[i],
                    gen_config.denoise,
                )
                for i in range(B)
            ]

            c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
            max_c_len = max(c_lens)
            pad_id = self.config.audio_mask_id

            batch_input_ids = torch.full(
                (2 * B, self.config.num_audio_codebook, max_c_len),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            batch_audio_mask = torch.zeros((2 * B, max_c_len), dtype=torch.bool, device=self.device)
            batch_attention_mask = torch.zeros((2 * B, max_c_len), dtype=torch.int64, device=self.device)

            for i, inp in enumerate(inputs_list):
                c_len, u_len = c_lens[i], task.target_lens[i]
                batch_input_ids[i, :, :c_len] = inp["input_ids"]
                batch_audio_mask[i, :c_len] = inp["audio_mask"]
                batch_attention_mask[i, :c_len] = 1

                batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
                batch_audio_mask[B + i, :u_len] = inp["audio_mask"][..., -u_len:]
                batch_attention_mask[B + i, :u_len] = 1

            tokens = torch.full(
                (B, self.config.num_audio_codebook, max(task.target_lens)),
                self.config.audio_mask_id,
                dtype=torch.long,
                device=self.device,
            )

            timesteps = _get_time_steps(
                t_start=0.0,
                t_end=1.0,
                num_step=gen_config.num_step + 1,
                t_shift=gen_config.t_shift,
            ).tolist()
            schedules = []
            for t_len in task.target_lens:
                total_mask = t_len * self.config.num_audio_codebook
                rem = total_mask
                sched = []
                for step in range(gen_config.num_step):
                    num = (
                        rem
                        if step == gen_config.num_step - 1
                        else min(int(np.ceil(total_mask * (timesteps[step + 1] - timesteps[step]))), rem)
                    )
                    sched.append(int(num))
                    rem -= int(num)
                schedules.append(sched)

            layer_ids = torch.arange(self.config.num_audio_codebook, device=self.device).view(1, -1, 1)

            for step in range(gen_config.num_step):
                batch_logits = self(
                    input_ids=batch_input_ids,
                    audio_mask=batch_audio_mask,
                    attention_mask=batch_attention_mask,
                ).logits
                if batch_logits.dtype != torch.float32:
                    batch_logits = batch_logits.to(torch.float32)

                for i in range(B):
                    k = schedules[i][step]
                    if k <= 0:
                        continue

                    c_len, t_len = c_lens[i], task.target_lens[i]
                    c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :]
                    u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :]

                    pred_tokens, scores = self._predict_tokens_with_scoring(c_logits, u_logits, gen_config)
                    scores = scores - (layer_ids * gen_config.layer_penalty_factor)

                    if gen_config.position_temperature > 0.0:
                        scores = _gumbel_sample(scores, gen_config.position_temperature)

                    sample_tokens = tokens[i : i + 1, :, :t_len]
                    scores.masked_fill_(sample_tokens != self.config.audio_mask_id, -float("inf"))

                    _, topk_idx = torch.topk(scores.flatten(), k)
                    flat_tokens = sample_tokens.flatten()
                    flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                    sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                    tokens[i : i + 1, :, :t_len] = sample_tokens
                    batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = sample_tokens
                    batch_input_ids[B + i : B + i + 1, :, :t_len] = sample_tokens

            return [tokens[i, :, : task.target_lens[i]] for i in range(B)]

    session_options = ort.SessionOptions()
    if OMNIVOICE_ONNX_INTRA_OP_THREADS > 0:
        session_options.intra_op_num_threads = OMNIVOICE_ONNX_INTRA_OP_THREADS
    if OMNIVOICE_ONNX_INTER_OP_THREADS > 0:
        session_options.inter_op_num_threads = OMNIVOICE_ONNX_INTER_OP_THREADS
    graph_opt_map = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    session_options.graph_optimization_level = graph_opt_map.get(
        OMNIVOICE_ONNX_GRAPH_OPT,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    )
    available_providers = set(ort.get_available_providers())
    effective_providers = [p for p in providers if p in available_providers]
    if not effective_providers:
        raise RuntimeError(
            f"None of the requested ONNX providers are available. "
            f"requested={providers}, available={sorted(list(available_providers))}"
        )

    ort_session = ort.InferenceSession(
        onnx_model_file,
        sess_options=session_options,
        providers=effective_providers,
    )

    config = OmniVoiceConfig.from_pretrained(model_id)
    model = OmniVoiceONNX(config, ort_session)
    model.eval()

    resolved_path = snapshot_download(
        model_id,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "audio_tokenizer/*",
        ],
    )
    model.text_tokenizer = AutoTokenizer.from_pretrained(model_id)

    audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")
    if not os.path.isdir(audio_tokenizer_path):
        audio_tokenizer_path = "eustlb/higgs-audio-v2-tokenizer"

    model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        audio_tokenizer_path,
        device_map="cpu",
    )
    model.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_tokenizer_path)
    model.sampling_rate = model.feature_extractor.sampling_rate
    model.duration_estimator = RuleDurationEstimator()

    return OmniVoiceONNXRuntimeModel(model, ort_session)


def _load_onnx_runtime_model(runtime: str, model_id: str):
    onnx_model_file = _ensure_onnx_model_file()
    providers = _providers_for_runtime(runtime)
    if runtime == "onnx-gpu" and OMNIVOICE_ONNX_GPU_REQUIRE_CUDA and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "onnx-gpu runtime requires CUDAExecutionProvider in OMNIVOICE_ONNX_GPU_PROVIDERS"
        )
    model = _build_onnx_runtime_model(model_id, onnx_model_file, providers)
    if runtime == "onnx-gpu" and "CUDAExecutionProvider" not in getattr(model, "providers", []):
        msg = (
            "onnx-gpu runtime requested but CUDAExecutionProvider is unavailable; "
            f"active providers={getattr(model, 'providers', providers)}"
        )
        if OMNIVOICE_ONNX_GPU_REQUIRE_CUDA:
            raise RuntimeError(msg)
        logger.warning(msg)
    detail = (
        f"{runtime} runtime (providers={getattr(model, 'providers', providers)}, "
        f"file={onnx_model_file}, model={model_id})"
    )
    return model, detail


@asynccontextmanager
async def lifespan(app: FastAPI):
    if OMNIVOICE_PRELOAD_DEFAULT_RUNTIME:
        logger.info("Preloading OmniVoice default runtime model...")
        try:
            if OMNIVOICE_DEFAULT_RUNTIME in {"onnx-cpu", "onnx-gpu"}:
                model, detail = _load_onnx_runtime_model(
                    OMNIVOICE_DEFAULT_RUNTIME,
                    _model_id_for_runtime(OMNIVOICE_DEFAULT_RUNTIME),
                )
            else:
                model, detail = _load_torch_runtime_model(_model_id_for_runtime("torch"))
            omnivoice_models[OMNIVOICE_DEFAULT_RUNTIME] = model
            _model_details[OMNIVOICE_DEFAULT_RUNTIME] = detail
            logger.info(f"OmniVoice loaded successfully with {detail}")
        except Exception as e:
            logger.error(f"Failed to preload OmniVoice model: {e}")
            raise
    else:
        logger.info(
            "OmniVoice runtime preload disabled; models will lazy-load on first request. "
            f"Default runtime={OMNIVOICE_DEFAULT_RUNTIME}"
        )

    yield
    logger.info("Shutting down OmniVoice server...")


app = FastAPI(title="OmniVoice Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_or_load_model_for_runtime(runtime: str):
    model = omnivoice_models.get(runtime)
    if model is not None:
        return model

    failure = _model_load_failures.get(runtime)
    if failure:
        age = time.time() - float(failure.get("ts", 0.0))
        if age < OMNIVOICE_MODEL_LOAD_RETRY_SECONDS:
            wait_s = int(max(1, OMNIVOICE_MODEL_LOAD_RETRY_SECONDS - age))
            raise RuntimeError(
                f"Previous runtime load failed recently; retry in {wait_s}s. "
                f"Last error: {failure.get('error', 'unknown')}"
            )

    with _model_load_lock:
        model = omnivoice_models.get(runtime)
        if model is not None:
            return model

        failure = _model_load_failures.get(runtime)
        if failure:
            age = time.time() - float(failure.get("ts", 0.0))
            if age < OMNIVOICE_MODEL_LOAD_RETRY_SECONDS:
                wait_s = int(max(1, OMNIVOICE_MODEL_LOAD_RETRY_SECONDS - age))
                raise RuntimeError(
                    f"Previous runtime load failed recently; retry in {wait_s}s. "
                    f"Last error: {failure.get('error', 'unknown')}"
                )

        model_id = _model_id_for_runtime(runtime)
        logger.info(f"Lazy-loading OmniVoice runtime={runtime} model={model_id}...")
        try:
            if runtime in {"onnx-cpu", "onnx-gpu"}:
                model, detail = _load_onnx_runtime_model(runtime, model_id)
            else:
                model, detail = _load_torch_runtime_model(model_id)
        except Exception as e:
            _model_load_failures[runtime] = {
                "ts": time.time(),
                "error": str(e),
            }
            logger.error(f"OmniVoice runtime={runtime} failed to load: {e}")
            raise

        _model_load_failures.pop(runtime, None)
        omnivoice_models[runtime] = model
        _model_details[runtime] = detail
        logger.info(f"OmniVoice runtime={runtime} ready ({detail})")
        return model


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}: {json.dumps(exc.errors(), indent=2)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            },
            "detail": exc.errors(),
        },
    )


@app.get("/")
async def root():
    return {
        "service": "OmniVoice Server",
        "default_runtime": OMNIVOICE_DEFAULT_RUNTIME,
        "loaded_runtimes": sorted(list(omnivoice_models.keys())),
        "runtime_details": _model_details,
        "failed_runtimes": sorted(list(_model_load_failures.keys())),
        "models": {
            "torch": OMNIVOICE_TORCH_MODEL_ID,
            "onnx-cpu": OMNIVOICE_ONNX_MODEL_ID,
            "onnx-gpu": OMNIVOICE_ONNX_MODEL_ID,
        },
        "onnx": {
            "repo": OMNIVOICE_ONNX_REPO_ID,
            "remote_path": OMNIVOICE_ONNX_REMOTE_PATH,
            "local_file": _resolved_onnx_model_file or OMNIVOICE_ONNX_MODEL_FILE,
            "default_num_step": OMNIVOICE_ONNX_NUM_STEP,
            "providers_cpu": OMNIVOICE_ONNX_PROVIDERS,
            "providers_gpu": OMNIVOICE_ONNX_GPU_PROVIDERS,
            "intra_op_threads": OMNIVOICE_ONNX_INTRA_OP_THREADS,
            "inter_op_threads": OMNIVOICE_ONNX_INTER_OP_THREADS,
            "graph_opt": OMNIVOICE_ONNX_GRAPH_OPT,
        },
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "voices": "/v1/voices",
            "speech": "/v1/audio/speech",
            "speech_stream": "/v1/audio/speech/stream",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": bool(omnivoice_models),
        "default_runtime": OMNIVOICE_DEFAULT_RUNTIME,
        "loaded_runtimes": sorted(list(omnivoice_models.keys())),
        "runtime_details": _model_details,
        "failed_runtimes": sorted(list(_model_load_failures.keys())),
        "models": {
            "torch": OMNIVOICE_TORCH_MODEL_ID,
            "onnx-cpu": OMNIVOICE_ONNX_MODEL_ID,
            "onnx-gpu": OMNIVOICE_ONNX_MODEL_ID,
        },
        "onnx_model_file": _resolved_onnx_model_file or OMNIVOICE_ONNX_MODEL_FILE,
        "onnx_repo": OMNIVOICE_ONNX_REPO_ID,
        "onnx_remote_path": OMNIVOICE_ONNX_REMOTE_PATH,
        "onnx_default_num_step": OMNIVOICE_ONNX_NUM_STEP,
        "onnx_providers_cpu": OMNIVOICE_ONNX_PROVIDERS,
        "onnx_providers_gpu": OMNIVOICE_ONNX_GPU_PROVIDERS,
        "onnx_intra_op_threads": OMNIVOICE_ONNX_INTRA_OP_THREADS,
        "onnx_inter_op_threads": OMNIVOICE_ONNX_INTER_OP_THREADS,
        "onnx_graph_opt": OMNIVOICE_ONNX_GRAPH_OPT,
        "output_sample_rate": TARGET_SAMPLE_RATE,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "omnivoice-tts",
                "object": "model",
                "created": 0,
                "owned_by": "k2-fsa",
                "source": {
                    "torch": OMNIVOICE_TORCH_MODEL_ID,
                    "onnx-cpu": OMNIVOICE_ONNX_MODEL_ID,
                    "onnx-gpu": OMNIVOICE_ONNX_MODEL_ID,
                },
            }
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    return {
        "voices": VOICE_GUIDE,
        "notes": {
            "auto": "No voice prompt, model chooses voice automatically",
            "instruct": "Prefix with 'instruct:' for voice design attributes",
            "path": "Provide a local reference WAV path for voice cloning",
        },
    }


@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    runtime = _resolve_runtime_override(request.runtime)
    try:
        model = _get_or_load_model_for_runtime(runtime)
    except Exception as e:
        logger.error(f"Runtime load failed for runtime={runtime}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load OmniVoice runtime '{runtime}': {e}")

    effective_num_step = request.num_step
    if effective_num_step is None and runtime in {"onnx-cpu", "onnx-gpu"}:
        effective_num_step = OMNIVOICE_ONNX_NUM_STEP

    if request.prechunked:
        chunks = [request.input.strip()]
    else:
        chunks = _split_text_into_chunks(
            request.input,
            max_tokens=request.max_tokens,
            token_method=request.token_method,
        )
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text to synthesize")

    prepared_ref_text = _prepare_reference_text_once(request.voice, request.ref_text)

    mode = "prechunked" if request.prechunked else "chunked"
    logger.info(f"Generating {len(chunks)} OmniVoice chunks (mode={mode}, runtime={runtime})...")

    all_audio = []
    start = time.perf_counter()
    total_audio_sec = 0.0
    for idx, chunk in enumerate(chunks, start=1):
        try:
            chunk_start = time.perf_counter()
            chunk_audio = _generate_chunk_audio(
                model,
                chunk,
                request.voice,
                prepared_ref_text,
                request.speed,
                effective_num_step,
            )
            if len(chunk_audio) > 0:
                all_audio.append(chunk_audio)
            chunk_audio_sec = len(chunk_audio) / TARGET_SAMPLE_RATE if len(chunk_audio) > 0 else 0.0
            total_audio_sec += chunk_audio_sec
            chunk_wall = time.perf_counter() - chunk_start
            chunk_xrt = (chunk_audio_sec / chunk_wall) if chunk_wall > 0 else 0.0
            logger.info(f"[{idx}/{len(chunks)}] generated {len(chunk_audio)} samples ({chunk_audio_sec:.2f}s audio in {chunk_wall:.2f}s, {chunk_xrt:.2f}x RT)")
        except Exception as e:
            logger.error(f"[{idx}/{len(chunks)}] chunk failed: {e}")

    if not all_audio:
        raise HTTPException(status_code=500, detail="No audio chunks generated")

    audio_np = np.concatenate(all_audio)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, TARGET_SAMPLE_RATE, audio_np)
    elapsed = time.perf_counter() - start
    xrt = (total_audio_sec / elapsed) if elapsed > 0 else 0.0
    rtf = (elapsed / total_audio_sec) if total_audio_sec > 0 else 0.0
    logger.info(f"Done: {len(audio_np)} samples ({total_audio_sec:.2f}s audio) in {elapsed:.2f}s | speed={xrt:.2f}x RT | RTF={rtf:.3f}")

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@app.post("/v1/audio/speech/stream")
async def create_speech_streaming(request: AudioSpeechRequest):
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    runtime = _resolve_runtime_override(request.runtime)
    try:
        model = _get_or_load_model_for_runtime(runtime)
    except Exception as e:
        logger.error(f"Runtime load failed for runtime={runtime}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load OmniVoice runtime '{runtime}': {e}")

    effective_num_step = request.num_step
    if effective_num_step is None and runtime in {"onnx-cpu", "onnx-gpu"}:
        effective_num_step = OMNIVOICE_ONNX_NUM_STEP

    async def event_generator():
        import base64

        try:
            chunks = _split_text_into_chunks(
                request.input,
                max_tokens=request.max_tokens,
                token_method=request.token_method,
            )
            if not chunks:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No valid text to synthesize'})}\n\n"
                return

            prepared_ref_text = _prepare_reference_text_once(request.voice, request.ref_text)

            yield f"data: {json.dumps({'type': 'start', 'chunks': len(chunks), 'sample_rate': TARGET_SAMPLE_RATE, 'runtime': runtime})}\n\n"

            total_duration = 0.0
            start = time.perf_counter()
            sent = 0

            for idx, chunk in enumerate(chunks):
                try:
                    chunk_start = time.perf_counter()
                    audio_np = _generate_chunk_audio(
                        model,
                        chunk,
                        request.voice,
                        prepared_ref_text,
                        request.speed,
                        effective_num_step,
                    )
                    if len(audio_np) == 0:
                        continue

                    audio_buffer = io.BytesIO()
                    scipy.io.wavfile.write(audio_buffer, TARGET_SAMPLE_RATE, audio_np)
                    audio_bytes = audio_buffer.getvalue()
                    duration = len(audio_np) / TARGET_SAMPLE_RATE
                    total_duration += duration
                    gen_time = time.perf_counter() - chunk_start
                    sent += 1
                    chunk_xrt = (duration / gen_time) if gen_time > 0 else 0.0
                    logger.info(f"[stream {idx+1}/{len(chunks)}] {duration:.2f}s audio in {gen_time:.2f}s ({chunk_xrt:.2f}x RT)")

                    event = {
                        "type": "chunk",
                        "index": idx,
                        "audio_bytes_b64": base64.b64encode(audio_bytes).decode("utf-8"),
                        "duration": round(duration, 2),
                        "generation_time": round(gen_time, 2),
                        "text_preview": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception as e:
                    logger.error(f"Chunk {idx + 1} failed: {e}")
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Chunk {idx + 1} failed: {str(e)}'})}\n\n"

            if sent == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No audio generated'})}\n\n"
                return

            total_time = time.perf_counter() - start
            xrt = (total_duration / total_time) if total_time > 0 else 0.0
            rtf = (total_time / total_duration) if total_duration > 0 else 0.0
            logger.info(f"[stream complete] chunks={sent}, audio={total_duration:.2f}s, wall={total_time:.2f}s, speed={xrt:.2f}x RT, RTF={rtf:.3f}")
            yield f"data: {json.dumps({'type': 'complete', 'chunks_sent': sent, 'audio_duration': round(total_duration, 2), 'total_time': round(total_time, 2), 'xrt': round(xrt, 2), 'rtf': round(rtf, 3)})}\n\n"
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(description="OmniVoice Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8898, help="Port to bind to")
    parser.add_argument(
        "--proxy-headers",
        action="store_true",
        default=True,
        help="Trust X-Forwarded-* headers from reverse proxy (Tailscale, nginx, etc.)",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default="*",
        help="IPs allowed to send forwarded headers",
    )
    args = parser.parse_args()

    logger.info(f"Starting OmniVoice Server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level=os.getenv("VF_UVICORN_LOG_LEVEL", "warning").lower(),
        access_log=_env_flag("VF_ACCESS_LOGS", "0"),
    )
