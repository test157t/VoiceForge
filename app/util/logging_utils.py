# logging_utils.py
"""
Centralized logging configuration for VoiceForge.

ALL logging logic for the entire program lives here.
Import and use these functions instead of raw logging/print statements.
"""

import warnings
import logging
import sys
import os
from typing import Optional

# ============================================
# VOICEFORGE LOGGER
# ============================================

# Main VoiceForge logger
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "voiceforge") -> logging.Logger:
    """
    Get a logger instance for VoiceForge.
    
    Args:
        name: Logger name (default: 'voiceforge')
    
    Returns:
        Configured Logger instance
    """
    return logging.getLogger(name)


def setup_logger(
    name: str = "voiceforge",
    level: int = logging.INFO,
    format_string: str = "[%(name)s] %(message)s"
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Log message format
    
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add stream handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(handler)
    
    return logger


# ============================================
# LOG FUNCTIONS (consistent interface)
# ============================================

def log_info(msg: str, prefix: str = ""):
    """Log info message."""
    level_name = os.getenv("VF_SERVER_LOG_LEVEL", "WARNING").strip().upper()
    if level_name not in {"DEBUG", "INFO"}:
        return
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}{msg}", flush=True)


def log_warn(msg: str, prefix: str = ""):
    """Log warning message."""
    level_name = os.getenv("VF_SERVER_LOG_LEVEL", "WARNING").strip().upper()
    if level_name not in {"DEBUG", "INFO", "WARNING"}:
        return
    tag = f"[{prefix} WARNING] " if prefix else "[WARNING] "
    print(f"{tag}{msg}", flush=True)


def log_error(msg: str, prefix: str = ""):
    """Log error message."""
    tag = f"[{prefix} ERROR] " if prefix else "[ERROR] "
    print(f"{tag}{msg}", flush=True)


def log_debug(msg: str, prefix: str = ""):
    """Log debug message."""
    level_name = os.getenv("VF_SERVER_LOG_LEVEL", "WARNING").strip().upper()
    if level_name != "DEBUG":
        return
    tag = f"[{prefix} DEBUG] " if prefix else "[DEBUG] "
    print(f"{tag}{msg}", flush=True)


# ============================================
# WARNING SUPPRESSION
# ============================================

# Noisy loggers to suppress
NOISY_LOGGERS = [
    # NVIDIA libraries
    "nv_one_logger", "nv_one_logger.api", "nv_one_logger.exporter",
    # ModelScope
    "modelscope", "modelscope.utils", "modelscope.pipelines", "modelscope.models",
    # ML frameworks
    "transformers", "torch", "torchaudio", "fairseq",
    "librosa", "numba", "faiss",
    # Other
    "urllib3", "asyncio", "PIL",
]


def configure_warnings():
    """
    Configure warning filters for third-party packages.
    
    Should be called early in the program before importing libraries
    that generate warnings (faiss, setuptools, pyworld, etc.)
    """
    # Deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*distutils.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*LooseVersion.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="setuptools")
    
    # User warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")


def configure_server_warnings():
    """
    Configure additional warning filters for server mode.
    
    Includes extra filters for torch.compile and other
    server-specific noisy libraries.
    """
    configure_warnings()
    
    # Training warnings (we only do inference)
    warnings.filterwarnings("ignore", message=".*If you intend to do training.*")
    warnings.filterwarnings("ignore", message=".*If you intend to do validation.*")
    
    # ModelScope/torch warnings
    warnings.filterwarnings("ignore", message=".*torch.compile.*")
    warnings.filterwarnings("ignore", message=".*triton.*")
    warnings.filterwarnings("ignore", message=".*safetensors.*")
    warnings.filterwarnings("ignore", message=".*No preprocessor.*")
    warnings.filterwarnings("ignore", message=".*preprocessor.*")
    warnings.filterwarnings("ignore", message=".*PREPROCESSOR_MAP.*")
    warnings.filterwarnings("ignore", message=".*Downloading Model.*")
    warnings.filterwarnings("ignore", message=".*initiate model.*")
    warnings.filterwarnings("ignore", message=".*initialize model.*")


def configure_logging(level: int = logging.WARNING):
    """
    Configure logging to suppress noisy library output.
    
    Args:
        level: Base logging level (default: WARNING)
    """
    logging.basicConfig(level=level, format='%(message)s', force=True)
    logging.getLogger().setLevel(level)
    
    # Suppress noisy loggers
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def suppress_all_logging():
    """Suppress all logging (for silent operation)."""
    logging.disable(logging.CRITICAL)


def suppress_library_loggers():
    """Suppress only third-party library loggers."""
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


# ============================================
# SERVER-SPECIFIC LOGGING
# ============================================

def create_server_logger(
    server_name: str,
    level: int = logging.INFO
) -> tuple:
    """
    Create logging functions for a server.
    
    Args:
        server_name: Server identifier (e.g., 'ASR', 'RVC', 'Chatterbox')
        level: Logging level
    
    Returns:
        Tuple of (log_info, log_warn, log_error) functions
    """
    def _effective_level() -> str:
        return os.getenv("VF_SERVER_LOG_LEVEL", "WARNING").strip().upper()

    def _log_info(msg: str):
        if _effective_level() in {"DEBUG", "INFO"}:
            print(f"[{server_name}] {msg}", flush=True)
    
    def _log_warn(msg: str):
        if _effective_level() in {"DEBUG", "INFO", "WARNING"}:
            print(f"[{server_name} WARNING] {msg}", flush=True)
    
    def _log_error(msg: str):
        print(f"[{server_name} ERROR] {msg}", flush=True)
    
    return _log_info, _log_warn, _log_error


# ============================================
# ENVIRONMENT-SPECIFIC SETUP
# ============================================

def setup_asr_logging():
    """Configure logging for ASR environment."""
    import os
    os.environ['HYDRA_FULL_ERROR'] = '0'
    os.environ['ONE_LOGGER_ENABLED'] = 'false'
    os.environ['DISABLE_ONE_LOGGER'] = '1'


def setup_rvc_logging():
    """Configure logging for RVC environment."""
    import os
    os.environ["ONNXRUNTIME_LOGLEVEL"] = "3"
    os.environ["ORT_LOG_LEVEL"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_general_logging():
    """Configure logging for general/main environment."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ============================================
# OUTPUT SUPPRESSION CONTEXT MANAGER
# ============================================

class SuppressOutput:
    """Context manager to suppress stdout/stderr."""
    
    def __init__(self, suppress_stdout: bool = True, suppress_stderr: bool = True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None
    
    def __enter__(self):
        import io
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = io.StringIO()
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        if self._stdout:
            sys.stdout = self._stdout
        if self._stderr:
            sys.stderr = self._stderr


def suppress_output():
    """Get a context manager to suppress output."""
    return SuppressOutput()


__all__ = [
    # Logger setup
    'get_logger',
    'setup_logger',
    # Log functions
    'log_info',
    'log_warn', 
    'log_error',
    'log_debug',
    # Warning configuration
    'configure_warnings',
    'configure_server_warnings',
    'configure_logging',
    # Suppression
    'suppress_all_logging',
    'suppress_library_loggers',
    'suppress_output',
    'SuppressOutput',
    # Server-specific
    'create_server_logger',
    # Environment setup
    'setup_asr_logging',
    'setup_rvc_logging',
    'setup_general_logging',
    # Constants
    'NOISY_LOGGERS',
]
