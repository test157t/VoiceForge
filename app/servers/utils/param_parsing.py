"""
Unified parameter parsing utilities for backend endpoints.

Provides consistent parsing, validation, and default handling for all endpoints.
"""

import json
from typing import Any, Dict, Optional

from servers.models.params import (
    RVCParams,
    PostProcessParams,
    get_default_rvc_params_dict,
    get_default_post_params_dict,
)


def parse_json_params(param_string: Optional[str], default_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse JSON parameter string with strict validation.
    
    Args:
        param_string: JSON string to parse (can be None)
        default_dict: Default values to use when param_string is None
        
    Returns:
        Dictionary of parsed parameters merged with defaults
    """
    if not param_string:
        return default_dict.copy()
    
    try:
        parsed = json.loads(param_string)
        # Merge with defaults to ensure all required keys exist
        result = default_dict.copy()
        result.update(parsed)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON parameter payload: {e}") from e


def parse_rvc_params(param_string: Optional[str]) -> RVCParams:
    """
    Parse RVC parameters from JSON string.
    
    Args:
        param_string: JSON string of RVC parameters
        
    Returns:
        RVCParams dataclass instance
    """
    defaults = get_default_rvc_params_dict()
    parsed = parse_json_params(param_string, defaults)
    return RVCParams.from_dict(parsed)


def parse_post_process_params(param_string: Optional[str]) -> PostProcessParams:
    """
    Parse post-processing parameters from JSON string.
    
    Args:
        param_string: JSON string of post-processing parameters
        
    Returns:
        PostProcessParams dataclass instance
    """
    defaults = get_default_post_params_dict()
    parsed = parse_json_params(param_string, defaults)
    return PostProcessParams.from_dict(parsed)


def get_rvc_params_dict(param_string: Optional[str]) -> Dict[str, Any]:
    """
    Get RVC parameters as dictionary (for endpoints that need dict format).
    
    Args:
        param_string: JSON string of RVC parameters
        
    Returns:
        Dictionary of RVC parameters
    """
    params = parse_rvc_params(param_string)
    return params.to_dict()


def get_post_process_params_dict(param_string: Optional[str]) -> Dict[str, Any]:
    """
    Get post-processing parameters as dictionary (for endpoints that need dict format).
    
    Args:
        param_string: JSON string of post-processing parameters
        
    Returns:
        Dictionary of post-processing parameters
    """
    params = parse_post_process_params(param_string)
    return params.to_dict()

