"""Helpers for preparing text for TTS synthesis."""

from __future__ import annotations

import re


_INTIFACE_BLOCK_RE = re.compile(
    r"<\s*intiface_commands\s*>.*?<\s*/\s*intiface_commands\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)

_COMMAND_TAG_BLOCK_RE = re.compile(
    r"<\s*(device|interface|media)\b[^>]*>.*?<\s*/\s*\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)

_COMMAND_TAG_INLINE_RE = re.compile(
    r"<\s*(device|interface|media)\s*:[^>]*>",
    flags=re.IGNORECASE,
)

_GENERIC_COLON_TAG_RE = re.compile(
    r"<\s*[a-z0-9_\-]+\s*:[^>]*>",
    flags=re.IGNORECASE,
)

_BARE_COMMAND_PREFIX_RE = re.compile(
    r"\b(?:any|device|interface|media)\s*:\s*(?:waveform|basic|preset|dual|gradient|vibrate|oscillate|linear|pattern|stop|intensity|scan|connect|disconnect|start)\b[^.!?\n]*",
    flags=re.IGNORECASE,
)

_BARE_COMMAND_RE = re.compile(
    r"\b(?:waveform|basic|preset|dual|gradient|vibrate|oscillate|linear|pattern)\s*:[^.!?\n]*",
    flags=re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def strip_nonspoken_tags(text: str) -> str:
    """
    Remove non-spoken control tags from text before TTS.

    Keeps normal prose intact while stripping command/control content such as:
    - <intiface_commands>...</intiface_commands>
    - <device:...>, <interface:...>, <media:...>
    - block forms like <device ...>...</device>
    """
    value = str(text or "")
    if not value:
        return ""

    value = _INTIFACE_BLOCK_RE.sub(" ", value)
    value = _COMMAND_TAG_BLOCK_RE.sub(" ", value)
    value = _COMMAND_TAG_INLINE_RE.sub(" ", value)
    value = _GENERIC_COLON_TAG_RE.sub(" ", value)
    value = _BARE_COMMAND_PREFIX_RE.sub(" ", value)
    value = _BARE_COMMAND_RE.sub(" ", value)
    value = _WS_RE.sub(" ", value).strip()
    value = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", value)
    return value
