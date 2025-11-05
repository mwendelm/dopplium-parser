"""Dopplium Parser - Parse Dopplium radar data formats."""

from .parse_dopplium_raw import (
    parse_dopplium_raw,
    FileHeader,
    BodyHeader,
    FrameHeader,
)

__version__ = "1.0.0"
__all__ = [
    "parse_dopplium_raw",
    "FileHeader",
    "BodyHeader",
    "FrameHeader",
]

