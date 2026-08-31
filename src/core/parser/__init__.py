"""
Lightweight Legal Contract Parser Module.
Replaces legacy multi-minute Dedoc/Camelot/OCR pipeline with high-speed (<0.03s)
PyMuPDF extraction and stateful regex hierarchy builder.
"""
from src.core.parser.lightweight_parser import (
    parse_and_chunk_contract,
    parse_and_chunk_text,
    HEADING_REGEX,
    SUBCLAUSE_REGEX,
    PREAMBLE_REGEX,
)

__all__ = [
    "parse_and_chunk_contract",
    "parse_and_chunk_text",
    "HEADING_REGEX",
    "SUBCLAUSE_REGEX",
    "PREAMBLE_REGEX",
]