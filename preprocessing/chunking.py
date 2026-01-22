from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: int
    text: str


def chunk_text_preserve_paragraphs(full_text: str, max_chars: int = 1200) -> List[Chunk]:
    """
    Chunk text while preserving paragraph boundaries.

    - Split by newline into paragraphs
    - Accumulate paragraphs until adding another would exceed max_chars
    - If a single paragraph is longer than max_chars, hard-split it
    """
    paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
    chunks: List[Chunk] = []

    buf: List[str] = []
    buf_len = 0
    chunk_id = 0

    def flush():
        nonlocal chunk_id, buf, buf_len
        if buf:
            chunks.append(Chunk(chunk_id=chunk_id, text="\n".join(buf)))
            chunk_id += 1
            buf = []
            buf_len = 0

    for p in paragraphs:
        # If paragraph alone is too long: flush buffer and split paragraph
        if len(p) > max_chars:
            flush()
            start = 0
            while start < len(p):
                part = p[start : start + max_chars]
                chunks.append(Chunk(chunk_id=chunk_id, text=part))
                chunk_id += 1
                start += max_chars
            continue

        # If adding this paragraph would overflow the chunk -> flush first
        extra = len(p) + (1 if buf else 0)  # +1 for newline join
        if buf_len + extra > max_chars:
            flush()

        buf.append(p)
        buf_len += extra

    flush()
    return chunks