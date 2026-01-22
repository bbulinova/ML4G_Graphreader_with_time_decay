from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: int
    text: str


def chunk_text_preserve_paragraphs(full_text: str, max_chars: int = 1800) -> List[Chunk]:
    """
    Simple chunker:
    - split by newline into paragraphs
    - pack paragraphs into chunks up to ~max_chars
    - preserves paragraph boundaries
    """
    paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
    chunks: List[Chunk] = []

    buf: List[str] = []
    buf_len = 0
    chunk_id = 0

    for p in paragraphs:
        # if a single paragraph is longer than max_chars, hard-split it
        if len(p) > max_chars:
            # flush current buffer first
            if buf:
                chunks.append(Chunk(chunk_id=chunk_id, text="\n".join(buf)))
                chunk_id += 1
                buf, buf_len = [], 0

            start = 0
            while start < len(p):
                part = p[start : start + max_chars]
                chunks.append(Chunk(chunk_id=chunk_id, text=part))
                chunk_id += 1
                start += max_chars
            continue

        # if adding this paragraph would exceed max, flush buffer
        if buf_len + len(p) + (1 if buf else 0) > max_chars:
            chunks.append(Chunk(chunk_id=chunk_id, text="\n".join(buf)))
            chunk_id += 1
            buf, buf_len = [], 0

        buf.append(p)
        buf_len += len(p) + (1 if buf_len > 0 else 0)

    # flush remainder
    if buf:
        chunks.append(Chunk(chunk_id=chunk_id, text="\n".join(buf)))

    return chunks