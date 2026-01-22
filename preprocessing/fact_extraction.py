from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re

from preprocessing.chunking import Chunk


@dataclass
class AtomicFact:
    fact_id: int
    chunk_id: int
    text: str


# Split on punctuation followed by whitespace.
# (Good enough for now; later you can refine.)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_atomic_facts_from_chunks(chunks: List[Chunk], min_len: int = 10) -> List[AtomicFact]:
    """
    Approximate atomic facts by sentence splitting within each chunk.
    """
    facts: List[AtomicFact] = []
    fact_id = 0

    for ch in chunks:
        # Normalize whitespace so sentence splitting behaves predictably
        normalized = " ".join(ch.text.split())
        sentences = _SENT_SPLIT.split(normalized)

        for s in sentences:
            s = s.strip()
            if len(s) < min_len:
                continue
            facts.append(AtomicFact(fact_id=fact_id, chunk_id=ch.chunk_id, text=s))
            fact_id += 1

    return facts