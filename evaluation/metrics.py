from __future__ import annotations

from typing import List
from query.retrieve import ScoredFact


def contains_answer(text: str, answer: str) -> bool:
    """Simple string match (case-insensitive)."""
    return answer.strip().lower() in text.lower()


def hit_at_k(ranked: List[ScoredFact], answer: str, k: int) -> bool:
    topk = ranked[:k]
    return any(contains_answer(r.text, answer) for r in topk)
