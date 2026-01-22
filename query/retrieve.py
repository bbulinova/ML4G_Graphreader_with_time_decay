from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable, Tuple
import re

from preprocessing.fact_extraction import AtomicFact
from decay.time_decay import WeightedFact


_WORD = re.compile(r"[A-Za-z']+")

# Tiny stopword set (keep it minimal + transparent)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "was",
    "are", "were", "be", "been", "being", "that", "this", "it", "as", "by",
    "with", "from", "at", "which", "what", "who", "whom", "when", "where",
    "why", "how"
}

def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _WORD.findall(text)]
    return [t for t in toks if t not in _STOPWORDS and len(t) >= 3]


def keyword_overlap_score(question: str, fact_text: str) -> int:
    q = set(tokenize(question))
    f = set(tokenize(fact_text))
    return len(q.intersection(f))


@dataclass
class ScoredFact:
    fact_id: int
    chunk_id: int
    text: str
    score: float


def rank_facts_no_decay(question: str, facts: List[AtomicFact], top_k: int = 5) -> List[ScoredFact]:
    """
    Baseline retrieval: score = keyword overlap
    """
    scored: List[ScoredFact] = []
    for f in facts:
        s = float(keyword_overlap_score(question, f.text))
        if s > 0:
            scored.append(ScoredFact(fact_id=f.fact_id, chunk_id=f.chunk_id, text=f.text, score=s))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def rank_facts_with_decay(question: str, facts: List[WeightedFact], top_k: int = 5) -> List[ScoredFact]:
    """
    Time-aware retrieval: score = overlap * decay_weight
    """
    scored: List[ScoredFact] = []
    for f in facts:
        overlap = float(keyword_overlap_score(question, f.text))
        s = overlap * float(f.weight)
        if s > 0:
            scored.append(ScoredFact(fact_id=f.fact_id, chunk_id=f.chunk_id, text=f.text, score=s))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]