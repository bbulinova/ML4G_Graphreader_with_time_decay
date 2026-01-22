from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

from preprocessing.fact_extraction import AtomicFact


@dataclass
class TemporalFact(AtomicFact):
    timestamp: int


def assign_random_timestamps(
    facts: List[AtomicFact],
    t_min: int = 0,
    t_max: int = 10,
    seed: int = 0,
) -> List[TemporalFact]:
    """
    Assign a random timestamp to each atomic fact.
    """
    rng = random.Random(seed)
    temporal_facts: List[TemporalFact] = []

    for f in facts:
        ts = rng.randint(t_min, t_max)
        temporal_facts.append(
            TemporalFact(
                fact_id=f.fact_id,
                chunk_id=f.chunk_id,
                text=f.text,
                timestamp=ts,
            )
        )

    return temporal_facts
