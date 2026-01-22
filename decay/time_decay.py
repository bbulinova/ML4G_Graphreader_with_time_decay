from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math

from preprocessing.temporal import TemporalFact


def exp_decay_weight(timestamp: int, t_now: int, lamb: float = 0.3) -> float:
    """
    Exponential decay weight:
      w = exp(-lamb * (t_now - timestamp))
    If timestamp is in the future (shouldn't happen), clamp age at 0.
    """
    age = max(0, t_now - timestamp)
    return math.exp(-lamb * age)


@dataclass
class WeightedFact(TemporalFact):
    weight: float


def apply_time_decay(
    temporal_facts: List[TemporalFact],
    t_now: int,
    lamb: float = 0.3,
) -> List[WeightedFact]:
    """
    Attach a decay weight to each temporal fact.
    """
    out: List[WeightedFact] = []
    for f in temporal_facts:
        w = exp_decay_weight(f.timestamp, t_now=t_now, lamb=lamb)
        out.append(
            WeightedFact(
                fact_id=f.fact_id,
                chunk_id=f.chunk_id,
                text=f.text,
                timestamp=f.timestamp,
                weight=w,
            )
        )
    return out