# atomic facts + timestamps
from dataclasses import dataclass
from typing import List

@dataclass
class FactNode:
    def __init__(self, fact_id, text, chunk_id, timestamp, score=0.0):
        self.fact_id = fact_id
        self.text = text
        self.chunk_id = chunk_id
        self.timestamp = timestamp
        self.score = score
        self.neighbors = []