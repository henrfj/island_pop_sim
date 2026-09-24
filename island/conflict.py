"""Temporary valley-wide conflict factions that cut across clan membership."""
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class FactionSplit:
    first: Dict[str, np.ndarray]
    second: Dict[str, np.ndarray]
    first_probabilities: Dict[str, float]


def partition_factions(cohorts: Dict[str, np.ndarray], clan_cohesion: float,
                       rng: np.random.Generator) -> FactionSplit:
    """Split every clan across two temporary sides using clan identity as a soft pull."""
    cohesion = min(max(clan_cohesion, 0.0), 1.0)
    valley_split = float(rng.uniform(0.35, 0.65))
    first, second, probabilities = {}, {}, {}
    for clan, counts in cohorts.items():
        clan_lean = float(rng.uniform(0.15, 0.85))
        probability = (1.0 - cohesion) * valley_split + cohesion * clan_lean
        probability = min(max(probability, 0.10), 0.90)
        first_counts = rng.binomial(counts, probability).astype(np.int64)
        first[clan] = first_counts
        second[clan] = counts - first_counts
        probabilities[clan] = probability
    return FactionSplit(first, second, probabilities)