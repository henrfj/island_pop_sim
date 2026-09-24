"""Clan membership assignment and lineage-readable branch naming."""
import numpy as np

from .config import TribeConfig


def allocate_mixed_children(birth_counts: np.ndarray, first_parent_weight: float,
                            local_first_share: float, cfg: TribeConfig,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    parental = min(max(first_parent_weight, 0.0), 1.0)
    local = min(max(local_first_share, 0.0), 1.0)
    first_probability = ((1.0 - cfg.local_child_influence) * parental
                         + cfg.local_child_influence * local)
    first = rng.binomial(np.asarray(birth_counts, dtype=np.int64), first_probability)
    return first, np.asarray(birth_counts, dtype=np.int64) - first


def unique_branch_name(parent: str, existing: set[str], cfg: TribeConfig) -> str:
    root = parent.split(" ", 1)[0]
    for suffix in cfg.secession_suffixes:
        candidate = f"{root} {suffix}"
        if candidate not in existing:
            return candidate
    index = 2
    while f"{root} {index}" in existing:
        index += 1
    return f"{root} {index}"