"""Generic weighted sampling-without-replacement utility.

Used for viability selection at culling time (each genotype can have a different
"survival weight") and for uniform surplus draws during migration (weights all equal
to 1.0 there, which collapses to the same exact multivariate-hypergeometric sampling
used throughout the single-valley sim in Part 1).
"""
import numpy as np


def weighted_sample_without_replacement_counts(counts, weights, k: int,
                                                rng: np.random.Generator) -> np.ndarray:
    """Pick `k` individuals without replacement from a population partitioned into
    categories with per-category `counts` and `weights` (higher weight = more likely
    to be picked). Returns how many were picked from each category.

    Uses the Efraimidis-Spirakis method: give every individual a key U^(1/w)
    (U ~ Uniform(0,1)) and take the k individuals with the largest keys. Individuals in
    the same category share the same weight, so this is vectorized per-category rather
    than looping over individuals one at a time.
    """
    counts = np.asarray(counts, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    total = int(counts.sum())
    k = int(k)
    if k >= total:
        return counts.copy()
    if k <= 0:
        return np.zeros_like(counts)

    # Fast, exact path when every category is weighted equally: uniform sampling without
    # replacement is exactly the multivariate hypergeometric distribution.
    if np.allclose(weights, weights[0]):
        return rng.multivariate_hypergeometric(counts, k)

    keys_chunks, group_chunks = [], []
    for group_idx, (count, weight) in enumerate(zip(counts, weights)):
        if count <= 0:
            continue
        w = max(float(weight), 1e-9)
        u = rng.random(count)
        keys_chunks.append(u ** (1.0 / w))
        group_chunks.append(np.full(count, group_idx, dtype=np.int64))

    keys = np.concatenate(keys_chunks)
    groups = np.concatenate(group_chunks)
    top = np.argpartition(-keys, k - 1)[:k]
    chosen_groups = groups[top]
    return np.array([np.sum(chosen_groups == gi) for gi in range(len(counts))], dtype=np.int64)
