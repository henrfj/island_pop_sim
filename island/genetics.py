"""Core Mendelian genetics: reproduction under random mating, with optional per-genotype
fertility weighting (the "aesthetic preference" selection knob).

Genotype convention throughout the island sim: (n_BB, n_Bb, n_bb) integer counts.
B = brown allele (dominant), b = blue allele (recessive; the blue phenotype is exactly bb).

This is the same exact multinomial-sampling model validated against Hardy-Weinberg
equilibrium in Part 1's eye_color_sim.py, generalized with an optional fertility weight
per genotype: see effective_allele_frequency for the derivation.
"""
from typing import Tuple
import numpy as np

GENOTYPES = ("BB", "Bb", "bb")


def effective_allele_frequency(n_BB: int, n_Bb: int, n_bb: int,
                                fertility_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Blue-allele frequency (q) among a population where parents are chosen for mating
    proportional to `fertility_weights` per genotype (1.0 each = neutral / unweighted).

    A Bb parent contributes a 'b' allele half the time; a bb parent always does. Weighting
    parent selection by fertility_weights just re-weights that average.
    """
    w_BB, w_Bb, w_bb = fertility_weights
    weighted_total = n_BB * w_BB + n_Bb * w_Bb + n_bb * w_bb
    if weighted_total <= 0:
        return 0.0
    q = (n_Bb * w_Bb * 0.5 + n_bb * w_bb) / weighted_total
    return min(max(q, 0.0), 1.0)


def multinomial_birth_counts(n_BB: int, n_Bb: int, n_bb: int, births: int, rng: np.random.Generator,
                              fertility_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                              ) -> Tuple[int, int, int]:
    """Draw `births` offspring via random mating. Each offspring's two alleles are i.i.d.
    Bernoulli(q) draws (parents chosen uniformly, or weighted by fertility_weights, then
    contributing one of their own two alleles at random) -- so genotype counts among the
    births are exactly Multinomial(births, [p^2, 2pq, q^2]), the Hardy-Weinberg proportions,
    generated as a random (drift-prone) process rather than assumed as a fixed formula.
    """
    if n_BB + n_Bb + n_bb == 0 or births <= 0:
        return (0, 0, 0)
    q = effective_allele_frequency(n_BB, n_Bb, n_bb, fertility_weights)
    p = 1.0 - q
    counts = rng.multinomial(births, (p * p, 2 * p * q, q * q))
    return int(counts[0]), int(counts[1]), int(counts[2])
