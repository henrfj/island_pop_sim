"""The island: a ring of valleys, each tracking genotype counts (split by native/arrived
origin), connected by mountain paths (ring-neighbors) and canoe routes (everyone else, at
some risk of storm loss).

Generation pipeline (Island.step) -- see README.md for the full diagram:
  volcano -> migration (+ storm risk) -> update overcrowding streak ->
  emigration-or-war (mutually exclusive per valley) -> gradual cull -> birth -> advance timers

Birth is deliberately LAST: a generation's actual living residents experience the
disaster and get to react to it (flee, fight, get culled) before anyone reproduces --
not the other way around, which is what a birth-right-after-the-eruption ordering would
imply (children reacting to a disaster their parents lived through, instead of the
parents themselves).

See migration.py and events.py for the mechanics; genetics.py and sampling.py for the
underlying exact-random-sampling math.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import math
import numpy as np

from . import events, genetics, migration, sampling, selection
from .config import SimulationConfig, ValleySpec

N_GENOTYPES = 3


@dataclass
class Valley:
    """Population is tracked as two parallel genotype triplets: `native_*` (born in this
    valley this generation) and `arrived_*` (born elsewhere, arrived this generation via
    migration or as war refugees). Because generations are non-overlapping -- every
    generation's entire population is freshly drawn from the previous generation's parents
    -- this distinction naturally resets every generation: it needs no history beyond "did
    you just move here or not". See selection.survival_weights_flat for how it's used
    (locally-born can get a survival edge at culling), and events.py for how eruptions,
    wars, and emigration all preserve it automatically by operating on the combined vector.
    """
    name: str
    base_capacity: int
    native_BB: int = 0
    native_Bb: int = 0
    native_bb: int = 0
    arrived_BB: int = 0
    arrived_Bb: int = 0
    arrived_bb: int = 0
    eruption_this_gen: bool = False
    war_occurred_this_gen: bool = False
    emigration_this_gen: bool = False
    devastated_gens_remaining: int = 0  # ~0 capacity for this many generations after erupting
    fertility_bonus_remaining_gens: int = 0
    over_capacity_streak: int = 0  # consecutive generations spent above capacity
    fill_history: List[float] = field(default_factory=list)  # past pop/capacity ratios, oldest
                                                               # first, one per generation -- see
                                                               # World._births_for / GrowthConfig

    @property
    def flat_counts(self) -> np.ndarray:
        """[native_BB, native_Bb, native_bb, arrived_BB, arrived_Bb, arrived_bb]"""
        return np.array([self.native_BB, self.native_Bb, self.native_bb,
                          self.arrived_BB, self.arrived_Bb, self.arrived_bb], dtype=np.int64)

    def set_flat_counts(self, vec) -> None:
        (self.native_BB, self.native_Bb, self.native_bb,
         self.arrived_BB, self.arrived_Bb, self.arrived_bb) = (int(x) for x in vec)

    def set_native(self, counts3) -> None:
        self.native_BB, self.native_Bb, self.native_bb = (int(x) for x in counts3)

    def set_arrived(self, counts3) -> None:
        self.arrived_BB, self.arrived_Bb, self.arrived_bb = (int(x) for x in counts3)

    @property
    def counts(self) -> np.ndarray:
        """Aggregate genotype counts, native + arrived combined: [BB, Bb, bb]."""
        return np.array([self.native_BB + self.arrived_BB,
                          self.native_Bb + self.arrived_Bb,
                          self.native_bb + self.arrived_bb], dtype=np.int64)

    @property
    def population(self) -> int:
        return int(self.flat_counts.sum())

    @property
    def q(self) -> float:
        """Blue-allele frequency, over the combined (native + arrived) population."""
        n_BB, n_Bb, n_bb = self.counts
        n = n_BB + n_Bb + n_bb
        if n == 0:
            return 0.0
        return (2 * n_bb + n_Bb) / (2 * n)


class Island:
    def __init__(self, valley_specs: List[ValleySpec], config: SimulationConfig, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self.valleys = [Valley(v.name, v.base_capacity, native_BB=v.init_BB, native_Bb=v.init_Bb,
                                native_bb=v.init_bb,
                                fertility_bonus_remaining_gens=(config.volcano.fertility_decay_gens
                                                                 if v.starts_with_ash_buff else 0))
                         for v in valley_specs]
        self.n = len(self.valleys)
        # Ring topology: each valley's mountain-path neighbors are the two adjacent valleys
        # (by index order); every other pair is reachable only by the (costlier, riskier) canoe route.
        self.path_neighbors = {i: {(i - 1) % self.n, (i + 1) % self.n} for i in range(self.n)}
        self.history: List[dict] = []
        self.event_log: List[dict] = []

    def travel_cost(self, i: int, j: int) -> float:
        return self.cfg.migration.path_cost if j in self.path_neighbors[i] else self.cfg.migration.water_cost

    def current_capacity(self, valley: Valley) -> float:
        if valley.devastated_gens_remaining > 0:
            # Functionally uninhabitable: the floor below still applies so nothing divides
            # by zero, but at this scale it reads as "practically nobody can live here."
            return 1.0
        cap = float(valley.base_capacity)
        if valley.fertility_bonus_remaining_gens > 0:
            bonus = self.cfg.volcano.fertility_bonus * (
                valley.fertility_bonus_remaining_gens / self.cfg.volcano.fertility_decay_gens)
            cap *= (1.0 + bonus)
        return max(1.0, cap)

    def _mean_kids_per_couple(self, x: float) -> float:
        """Smoothly decreasing from `kids_peak` (x=0) to `kids_at_capacity` (x=1), then on
        down toward `kids_floor` as x grows past 1 -- an exponential decay pinned to those
        two anchor points. See GrowthConfig for the reasoning."""
        g = self.cfg.growth
        span = g.kids_peak - g.kids_floor
        if span <= 1e-9:
            return g.kids_at_capacity
        ratio = min(max((g.kids_at_capacity - g.kids_floor) / span, 1e-6), 1.0 - 1e-6)
        tau = -1.0 / math.log(ratio)
        return g.kids_floor + span * math.exp(-x / tau)

    def _births_for(self, population: int, x: float, rng: np.random.Generator) -> int:
        """Pair the population into `population // 2` couples (any unpaired leftover
        doesn't reproduce) and draw the generation's total births as
        Poisson(couples * mean_kids_per_couple(x)) -- a sum of iid per-couple litter draws
        is itself Poisson with the summed rate, so this is equivalent to (and cheaper than)
        drawing each couple's litter separately, while still making the total a genuine
        random variable instead of a rounded formula.

        `x` is deliberately not necessarily *this* generation's fill fraction -- see
        Island.step's birth section: couples decide family size based on a delayed
        perception of crowding (fill_lag_gens generations stale), not real-time capacity
        tracking nobody in the valley actually has. That lag is what lets population
        overshoot capacity and correct afterward instead of snapping to equilibrium every
        generation -- a valley doesn't stop having kids the instant it gets crowded, and it
        doesn't start again the instant a disaster leaves it empty."""
        if population <= 0:
            return 0
        couples = population // 2
        if couples <= 0:
            return 0
        mean_kids = self._mean_kids_per_couple(x)
        return int(rng.poisson(couples * mean_kids))

    def step(self, gen_index: int) -> None:
        for v in self.valleys:
            v.eruption_this_gen = False
            v.war_occurred_this_gen = False
            v.emigration_this_gen = False

        # 1. Volcano: a fixed clockwise rotation, not per-valley random chance. Hits the
        # ACTUAL living residents (last generation's newborns, now grown into this
        # generation's population) -- the same people who get to react below, not a
        # cohort that hasn't been born yet.
        erupting_idx = events.scheduled_eruption_index(gen_index, self.n, self.cfg.volcano)
        if erupting_idx is not None:
            v = self.valleys[erupting_idx]
            events.erupt(v, self.cfg.volcano, self.rng)
            self.event_log.append({"gen": gen_index, "valley": v.name, "type": "volcano"})

        # 2. Migration -- a family/gathering-sized slice of any surplus tries to move;
        # canoe legs risk storms. Logged directly (already stamped with gen_index).
        self.event_log.extend(migration.migrate(self, self.rng, gen_index))

        # 3. Update each valley's "consecutive generations over capacity" streak, which
        # drives both the emigration and war probabilities below.
        for v in self.valleys:
            cap_now = int(round(self.current_capacity(v)))
            v.over_capacity_streak = v.over_capacity_streak + 1 if v.population > cap_now else 0

        # 4. Emigration (checked first, a gentler release valve) or war -- mutually
        # exclusive per valley per generation.
        for v in self.valleys:
            if self.cfg.emigration.enabled:
                emigrants = events.maybe_emigrate(v, self.cfg.emigration, self.rng)
                if emigrants is not None:
                    self.event_log.append({"gen": gen_index, "valley": v.name, "type": "emigration",
                                            "left": int(emigrants.sum())})
                    continue
            refugees = events.maybe_war(v, self.cfg.war, self.rng)
            if refugees is not None:
                self.event_log.append({"gen": gen_index, "valley": v.name, "type": "war",
                                        "refugees": int(refugees.sum())})
                self.event_log.extend(migration.redistribute_refugees(self, v, refugees, self.rng, gen_index))

        # 5. Gradual cull: only remove a fraction of whatever excess remains, so severe
        # overcrowding can take several generations to fully resolve.
        surv_w = selection.survival_weights_flat(self.cfg.selection)
        for v in self.valleys:
            cap_now = int(round(self.current_capacity(v)))
            pop = v.population
            excess = pop - cap_now
            if excess > 0:
                to_remove = min(pop, int(round(excess * self.cfg.cull.cull_fraction_per_gen)))
                if to_remove > 0:
                    v.set_flat_counts(sampling.weighted_sample_without_replacement_counts(
                        v.flat_counts, surv_w, pop - to_remove, self.rng))

        # 6. Birth: only NOW, after this generation's disaster and its survivors' own
        # reactions (flee, fight, get culled) have all played out, do the survivors have
        # children -- not the other way around. Draws this generation's replacement
        # population from what's left of this generation's own residents; all newborns
        # start out "native" here, and "arrived" resets to 0, since the cohort that just
        # migrated/fought/got culled is entirely replaced by its own children. Family-size
        # decisions use a LAGGED fill fraction (fill_lag_gens generations stale) rather
        # than this generation's real one -- nobody in the valley is running live capacity
        # numbers, so crowding (or a disaster clearing everyone out) takes a while to
        # actually show up in birth rates. See _births_for.
        fert_w = selection.fertility_weights(self.cfg.selection)
        lag = self.cfg.growth.fill_lag_gens
        for v in self.valleys:
            n_BB, n_Bb, n_bb = v.counts
            population = int(n_BB + n_Bb + n_bb)
            cap_now = self.current_capacity(v)
            current_x = population / cap_now if cap_now > 0 else 0.0
            x_for_births = v.fill_history[0] if lag > 0 and len(v.fill_history) >= lag else current_x
            births = self._births_for(population, x_for_births, self.rng)
            v.set_native(genetics.multinomial_birth_counts(n_BB, n_Bb, n_bb, births, self.rng, fert_w))
            v.set_arrived((0, 0, 0))
            v.fill_history.append(current_x)
            if len(v.fill_history) > max(lag, 1):
                v.fill_history.pop(0)

        self.history.append(self._snapshot(gen_index))

        # 7. Advance timers for the next generation: devastation counts down first; the
        # ash-fertility bonus only starts its own countdown once devastation actually ends.
        for v in self.valleys:
            if v.devastated_gens_remaining > 0:
                v.devastated_gens_remaining -= 1
                if v.devastated_gens_remaining == 0:
                    v.fertility_bonus_remaining_gens = self.cfg.volcano.fertility_decay_gens
            elif v.fertility_bonus_remaining_gens > 0:
                v.fertility_bonus_remaining_gens -= 1

    def _snapshot(self, gen_index: int) -> dict:
        return {
            "gen": gen_index,
            "valleys": [
                {"name": v.name, "n_BB": int(v.counts[0]), "n_Bb": int(v.counts[1]), "n_bb": int(v.counts[2]),
                 "native": int(v.native_BB + v.native_Bb + v.native_bb),
                 "arrived": int(v.arrived_BB + v.arrived_Bb + v.arrived_bb),
                 "population": v.population, "q": v.q, "capacity": self.current_capacity(v),
                 "over_capacity_streak": v.over_capacity_streak,
                 "eruption": v.eruption_this_gen, "war": v.war_occurred_this_gen,
                 "emigration": v.emigration_this_gen, "devastated": v.devastated_gens_remaining > 0}
                for v in self.valleys
            ],
        }

    def run(self, generations: Optional[int] = None) -> List[dict]:
        generations = generations or self.cfg.generations
        self.history = [self._snapshot(0)]
        for g in range(1, generations + 1):
            self.step(g)
        return self.history
