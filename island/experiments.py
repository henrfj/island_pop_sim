"""Small ensemble runner for checking whether behavior survives seed changes."""
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, List
import numpy as np

from .config import (SimulationConfig, default_valleys, main_volcanic_scenario,
                     neutral_control_scenario)
from .world import Island


@dataclass(frozen=True)
class RunSummary:
    seed: int
    final_population: int
    minimum_population: int
    minimum_food_security: float
    events: dict
    initial_q: float
    final_q: float
    final_valley_q: tuple
    allele_occupancy: int
    heterozygosity: float
    fst: float


def _genetic_metrics(snapshot: dict) -> tuple[float, tuple, int, float, float]:
    total_population = sum(valley["population"] for valley in snapshot["valleys"])
    b_alleles = sum(valley["n_Bb"] + 2 * valley["n_bb"] for valley in snapshot["valleys"])
    global_q = 0.0 if total_population == 0 else b_alleles / (2 * total_population)
    valley_q = tuple(valley["q"] for valley in snapshot["valleys"])
    occupancy = sum(
        valley["population"] > 0 and valley["n_Bb"] + valley["n_bb"] > 0
        for valley in snapshot["valleys"]
    )
    heterozygosity = 2 * global_q * (1 - global_q)
    weighted_within = 0.0
    if total_population > 0:
        weighted_within = sum(
            valley["population"] / total_population * 2 * valley["q"] * (1 - valley["q"])
            for valley in snapshot["valleys"] if valley["q"] is not None
        )
    fst = 0.0 if heterozygosity <= 0 else max(0.0, 1.0 - weighted_within / heterozygosity)
    return global_q, valley_q, occupancy, heterozygosity, fst


def run_ensemble(seeds: Iterable[int], turns: int = 80,
                 eruptions: bool = True,
                 scenario: Callable = None) -> List[RunSummary]:
    summaries = []
    for seed in seeds:
        if scenario is None:
            cfg = SimulationConfig(turns=turns, seed=int(seed))
            specs = default_valleys()
            if not eruptions:
                cfg.volcano.eruption_interval_turns = 0
        else:
            cfg, specs = scenario()
            cfg.turns = turns
            cfg.seed = int(seed)
        island = Island(specs, cfg, np.random.default_rng(cfg.seed))
        island.run()
        populations = [sum(v["population"] for v in snap["valleys"])
                       for snap in island.history]
        minimum_security = min(v["food_security"] for snap in island.history
                               for v in snap["valleys"])
        initial_q, _, _, _, _ = _genetic_metrics(island.history[0])
        final_q, valley_q, occupancy, heterozygosity, fst = _genetic_metrics(island.history[-1])
        summaries.append(RunSummary(
            seed=cfg.seed,
            final_population=populations[-1],
            minimum_population=min(populations),
            minimum_food_security=minimum_security,
            events=dict(Counter(event["type"] for event in island.event_log)),
            initial_q=initial_q,
            final_q=final_q,
            final_valley_q=valley_q,
            allele_occupancy=occupancy,
            heterozygosity=heterozygosity,
            fst=fst,
        ))
    return summaries


def main() -> None:
    seeds = np.random.SeedSequence(20260923).generate_state(100)
    seed_values = [int(seed) for seed in seeds]
    runs = run_ensemble(seed_values, scenario=main_volcanic_scenario)
    controls = run_ensemble(seed_values, scenario=neutral_control_scenario)
    final = np.asarray([run.final_population for run in runs])
    minimum_security = np.asarray([run.minimum_food_security for run in runs])
    print(f"runs={len(runs)}")
    print(f"final population median={np.median(final):.0f} "
          f"10-90%={np.quantile(final, 0.1):.0f}-{np.quantile(final, 0.9):.0f}")
    print(f"minimum food security median={np.median(minimum_security):.2f}")
    print(f"volcanic final q median={np.median([run.final_q for run in runs]):.3f} "
          f"occupancy median={np.median([run.allele_occupancy for run in runs]):.0f}/4 "
          f"FST median={np.median([run.fst for run in runs]):.3f}")
    print(f"neutral final q mean={np.mean([run.final_q for run in controls]):.3f} "
          f"(initial={controls[0].initial_q:.3f})")


if __name__ == "__main__":
    main()