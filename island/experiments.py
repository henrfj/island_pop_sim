"""Small ensemble runner for checking whether behavior survives seed changes."""
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, List
import numpy as np

from .config import (SimulationConfig, default_valleys, main_volcanic_scenario,
                     neutral_control_scenario)
from .viz.export_ensemble_html import export_ensemble_html
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
    population_series: tuple
    q_series: tuple
    min_food_security_series: tuple
    deaths_by_source_series: dict
    baseline_deaths_by_stage_series: dict
    deaths_by_genotype_series: dict


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


def _band(series_collection: List[tuple | list]) -> dict:
    matrix = np.asarray(series_collection, dtype=float)
    return {
        "p10": np.quantile(matrix, 0.1, axis=0).tolist(),
        "p50": np.median(matrix, axis=0).tolist(),
        "p90": np.quantile(matrix, 0.9, axis=0).tolist(),
    }


def _incremental(series: tuple | list) -> tuple:
    values = list(series)
    return tuple(
        values[index] - (values[index - 1] if index > 0 else 0)
        for index in range(len(values))
    )


def _per_1000(events: tuple | list, populations: tuple | list) -> tuple:
    return tuple(
        0.0 if index == 0 or populations[index - 1] <= 0 else values * 1000.0 / populations[index - 1]
        for index, values in enumerate(events)
    )


def _aggregate_runs(runs: List[RunSummary], label: str, color: str) -> dict:
    final_population = np.asarray([run.final_population for run in runs], dtype=float)
    min_population = np.asarray([run.minimum_population for run in runs], dtype=float)
    min_food_security = np.asarray([run.minimum_food_security for run in runs], dtype=float)
    final_q = np.asarray([run.final_q for run in runs], dtype=float)
    fst = np.asarray([run.fst for run in runs], dtype=float)
    source_names = list(runs[0].deaths_by_source_series)
    genotype_names = list(runs[0].deaths_by_genotype_series)
    stage_names = list(runs[0].baseline_deaths_by_stage_series)
    return {
        "label": label,
        "color": color,
        "count": len(runs),
        "summary": {
            "final_population_median": float(np.median(final_population)),
            "final_population_p10": float(np.quantile(final_population, 0.1)),
            "final_population_p90": float(np.quantile(final_population, 0.9)),
            "minimum_population_median": float(np.median(min_population)),
            "minimum_food_security_median": float(np.median(min_food_security)),
            "final_q_median": float(np.median(final_q)),
            "fst_median": float(np.median(fst)),
            "final_deaths_by_source": {
                name: float(np.median([run.deaths_by_source_series[name][-1] for run in runs]))
                for name in source_names
            },
            "final_deaths_by_genotype": {
                name: float(np.median([run.deaths_by_genotype_series[name][-1] for run in runs]))
                for name in genotype_names
            },
        },
        "series": {
            "population": _band([run.population_series for run in runs]),
            "q": _band([run.q_series for run in runs]),
            "min_food_security": _band([run.min_food_security_series for run in runs]),
        },
        "deaths_by_source": {
            name: _band([run.deaths_by_source_series[name] for run in runs])
            for name in source_names
        },
        "deaths_by_source_rate": {
            name: _band([
                _per_1000(_incremental(run.deaths_by_source_series[name]), run.population_series)
                for run in runs
            ])
            for name in source_names
        },
        "baseline_deaths_by_stage": {
            name: _band([
                _incremental(run.baseline_deaths_by_stage_series[name])
                for run in runs
            ])
            for name in stage_names
        },
        "deaths_by_genotype": {
            name: _band([run.deaths_by_genotype_series[name] for run in runs])
            for name in genotype_names
        },
    }


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
        q_series = [_genetic_metrics(snapshot)[0] for snapshot in island.history]
        min_food_security_series = [
            min(valley["food_security"] for valley in snapshot["valleys"])
            for snapshot in island.history
        ]
        minimum_security = min(v["food_security"] for snap in island.history
                               for v in snap["valleys"])
        initial_q, _, _, _, _ = _genetic_metrics(island.history[0])
        final_q, valley_q, occupancy, heterozygosity, fst = _genetic_metrics(island.history[-1])
        deaths_by_source_series = {
            source: tuple(snapshot["totals"]["deaths_by_source"].get(source, 0)
                          for snapshot in island.history)
            for source in island.history[-1]["totals"]["deaths_by_source"]
        }
        deaths_by_genotype_series = {
            genotype: tuple(snapshot["totals"]["deaths_by_genotype"].get(genotype, 0)
                            for snapshot in island.history)
            for genotype in island.history[-1]["totals"]["deaths_by_genotype"]
        }
        baseline_deaths_by_stage_series = {
            stage: tuple(snapshot["totals"]["baseline_deaths_by_stage"].get(stage, 0)
                         for snapshot in island.history)
            for stage in island.history[-1]["totals"]["baseline_deaths_by_stage"]
        }
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
            population_series=tuple(populations),
            q_series=tuple(q_series),
            min_food_security_series=tuple(min_food_security_series),
            deaths_by_source_series=deaths_by_source_series,
            baseline_deaths_by_stage_series=baseline_deaths_by_stage_series,
            deaths_by_genotype_series=deaths_by_genotype_series,
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
    volcanic_deaths = {
        source: np.median([run.deaths_by_source_series[source][-1] for run in runs])
        for source in runs[0].deaths_by_source_series
    }
    print(f"volcanic median cumulative deaths={{{', '.join(f'{k}: {v:.0f}' for k, v in volcanic_deaths.items())}}}")

    years = [index * 5 for index in range(len(runs[0].population_series))]
    report_data = {
        "years": years,
        "scenarios": [
            _aggregate_runs(runs, "Volcanic", "#B33A3A"),
            _aggregate_runs(controls, "Neutral control", "#237A57"),
        ],
        "provenance": {
            "seed_base": 20260923,
            "turns": len(years) - 1,
            "runs": len(runs),
        },
        "raw_runs": {
            "volcanic": [asdict(run) for run in runs],
            "neutral": [asdict(run) for run in controls],
        },
    }
    out = export_ensemble_html(report_data)
    print(f"wrote ensemble report: {out}")


if __name__ == "__main__":
    main()