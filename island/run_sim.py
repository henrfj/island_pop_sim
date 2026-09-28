"""Run the five-year island model and write a self-contained HTML replay."""
import argparse
from collections import Counter
from dataclasses import asdict
from typing import Callable, Optional
import numpy as np

from .config import (ArrivalEventConfig, SimulationConfig, ValleySpec,
                     main_volcanic_scenario, neutral_control_scenario,
                     slow_volcanic_scenario, tideborn_defensive_scenario)
from .viz.export_html import export_history_html
from .world import Island


SCENARIOS: dict[str, Callable[[], tuple[SimulationConfig, list[ValleySpec]]]] = {
    "main": main_volcanic_scenario,
    "slow-volcanic": slow_volcanic_scenario,
    "neutral-control": neutral_control_scenario,
    "tideborn-defensive": tideborn_defensive_scenario,
}


def build_run_config(turns: Optional[int] = None, seed: Optional[int] = None,
                     scenario_name: str = "main",
                     blackwake: str = "auto") -> tuple[SimulationConfig, list[ValleySpec]]:
    scenario = SCENARIOS[scenario_name]
    cfg, valley_specs = scenario()
    if turns is not None:
        cfg.turns = turns
    if seed is not None:
        cfg.seed = int(seed)
    if blackwake == "off":
        cfg.arrivals = ()
    elif blackwake == "on" and not cfg.arrivals:
        cfg.arrivals = (ArrivalEventConfig(),)
    return cfg, valley_specs


def main(turns: Optional[int] = None, out_path: str = "island_replay.html",
         seed: Optional[int] = None, scenario_name: str = "main",
         blackwake: str = "auto") -> Island:
    cfg, valley_specs = build_run_config(
        turns=turns,
        seed=seed,
        scenario_name=scenario_name,
        blackwake=blackwake,
    )
    island = Island(valley_specs, cfg, np.random.default_rng(cfg.seed))
    island.run()

    print(f"Ran {cfg.turns} five-year turns ({cfg.turns * cfg.years_per_turn} years).")
    final_totals = island.history[-1]["totals"]
    print("\nGene traits (BB / Bb / bb):")
    for name, values in asdict(cfg.traits.genes).items():
        print(f"  {name}: " + " / ".join(f"{value:+.0%}" for value in values))
    print("Clan traits:")
    for clan, traits in sorted(cfg.traits.clans.items()):
        active = [f"{name}={value:+.0%}" for name, value in asdict(traits).items() if value]
        print(f"  {clan}: {', '.join(active) if active else 'neutral'}")
    for turn in sorted({0, cfg.turns // 2, cfg.turns}):
        snapshot = island.history[turn]
        print(f"\nturn {turn} / year {snapshot['year']}:")
        for valley in snapshot["valleys"]:
            q_text = "n/a" if valley["q"] is None else f"{valley['q'] * 100:.1f}%"
            clan_text = ", ".join(
                f"{clan}={population}"
                for clan, population in valley["tribes"].items())
            trait_text = ", ".join(
                f"{trait}(+{population['positive']}/-{population['negative']}/0={population['neutral']})"
                for trait, population in valley["trait_populations"].items())
            print(
                f"  {valley['name']:<10} pop={valley['population']:<4} "
                f"food={valley['food_stock']:6.1f} security={valley['food_security']:.2f} "
                f"q={q_text} clans=[{clan_text}] traits=[{trait_text}]"
            )

    counts = Counter(event["type"] for event in island.event_log)
    populations = [sum(v["population"] for v in snap["valleys"]) for snap in island.history]
    print(f"\nEvents: {dict(counts)}")
    print(f"Island population: start={populations[0]}, min={min(populations)}, final={populations[-1]}")
    print("Cumulative deaths by source: " + ", ".join(
        f"{source}={deaths}" for source, deaths in final_totals["deaths_by_source"].items()))
    print("Cumulative deaths by genotype: " + ", ".join(
        f"{genotype}={deaths}" for genotype, deaths in final_totals["deaths_by_genotype"].items()))
    out = export_history_html(
        island.history, island.event_log, [v.name for v in island.valleys],
        island.provenance(), out_path,
    )
    print(f"Wrote replay with debug charts: {out}")
    return island


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the island replay simulation.")
    parser.add_argument("--turns", type=int, default=None,
                        help="Number of five-year turns to simulate.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the scenario seed. Same code + config + seed gives the same replay.")
    parser.add_argument("--out", default="island_replay.html",
                        help="Output HTML path.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="main",
                        help="Named scenario to run.")
    parser.add_argument("--blackwake", choices=("auto", "on", "off"), default="auto",
                        help="Control the late Blackwake arrival: keep scenario default, force on, or force off.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        turns=args.turns,
        out_path=args.out,
        seed=args.seed,
        scenario_name=args.scenario,
        blackwake=args.blackwake,
    )