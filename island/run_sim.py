"""Run the five-year island model and write a self-contained HTML replay."""
from collections import Counter
from dataclasses import asdict
from typing import Optional
import numpy as np

from .config import main_volcanic_scenario
from .viz.export_html import export_history_html
from .world import Island


def main(turns: Optional[int] = None, out_path: str = "island_replay.html") -> Island:
    cfg, valley_specs = main_volcanic_scenario()
    if turns is not None:
        cfg.turns = turns
    island = Island(valley_specs, cfg, np.random.default_rng(cfg.seed))
    island.run()

    print(f"Ran {cfg.turns} five-year turns ({cfg.turns * cfg.years_per_turn} years).")
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
    out = export_history_html(
        island.history, island.event_log, [v.name for v in island.valleys],
        island.provenance(), out_path,
    )
    print(f"Wrote replay: {out}")
    return island


if __name__ == "__main__":
    main()