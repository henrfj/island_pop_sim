"""Runnable entry point for the island simulation.

    python -m island.run_sim

Builds the default 4-valley island (boat lands in Landfall), runs it, prints a summary,
and writes a self-contained HTML replay you can open directly in a browser.
"""
from collections import Counter
import numpy as np

from .config import SimulationConfig, default_valleys
from .world import Island
from .viz.export_html import export_history_html


def main(generations: int = None, out_path: str = "island_replay.html") -> Island:
    cfg = SimulationConfig()
    if generations is not None:
        cfg.generations = generations
    rng = np.random.default_rng(cfg.seed)
    island = Island(default_valleys(), cfg, rng)
    island.run()

    print(f"Ran {cfg.generations} generations across {island.n} valleys.")
    for gen in (0, cfg.generations // 2, cfg.generations):
        snap = island.history[gen]
        print(f"\ngen {gen}:")
        for v in snap["valleys"]:
            print(f"  {v['name']:<10} pop={v['population']:<4} q={v['q']*100:5.1f}%  cap={v['capacity']:.0f}"
                  f"  (native={v['native']}, arrived={v['arrived']}, streak={v['over_capacity_streak']})")

    event_counts = Counter(e["type"] for e in island.event_log)
    print(f"\nEvents over {cfg.generations} generations: {dict(event_counts)}")

    island_pop = [sum(v["population"] for v in snap["valleys"]) for snap in island.history]
    print(f"Island total population: min={min(island_pop)}, max={max(island_pop)}, "
          f"final={island_pop[-1]} (started at {island_pop[0]})")

    print("\nEvent log:")
    for e in island.event_log:
        print(" ", e)

    valley_order = [v.name for v in island.valleys]
    out = export_history_html(island.history, island.event_log, valley_order, out_path)
    print(f"\nWrote replay: {out}")
    return island


if __name__ == "__main__":
    main()
