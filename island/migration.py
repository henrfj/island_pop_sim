"""Movement between valleys.

Each generation, a valley with a population surplus (over capacity) doesn't send the
*whole* surplus wandering as a continuous trickle -- instead, like the emigration and war
events, a single randomly-sized group (a "family" at the small end, a "gathering" at the
large end -- see MigrationConfig.migration_min/max_fraction) is drawn from the surplus as
this generation's candidate movers. That group then rolls one weighted lottery between
staying put and moving to one other valley, weighted by how much free room that valley
has and how expensive it is to reach (cheap mountain path to a ring-neighbor, expensive
canoe route to anyone else). Whatever's left of the surplus (not selected as this
generation's group) simply stays behind, for the gradual cull, next generation's migration
roll, or an escalating war to sort out. War refugees use the same room/cost weighting but
have no "stay" option, and move as a whole -- they've been expelled from their home valley
and must go somewhere else.

Canoe crossings (water routes) are risky: each (origin, destination) party that tries a
water route this generation rolls a single storm check -- if it hits, that whole party is
lost at sea (never arrives, and doesn't return home either).

A mover's native/arrived status is preserved if they stay home, but always becomes
"arrived" if they land in a different valley -- see world.py's Valley.flat_counts layout
([native_BB, native_Bb, native_bb, arrived_BB, arrived_Bb, arrived_bb]).

All decisions for a generation are made from a single snapshot of population/capacity
taken before any moves are applied, so multiple valleys can (realistically) converge on
the same attractive destination and temporarily overshoot its capacity -- that gets
sorted out by the (now gradual) end-of-generation cull, not by migration itself.
"""
import numpy as np

from . import sampling
from .events import storm_strikes

N_GENOTYPES = 3
N_CATEGORIES = 6  # 2 origins (native, arrived) x 3 genotypes
ARRIVED_OFFSET = 3


def _capacity_snapshot(island):
    return ([v.population for v in island.valleys],
            [island.current_capacity(v) for v in island.valleys])


def _room_and_cost_weights(island, origin: int, destinations, pops, caps) -> np.ndarray:
    weights = []
    for j in destinations:
        free_fraction = max(0.0, (caps[j] - pops[j]) / caps[j]) if caps[j] > 0 else 0.0
        weights.append(free_fraction / island.travel_cost(origin, j))
    return np.array(weights, dtype=float)


def migrate(island, rng: np.random.Generator, gen_index: int) -> list:
    """Returns a list of event dicts (type "migration" for a successful move, "storm" for a
    canoe party lost at sea), each already stamped with `gen_index` -- empty if no valley
    had a surplus this generation."""
    cfg = island.cfg.migration
    storm_chance = island.cfg.storm.storm_chance
    pops, caps = _capacity_snapshot(island)

    group_by_valley = []
    for i, v in enumerate(island.valleys):
        surplus_total = max(0, pops[i] - int(round(caps[i])))
        if surplus_total > 0:
            fraction = rng.uniform(cfg.migration_min_fraction, cfg.migration_max_fraction)
            group_size = min(surplus_total, int(round(surplus_total * fraction)))
        else:
            group_size = 0
        if group_size > 0:
            group_by_valley.append(sampling.weighted_sample_without_replacement_counts(
                v.flat_counts, np.ones(N_CATEGORIES), group_size, rng))
        else:
            group_by_valley.append(np.zeros(N_CATEGORIES, dtype=np.int64))

    deltas = [np.zeros(N_CATEGORIES, dtype=np.int64) for _ in range(island.n)]
    events_out = []

    for i, group in enumerate(group_by_valley):
        if group.sum() == 0:
            continue
        others = [j for j in range(island.n) if j != i]
        move_weights = cfg.mobility * _room_and_cost_weights(island, i, others, pops, caps)
        weights = np.concatenate(([cfg.stay_weight], move_weights))
        if weights.sum() <= 0:
            weights = np.array([1.0] + [0.0] * len(others))
        probs = weights / weights.sum()

        outcome = int(rng.choice(len(probs), p=probs))
        if outcome == 0:
            continue

        dest = others[outcome - 1]
        genotype_totals = np.array([
            group[0] + group[ARRIVED_OFFSET],
            group[1] + group[ARRIVED_OFFSET + 1],
            group[2] + group[ARRIVED_OFFSET + 2],
        ], dtype=np.int64)
        total = int(genotype_totals.sum())
        if total == 0:
            continue

        deltas[i] -= group
        is_water_route = dest not in island.path_neighbors[i]
        route = "canoe" if is_water_route else "path"
        if is_water_route and storm_strikes(rng, storm_chance):
            events_out.append({"gen": gen_index, "type": "storm",
                                "from": island.valleys[i].name, "to": island.valleys[dest].name,
                                "lost": total})
            continue  # the whole party is lost at sea -- never added anywhere
        for genotype in range(N_GENOTYPES):
            deltas[dest][ARRIVED_OFFSET + genotype] += genotype_totals[genotype]
        events_out.append({"gen": gen_index, "type": "migration",
                            "from": island.valleys[i].name, "to": island.valleys[dest].name,
                            "count": total, "route": route})

    for v, delta in zip(island.valleys, deltas):
        v.set_flat_counts(v.flat_counts + delta)

    return events_out


def redistribute_refugees(island, home_valley, refugee_counts, rng: np.random.Generator, gen_index: int) -> list:
    """Force-migrate war refugees out of `home_valley` into whichever other valleys look
    most attractive (room/cost weighted, same as ordinary migration but with no stay
    option). Canoe legs carry the same storm risk as ordinary migration. If every other
    valley is completely full, refugees scatter evenly rather than vanish -- the resulting
    overcrowding is resolved by that valley's own gradual cull, same as any other
    over-capacity situation. Returns a list of storm-event dicts."""
    origin = island.valleys.index(home_valley)
    pops, caps = _capacity_snapshot(island)
    others = [j for j in range(island.n) if j != origin]
    weights = _room_and_cost_weights(island, origin, others, pops, caps)
    if weights.sum() <= 0:
        weights = np.ones(len(others))
    probs = weights / weights.sum()
    storm_chance = island.cfg.storm.storm_chance

    dest = others[int(rng.choice(len(others), p=probs))]
    genotype_totals = np.array([
        refugee_counts[0] + refugee_counts[ARRIVED_OFFSET],
        refugee_counts[1] + refugee_counts[ARRIVED_OFFSET + 1],
        refugee_counts[2] + refugee_counts[ARRIVED_OFFSET + 2],
    ], dtype=np.int64)
    total = int(genotype_totals.sum())
    if total == 0:
        return []

    is_water_route = dest not in island.path_neighbors[origin]
    route = "canoe" if is_water_route else "path"
    if is_water_route and storm_strikes(rng, storm_chance):
        return [{"gen": gen_index, "type": "storm", "kind": "refugee",
                 "from": home_valley.name, "to": island.valleys[dest].name,
                 "lost": total}]

    counts = island.valleys[dest].flat_counts
    for genotype in range(N_GENOTYPES):
        counts[ARRIVED_OFFSET + genotype] += genotype_totals[genotype]
    island.valleys[dest].set_flat_counts(counts)
    return [{"gen": gen_index, "type": "migration", "kind": "refugee",
             "from": home_valley.name, "to": island.valleys[dest].name,
             "count": total, "route": route}]
