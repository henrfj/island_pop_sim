# Island population-genetics simulation

A multi-valley extension of the single-population eye-color simulation in
[`../eye_color_sim.py`](../eye_color_sim.py) (see the write-up in that session for the
Hardy-Weinberg derivation this is all built on). Four valleys sit in a ring around a
central volcano; blue-eyed founders land in one of them (`Landfall`), and the rest of the
island starts pure brown-eyed. Run it with:

```
python -m island.run_sim
```

This prints a summary and writes `island_replay.html` -- a self-contained, double-clickable
replay you can open in any browser (no server needed).

## Files, and what each one owns

```
config.py     - every tunable number, in one place (dataclasses, all documented inline)
genetics.py   - the core Mendelian math: random-mating birth sampling (from Part 1)
sampling.py   - generic weighted sampling-without-replacement (used by culling, migration,
                and emigration -- knows nothing about genotypes or valleys)
selection.py  - turns SelectionConfig's human-readable knobs ("bb", "brown", ...) into the
                per-category weight vectors genetics.py and world.py actually use
events.py     - the four disruptive events (volcano, war, emigration, storm), each a plain
                function operating on a Valley's generic 6-category count vector
migration.py  - movement between valleys: who leaves, where they go, and canoe storm risk
world.py      - the Valley/Island classes and Island.step(), which wires everything above
                into one generation's pipeline (see diagram below)
run_sim.py    - CLI entry point: builds the default island, runs it, prints a summary,
                exports the replay
viz/export_html.py - renders a finished run's history into the self-contained HTML replay
```

Rough call graph for a single run:

```
run_sim.py
  -> config.default_valleys()          (island/config.py)
  -> world.Island(valleys, cfg, rng)   (island/world.py)
  -> island.run()
       for each generation: island.step()  <-- see pipeline diagram below
           -> events.scheduled_eruption_index() / events.erupt()  (island/events.py)
           -> migration.migrate()           (island/migration.py)
                -> sampling.weighted_sample_without_replacement_counts()  (island/sampling.py)
                -> events.storm_strikes()   (island/events.py)
           -> events.maybe_emigrate() [if enabled] / events.maybe_war()   (island/events.py)
                -> migration.redistribute_refugees()  (island/migration.py, for war only)
           -> sampling.weighted_sample_without_replacement_counts()  (gradual cull)
           -> world._mean_kids_per_couple() / genetics.multinomial_birth_counts()
  -> viz.export_html.export_history_html()  (island/viz/export_html.py)
       -> writes island_replay.html
```

## One generation, step by step

Population in each valley is tracked as genotype counts (`BB`/`Bb`/`bb`) split by
**origin**: `native` (born in this valley, this generation) vs. `arrived` (born
elsewhere, showed up here this generation via migration or as a war refugee). Because
generations don't overlap -- every generation's entire population is freshly drawn from
the previous generation's parents -- that split resets automatically each generation; it
needs no memory beyond "did you just move here."

Birth is deliberately the SECOND-TO-LAST step, not the first thing that happens after a
disaster. A valley's actual living residents experience the eruption and get to react to
it themselves -- flee, fight, get culled -- before anyone has children; the newborns are
the *outcome* of how a generation's disasters played out, not a cohort that absorbs the
disaster on their parents' behalf and only then starts reacting.

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  GENERATION N, for every valley in parallel                             │
  └─────────────────────────────────────────────────────────────────────────┘

  1. VOLCANO CHECK  (events.scheduled_eruption_index, events.erupt)
     a fixed clockwise rotation, not random chance: every eruption_interval_gens
     generations (default 20), the next valley clockwise erupts -- so any one
     valley is hit once every n_valleys x eruption_interval_gens gens (80, by
     default). Landfall counts as having just erupted before gen 0.
        erupts ──▶ kill 10-20% of current residents (native & arrived alike)
                   capacity floored to ~0 for devastation_gens generations (2) --
                   functionally uninhabitable: some residents flee immediately as
                   a normal migration group (step 2), the rest get gradually
                   culled (step 5) or fight over the ashes (step 4)
                   once devastation ends: 20-gen ash bonus to capacity, decaying to 0
        │
        ▼
  2. MIGRATION  (migration.migrate)
     any valley with a surplus ABOVE capacity draws ONE randomly-sized group from
     that surplus this generation -- a "family" at the small end, a "gathering"
     at the large end (migration_min/max_fraction) -- not the whole surplus as a
     continuous trickle. That group rolls a lottery: stay put (preferred) or move
     to another valley, weighted by that valley's free room and how expensive the
     route is (cheap ring-neighbor path, costly canoe elsewhere). Whatever surplus
     isn't picked this generation simply stays behind for next generation's roll,
     the cull, or a war to sort out. Every successful move AND every storm loss is
     logged.
        canoe leg taken ──▶ STORM CHECK (events.storm_strikes): % chance the
                             whole party is lost at sea, arrives nowhere
     movers who land elsewhere become "arrived" there; movers who stay keep
     their native/arrived status
        │
        ▼
  3. OVER-CAPACITY STREAK UPDATE
     still over capacity after migration? streak += 1 : streak = 0
     (this streak is what drives step 4's probabilities -- duration matters,
     not just how far over capacity you are)
        │
        ▼
  4. EMIGRATION-OR-WAR  (events.maybe_emigrate [only if EmigrationConfig.enabled,
     off by default], events.maybe_war -- mutually exclusive per valley per
     generation; emigration is checked first when enabled)
     both probabilities are ~0% for a fresh overshoot, ramping (quadratically)
     the longer the streak persists:
        EMIGRATION (gentler, triggers earlier, caps below 100% -- currently
           disabled to cut down on event friction):
           a random fraction of the population leaves the island for good --
           gone from the simulation entirely, no destination
        WAR (harsher, only near-certain if the crisis drags on for a long time):
           population splits into random factions; the winner (bigger factions
           favored, not guaranteed) stays; every faction takes casualties;
           the losing faction's survivors become refugees
              ──▶ migration.redistribute_refugees() sends them toward other
                  valleys (same room/cost weighting + storm risk as step 2)
        │
        ▼
  5. GRADUAL CULL  (sampling.weighted_sample_without_replacement_counts)
     only a FRACTION of any remaining excess-over-capacity is removed this
     generation -- severe overcrowding can take several generations to fully
     resolve, which is exactly the window steps 3-4 need to act in.
     survival weight per individual = genetic-advantage weight (selection.py)
                                     x resident-advantage weight (native > arrived,
                                       if SelectionConfig.resident_survival_bonus > 1)
        │
        ▼
  6. BIRTH  (world._births_for / _mean_kids_per_couple, genetics.multinomial_birth_counts)
     THIS generation's survivors -- after living through steps 1-5 -- become
     parents, entirely; they are replaced, not aged forward, only their children
     exist next gen. Parents are paired into population//2 couples; total births
     = Poisson(couples x mean_kids(x)), a genuine random draw, not a rounded
     formula. mean_kids(x) falls smoothly as fill x = population/capacity rises:
     near-empty valleys are generous (kids_peak), a valley sitting AT capacity is
     already trending down (kids_at_capacity < 2), severe overcrowding asymptotes
     to a bare kids_floor. Because couples scale with population while mean_kids
     scales with fill, the S-curve in *total* births falls out naturally.
        x is itself LAGGED: mean_kids uses the fill fraction from
        fill_lag_gens generations ago (default 4), not this generation's --
        nobody in these valleys is running live capacity numbers, so a couple's
        sense of "is it crowded?" is stale. This is the classic delayed-logistic
        growth trick from ecology: population overshoots capacity and corrects
        afterward instead of snapping to equilibrium every generation.
     -> this generation's newborns are all "native"; "arrived" resets to 0
        │
        ▼
  7. ADVANCE TIMERS  (devastation countdown -1; once it hits 0, the 20-gen ash
     bonus countdown starts)
        │
        ▼
  snapshot recorded -> next generation
```

## Tuning knobs worth knowing about (all in `config.py`)

- `GrowthConfig.kids_peak` / `kids_at_capacity` / `kids_floor` -- the mean-kids-per-couple
  curve: fertility when nearly empty, right at capacity (kept < 2 so capacity itself isn't
  a stable resting point), and under severe overcrowding.
- `VolcanoConfig.eruption_interval_gens` -- gap between eruptions in the clockwise rotation
  (a given valley is hit every `n_valleys x` this many generations).
  `death_fraction_min/max`, `devastation_gens` -- how damaging and how long an eruption's
  aftermath is.
- `MigrationConfig.migration_min/max_fraction` -- how big a slice of a valley's surplus
  moves as one group per generation ("family" vs. "gathering").
- `CullConfig.cull_fraction_per_gen` -- how quickly overcrowding gets resolved by culling
  alone, before war even matters.
- `WarConfig.patience_gens` / `EmigrationConfig.patience_gens` -- how many consecutive
  over-capacity generations it takes before each event becomes likely. War's ramp is
  quadratic (genuinely rare for a brief overshoot); emigration's is the same shape but
  caps at `max_chance` instead of certainty. `EmigrationConfig.enabled` is `False` by
  default -- flip it on to bring off-island emigration back as a release valve.
- `StormConfig.storm_chance` -- per-crossing risk of losing a canoe party entirely.
- `SelectionConfig` -- all three knobs (aesthetic preference, genetic advantage, resident
  survival bonus) default to neutral (no effect); set any of them away from 1.0 to run a
  selection experiment instead of the pure-drift baseline.

## Known simplifications / ideas for later

- War factions are currently a random 50/50 split, not native-vs-arrived -- the data
  model (the native/arrived split itself) was built specifically so that's a small change
  later, not a redesign.
- Emigrants and storm-lost migrants are simply removed from the simulation; there's no
  "off-island" population being tracked anywhere.
- All SelectionConfig weights are constants; making them react to a valley's own state
  (e.g., a genetic advantage that only matters in the generations right after an eruption)
  is a natural next step and doesn't require restructuring anything.
