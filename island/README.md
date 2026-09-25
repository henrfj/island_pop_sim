# Island population-genetics simulation

This is a stochastic, aggregate simulation of four valleys over five-year turns. It is
a hybrid model: richer than complete generational replacement, but not an
individual-person historical reconstruction.

Run it from the repository root:

```powershell
uv run python -m island.run_sim
```

The command prints a summary and writes `island_replay.html`, which opens directly in a
browser without a server.

## State

Population counts are partitioned by valley, current tribe membership, newcomer or
established residency, child/adult/elder life stage, and `BB`/`Bb`/`bb` genotype.

Eye color is deliberately represented as one dominant/recessive locus. Real eye color is
polygenic; here it remains a visible marker for drift, migration, and later social
selection experiments.

## Five-year lifecycle

1. A scheduled eruption may kill residents, destroy food, and damage productive land.
2. Adults produce food; stored food spoils; everyone consumes according to life stage.
3. Baseline mortality occurs, with additional mortality only under severe shortage.
4. Surviving cohorts age statistically.
5. Population-scaled household movement provides steady gene flow, with extra
	shortage-driven waves. Canoe trips risk storms and clan branches provide only a soft
	destination pull.
6. Food stress and land damage may trigger conflict. Temporary factions cut across clan
	labels, so one clan can split and several clans can form a coalition.
7. Adults form families with a bounded same-clan preference. Cross-clan children inherit
	genotype from those same parental pools and receive one parental membership.
8. Newcomers become established and damaged land recovers gradually. Routine automatic
	assimilation is disabled; membership changes through families and rare secession.

Culture traits belong to clan records, not clan name roots. Cross-clan families can found
a persistent mixed clan with a weighted blend of both parent cultures and one bounded
mutation; secession creates a mutated branch culture. Trait values remain fixed after a
clan is created. The replay reports clan populations and, separately, populations carrying
positive, negative, or neutral values for each culture trait.

Gene traits are genotype-linked selective effects, not permanent guarantees. In a finite
population the recessive blue-eye allele can rise during Landfall's ash recovery and later
be lost through drift or fixation, especially over thousands of five-year turns. A stable
long-run polymorphism would require larger effective populations, ongoing gene mutation,
or frequency-dependent selection.

There is no hard carrying-capacity cull. Land productivity limits food production; food
stores buffer shocks; declining food security lowers fertility before severe shortage
raises mortality. The defaults are exploratory assumptions, not fitted historical
estimates.

The main scenario restores Landfall's intended advantage: blue-eyed founders enter a
sparsely populated, recovering, high-potential ash valley. Eruptions rotate every 20
years, returning to a valley after 80 years. `neutral_control_scenario()` disables
eruptions and equalizes environments; `slow_volcanic_scenario()` uses a 160-year return.

## Modules

```text
config.py          validated parameters in five-year units
demography.py      mortality, aging, and food-sensitive fertility
resources.py       production, storage, spoilage, consumption, food security
genetics.py        random-mating Mendelian birth sampling
sampling.py        weighted sampling without replacement
tribes.py          mixed-parent membership and branch naming
conflict.py        temporary cross-clan faction construction
world.py           aggregate state, lifecycle, and accounting ledger
experiments.py     repeat runs across independent seeds
run_sim.py         single-run command-line entry point
viz/export_html.py self-contained diagnostic replay
```

## Validation

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m island.experiments
```

Interpret ensembles rather than treating one replay as evidence. The calm baseline is
tuned to remain broadly stable and well-fed; scheduled eruptions are intentionally more
damaging and still require sensitivity analysis.