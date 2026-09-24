"""
Blue-eye / brown-eye island simulation.

Model
-----
One gene, two alleles:
    B = brown, dominant
    b = blue,  recessive
Genotypes: BB (pure brown), Bb (carrier, brown-eyed), bb (blue-eyed)

Population dynamics per generation (a "Wright-Fisher with culling" model):
  1. Random mating pool: every offspring is produced by picking two parents
     uniformly at random (with replacement) from the current population and
     giving the child one randomly-chosen allele from each parent. Because
     parents are drawn uniformly and each contributes one uniformly-random
     allele of its own, a contributed allele is 'b' with probability exactly
     equal to the population's current allele frequency q. So each child's
     two alleles are i.i.d. Bernoulli(q) draws, and genotype counts among a
     batch of births follow Multinomial(births, [p^2, 2pq, q^2]) -- exactly
     the Hardy-Weinberg proportions, generated as a random (drift-prone)
     process rather than assumed as a fixed formula.
  2. The island can only support K individuals, so if births > K we cull
     back down to K by picking survivors uniformly at random regardless of
     genotype (pure random death -- no selection on eye colour). This is a
     multivariate-hypergeometric draw from the birth cohort's genotype
     counts.

This is exact (not an approximation) under the stated assumptions: random
mating, no selection, no assortative mating, no migration, no mutation.

We track (n_BB, n_Bb, n_bb) each generation, which lets us also read off the
blue-allele frequency q_t = (2*n_bb + n_Bb) / (2N).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------------------------------------------------
# Palette (literal brown/blue mapping is clearer here than an abstract
# categorical palette, since the whole point is "which eye colour").
# ----------------------------------------------------------------------
COL_BB = "#7A4B2A"    # pure brown (BB)
COL_Bb = "#D9A05B"    # carrier, brown-eyed (Bb)
COL_bb = "#2E75B6"    # blue-eyed (bb)
COL_Q = "#2E75B6"     # blue allele frequency
COL_MEAN = "#1B3A5C"  # dark navy for mean/summary lines
COL_GRID = "#DDDDDD"
COL_MUTED = "#8A8A8A"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": COL_GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.frameon": False,
})


def strip_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ----------------------------------------------------------------------
# Core simulation
# ----------------------------------------------------------------------
def simulate(n_BB0, n_Bb0, n_bb0, K, generations, growth_factor, rng):
    """Run one stochastic realization. Returns dict of arrays length generations+1."""
    n_BB, n_Bb, n_bb = n_BB0, n_Bb0, n_bb0
    hist_BB = np.empty(generations + 1, dtype=np.int64)
    hist_Bb = np.empty(generations + 1, dtype=np.int64)
    hist_bb = np.empty(generations + 1, dtype=np.int64)

    for t in range(generations + 1):
        hist_BB[t], hist_Bb[t], hist_bb[t] = n_BB, n_Bb, n_bb
        if t == generations:
            break
        N = n_BB + n_Bb + n_bb
        q = (2 * n_bb + n_Bb) / (2 * N)
        p = 1.0 - q
        births = int(round(K * growth_factor))
        probs = (p * p, 2 * p * q, q * q)
        counts = rng.multinomial(births, probs)  # [BB, Bb, bb] among newborns
        if births > K:
            survivors = rng.multivariate_hypergeometric(counts, K)
        else:
            survivors = counts
        n_BB, n_Bb, n_bb = int(survivors[0]), int(survivors[1]), int(survivors[2])

    q_hist = (2 * hist_bb + hist_Bb) / (2 * (hist_BB + hist_Bb + hist_bb))
    return {"BB": hist_BB, "Bb": hist_Bb, "bb": hist_bb, "q": q_hist}


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
N_BROWN0 = 100
N_BLUE0 = 10
K_ISLAND = 110          # carrying capacity of the island
K_LARGE = 3000          # a much bigger, more "interconnected" population
GENERATIONS = 150
GROWTH_FACTOR = 3.0     # each generation, produce 3x K births, then cull to K
N_REPLICATES = 400
SEED = 20260923

rng_master = np.random.default_rng(SEED)

q0 = (2 * N_BLUE0) / (2 * (N_BROWN0 + N_BLUE0))
p0 = 1 - q0
hw_BB = p0 ** 2
hw_Bb = 2 * p0 * q0
hw_bb = q0 ** 2

print(f"Initial: {N_BROWN0} BB + {N_BLUE0} bb = {N_BROWN0+N_BLUE0} people")
print(f"Initial blue allele frequency q0 = {q0:.4f} ({q0*100:.2f}%)")
print(f"Hardy-Weinberg equilibrium genotype proportions (reached after 1 gen of random mating):")
print(f"  BB (pure brown): {hw_BB*100:.2f}%")
print(f"  Bb (carrier, brown-eyed): {hw_Bb*100:.2f}%")
print(f"  bb (blue-eyed): {hw_bb*100:.2f}%")

# ----------------------------------------------------------------------
# 1. One representative, detailed run on the small island
# ----------------------------------------------------------------------
REP_SEED = 41  # chosen to show a typical history: wanders, then the blue allele
               # is eventually lost by chance -- the modal outcome at this K
rep_run = simulate(N_BROWN0, 0, N_BLUE0, K_ISLAND, GENERATIONS, GROWTH_FACTOR,
                    np.random.default_rng(REP_SEED))

gens = np.arange(GENERATIONS + 1)
N_ISLAND = rep_run["BB"] + rep_run["Bb"] + rep_run["bb"]
frac_BB = rep_run["BB"] / N_ISLAND
frac_Bb = rep_run["Bb"] / N_ISLAND
frac_bb = rep_run["bb"] / N_ISLAND

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.stackplot(gens, frac_BB * 100, frac_Bb * 100, frac_bb * 100,
             colors=[COL_BB, COL_Bb, COL_bb],
             labels=["BB — pure brown", "Bb — carrier (brown-eyed)", "bb — blue-eyed"],
             edgecolor="white", linewidth=0.3)

# Theoretical Hardy-Weinberg equilibrium reference lines (cumulative, to match stack)
ax.axhline(hw_BB * 100, color="white", linewidth=1.1, linestyle=(0, (2, 2)))
ax.axhline((hw_BB + hw_Bb) * 100, color="white", linewidth=1.1, linestyle=(0, (2, 2)))
ax.text(GENERATIONS * 1.01, hw_BB * 100, f"HW: {hw_BB*100:.1f}%", va="center",
        ha="left", fontsize=9, color=COL_MUTED)
ax.text(GENERATIONS * 1.01, (hw_BB + hw_Bb / 2) * 100, f"HW carrier\nband: {hw_Bb*100:.1f}%",
        va="center", ha="left", fontsize=9, color=COL_MUTED)
ax.text(GENERATIONS * 1.01, 100 - hw_bb * 100 / 2, f"HW blue: {hw_bb*100:.2f}%",
        va="center", ha="left", fontsize=9, color=COL_MUTED)

ax.set_xlim(0, GENERATIONS)
ax.set_ylim(0, 100)
ax.set_xlabel("Generation")
ax.set_ylabel("Share of population (%)")
ax.set_title(f"One island, one history: genotype mix over {GENERATIONS} generations\n"
             f"(K={K_ISLAND}, start: {N_BROWN0} BB + {N_BLUE0} bb)")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
strip_spines(ax)
fig.tight_layout()
fig.savefig("fig1_genotype_composition.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 2. Many replicate runs on the island -> allele frequency trajectories (drift)
# ----------------------------------------------------------------------
q_trajectories = np.empty((N_REPLICATES, GENERATIONS + 1))
lost_gen = np.full(N_REPLICATES, -1)  # generation at which blue allele hits 0, -1 if never
for r in range(N_REPLICATES):
    res = simulate(N_BROWN0, 0, N_BLUE0, K_ISLAND, GENERATIONS, GROWTH_FACTOR, rng_master)
    q_trajectories[r] = res["q"]
    zero_gens = np.where(res["q"] == 0)[0]
    if len(zero_gens) > 0:
        lost_gen[r] = zero_gens[0]

fig, ax = plt.subplots(figsize=(9, 5.5))
for r in range(min(N_REPLICATES, 60)):
    ax.plot(gens, q_trajectories[r] * 100, color=COL_Q, alpha=0.10, linewidth=1)
mean_q = q_trajectories.mean(axis=0)
ax.plot(gens, mean_q * 100, color=COL_MEAN, linewidth=2.5,
        label=f"Mean over {N_REPLICATES} runs")
ax.axhline(q0 * 100, color="#B23A3A", linewidth=1.8, linestyle="--",
           label=f"Theoretical constant (infinite population): {q0*100:.2f}%")
ax.set_xlim(0, GENERATIONS)
ax.set_ylim(0, max(q_trajectories.max() * 100 * 1.15, q0 * 100 * 1.3))
ax.set_xlabel("Generation")
ax.set_ylabel("Blue allele frequency, q (%)")
ax.set_title(f"Blue-allele frequency doesn't trend anywhere — it drifts\n"
             f"({min(N_REPLICATES,60)} of {N_REPLICATES} sample island histories, K={K_ISLAND})")
ax.legend(loc="upper right")
strip_spines(ax)
fig.tight_layout()
fig.savefig("fig2_allele_frequency_drift.png", dpi=150)
plt.close(fig)

frac_lost = np.mean(lost_gen >= 0)
print(f"\nOn the small island (K={K_ISLAND}), across {N_REPLICATES} replicate histories:")
print(f"  Blue allele went fully extinct by generation {GENERATIONS} in "
      f"{frac_lost*100:.1f}% of runs")

# ----------------------------------------------------------------------
# 3. Probability of extinction over time (island)
# ----------------------------------------------------------------------
extinct_by_gen = np.array([
    np.mean((lost_gen >= 0) & (lost_gen <= g)) for g in gens
])
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gens, extinct_by_gen * 100, color=COL_bb, linewidth=2.5)
ax.fill_between(gens, 0, extinct_by_gen * 100, color=COL_bb, alpha=0.12)
ax.set_xlim(0, GENERATIONS)
ax.set_ylim(0, max(5, extinct_by_gen.max() * 100 * 1.3))
ax.set_xlabel("Generation")
ax.set_ylabel("Runs where blue allele is fully gone (%)")
ax.set_title(f"Chance the blue allele has vanished entirely by chance\n(small island, K={K_ISLAND})")
strip_spines(ax)
fig.tight_layout()
fig.savefig("fig3_extinction_probability.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 4. Island (small K) vs large interconnected population (large K):
#    distribution of final allele frequency after GENERATIONS generations
# ----------------------------------------------------------------------
# Scale up the large population proportionally (same 10:100 founder ratio)
scale = K_LARGE / (N_BROWN0 + N_BLUE0)
N_BROWN0_L = int(round(N_BROWN0 * scale))
N_BLUE0_L = int(round(N_BLUE0 * scale))

final_q_small = np.empty(N_REPLICATES)
final_q_large = np.empty(N_REPLICATES)
for r in range(N_REPLICATES):
    res_s = simulate(N_BROWN0, 0, N_BLUE0, K_ISLAND, GENERATIONS, GROWTH_FACTOR, rng_master)
    res_l = simulate(N_BROWN0_L, 0, N_BLUE0_L, K_LARGE, GENERATIONS, GROWTH_FACTOR, rng_master)
    final_q_small[r] = res_s["q"][-1]
    final_q_large[r] = res_l["q"][-1]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
bins = np.linspace(0, max(final_q_small.max(), q0 * 3), 40)
axes[0].hist(final_q_small * 100, bins=bins * 100, color=COL_Q, alpha=0.85, edgecolor="white")
axes[0].axvline(q0 * 100, color="#B23A3A", linewidth=2, linestyle="--", label="Starting q₀")
axes[0].set_title(f"Small island\nK={K_ISLAND}")
axes[0].set_xlabel("Blue allele frequency at\ngeneration %d (%%)" % GENERATIONS)
axes[0].set_ylabel("Number of replicate histories")
axes[0].legend(loc="upper right", fontsize=9)
strip_spines(axes[0])

bins2 = np.linspace(0, max(final_q_large.max() * 1.2, q0 * 1.5), 40)
axes[1].hist(final_q_large * 100, bins=bins2 * 100, color=COL_MEAN, alpha=0.85, edgecolor="white")
axes[1].axvline(q0 * 100, color="#B23A3A", linewidth=2, linestyle="--", label="Starting q₀")
axes[1].set_title(f"Large, well-mixed population\nK={K_LARGE}")
axes[1].set_xlabel("Blue allele frequency at\ngeneration %d (%%)" % GENERATIONS)
axes[1].legend(loc="upper right", fontsize=9)
strip_spines(axes[1])

fig.suptitle("Same starting ratio, same rules — population size controls how much q wanders",
             fontweight="bold")
fig.tight_layout()
fig.savefig("fig4_island_vs_large_population.png", dpi=150)
plt.close(fig)

print(f"\nSmall island (K={K_ISLAND}) final q: mean={final_q_small.mean()*100:.2f}%, "
      f"std={final_q_small.std()*100:.2f}%, "
      f"lost in {np.mean(final_q_small==0)*100:.1f}% of runs")
print(f"Large population (K={K_LARGE}) final q: mean={final_q_large.mean()*100:.2f}%, "
      f"std={final_q_large.std()*100:.2f}%, "
      f"lost in {np.mean(final_q_large==0)*100:.1f}% of runs")

print("\nSaved: fig1_genotype_composition.png, fig2_allele_frequency_drift.png, "
      "fig3_extinction_probability.png, fig4_island_vs_large_population.png")
