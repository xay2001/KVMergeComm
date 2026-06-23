#!/usr/bin/env python3
"""Step-0 visualization: how samples split by their oracle minimal budget,
and the coverage curve. Data hard-coded from analyze_oracle.py output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

budgets = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
xlabels = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.7"]

# oracle minimal budget distribution (% of all N samples); unsolved bin dropped
dist = {
    "hotpotqa":   [7.2, 16.8, 27.6, 13.8, 9.8, 3.0, 2.0, 0.6],
    "musique":    [8.8, 22.4, 9.8, 5.4, 6.4, 2.4, 1.8, 1.0],
    "twowikimqa": [40.0, 7.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
}
# coverage: % of solvable samples with oracle <= r_high
cover = {
    "hotpotqa":   [8.9, 29.7, 63.9, 80.9, 93.1, 96.8, 99.3, 100.0],
    "musique":    [15.2, 53.8, 70.7, 80.0, 91.0, 95.2, 98.3, 100.0],
    "twowikimqa": [76.9, 90.4, 92.3, 94.2, 96.2, 98.1, 98.1, 100.0],
}
colors = {"hotpotqa": "#2c7fb8", "musique": "#d95f0e", "twowikimqa": "#31a354"}

plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# ---- left: grouped bar of oracle-min-budget distribution ----
x = np.arange(len(xlabels))
w = 0.26
for i, ds in enumerate(["hotpotqa", "musique", "twowikimqa"]):
    ax1.bar(x + (i - 1) * w, dist[ds], w, label=ds, color=colors[ds], edgecolor="white")
ax1.set_xticks(x); ax1.set_xticklabels(xlabels)
ax1.set_xlabel("Oracle minimal budget r per query (min. KV fraction to answer)")
ax1.set_ylabel("Share of samples (%)")
ax1.set_title("(1) Samples by required budget: needs vary widely")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)
ax1.axvspan(-0.5, 3.5, color="green", alpha=0.05)
ax1.text(1.5, ax1.get_ylim()[1] * 0.92, "low budget suffices", ha="center", color="green", fontsize=10)

# ---- right: coverage curve ----
for ds in ["hotpotqa", "musique", "twowikimqa"]:
    ax2.plot(budgets, cover[ds], "o-", label=ds, color=colors[ds], lw=2, ms=6)
ax2.axhline(90, ls="--", color="gray", alpha=0.7)
ax2.text(0.46, 91, "90% coverage", color="gray", fontsize=9)
ax2.axvline(0.3, ls=":", color="red", alpha=0.6)
ax2.text(0.31, 30, "r=0.3:\n~90% covered", color="red", fontsize=9)
ax2.set_xlabel("Budget upper bound r_high")
ax2.set_ylabel("Coverage of solvable samples (%)")
ax2.set_title("(2) Coverage curve: 2-3 rungs cover ~90% of solvable")
ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 102)

fig.suptitle("Step 0 - Oracle minimal budget distribution & coverage (headroom check)", fontsize=14, y=1.02)
fig.tight_layout()
out = "snapshots/oracle_headroom.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
