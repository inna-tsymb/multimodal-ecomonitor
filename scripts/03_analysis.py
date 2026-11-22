#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 3: Analysis & Publication Figures
======================================
Run: python -I 03_analysis.py

Generates: fig1_ablation.pdf, fig2_learning_curves.pdf,
           fig3_per_species.pdf, LaTeX table
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
OUTPUT = RESULTS

results_df = pd.read_csv(RESULTS / "ablation_results.csv")
with open(RESULTS / "ablation_full.json") as f:
    full_results = json.load(f)

print("Ablation results:")
print(results_df[["name", "test_mse", "test_top30", "n_params"]].to_string(index=False))

# ══════════════════════════════════════════════════════
# Figure 1: Ablation bar chart
# ══════════════════════════════════════════════════════
print("\nGenerating Figure 1...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

n = len(results_df)
colors = plt.cm.tab10(np.linspace(0, 0.7, n))
x = np.arange(n)

# MSE
df_mse = results_df.sort_values("test_mse")
bars = ax1.barh(x, df_mse["test_mse"], color=colors, height=0.6, edgecolor="white")
ax1.set_yticks(x)
ax1.set_yticklabels(df_mse["name"], fontsize=9)
ax1.set_xlabel("Test MSE ↓")
ax1.set_title("(a) Test MSE by Configuration")
ax1.invert_yaxis()
for bar, val in zip(bars, df_mse["test_mse"]):
    ax1.text(bar.get_width() + 0.00005, bar.get_y() + bar.get_height()/2,
             f"{val:.5f}", va="center", fontsize=8, fontweight="bold")

# Top-30
df_top = results_df.sort_values("test_top30", ascending=False)
bars = ax2.barh(x, df_top["test_top30"], color=colors, height=0.6, edgecolor="white")
ax2.set_yticks(x)
ax2.set_yticklabels(df_top["name"], fontsize=9)
ax2.set_xlabel("Top-30 Recall ↑")
ax2.set_title("(b) Top-30 Recall by Configuration")
ax2.invert_yaxis()
for bar, val in zip(bars, df_top["test_top30"]):
    ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT / "fig1_ablation.pdf", bbox_inches="tight")
plt.savefig(OUTPUT / "fig1_ablation.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✅ fig1_ablation.pdf")

# ══════════════════════════════════════════════════════
# Figure 2: Learning curves
# ══════════════════════════════════════════════════════
print("Generating Figure 2...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
colors_lc = plt.cm.tab10(np.linspace(0, 0.7, len(full_results)))

for res, c in zip(full_results, colors_lc):
    hist = pd.DataFrame(res["history"])
    ax1.plot(hist["epoch"], hist["val_mse"], label=res["name"], color=c, linewidth=2)
    ax2.plot(hist["epoch"], hist["val_top30"], label=res["name"], color=c, linewidth=2)

ax1.set(xlabel="Epoch", ylabel="Validation MSE")
ax1.set_title("(a) Validation MSE"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2)
ax2.set(xlabel="Epoch", ylabel="Top-30 Recall")
ax2.set_title("(b) Validation Top-30 Recall"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT / "fig2_learning_curves.pdf", bbox_inches="tight")
plt.savefig(OUTPUT / "fig2_learning_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✅ fig2_learning_curves.pdf")

# ══════════════════════════════════════════════════════
# Figure 3: Per-species analysis
# ══════════════════════════════════════════════════════
print("Generating Figure 3 (per-species)...")

# Find baseline and best multimodal
baseline_name = "env_only"
best_name = None
for candidate in ["sat+env+audio+move", "env+audio+move", "env+audio"]:
    if any(r["name"] == candidate for r in full_results):
        best_name = candidate
        break

if best_name and baseline_name:
    base_file = RESULTS / f"preds_{baseline_name}.npz"
    best_file = RESULTS / f"preds_{best_name}.npz"

    if base_file.exists() and best_file.exists():
        base_data = np.load(base_file)
        best_data = np.load(best_file)

        base_preds = base_data["test_preds"]
        best_preds = best_data["test_preds"]
        targets = base_data["test_targets"]

        # Per-species MSE
        base_mse = ((base_preds - targets) ** 2).mean(axis=0)
        best_mse = ((best_preds - targets) ** 2).mean(axis=0)
        improvement = base_mse - best_mse

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Scatter
        ax = axes[0]
        ax.scatter(base_mse, best_mse, alpha=0.3, s=8, c="#4C78A8")
        lim = max(base_mse.max(), best_mse.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"Per-species MSE ({baseline_name})")
        ax.set_ylabel(f"Per-species MSE ({best_name})")
        ax.set_title("(a) Per-species Comparison")

        # Improvement distribution
        ax = axes[1]
        ax.hist(improvement, bins=50, color="#72B7B2", edgecolor="white")
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("MSE Improvement (positive = multimodal better)")
        ax.set_title("(b) Improvement Distribution")
        pct_improved = (improvement > 0).mean() * 100
        ax.text(0.95, 0.95, f"{pct_improved:.0f}% improved",
                transform=ax.transAxes, ha="right", va="top", fontsize=10)

        # Top improved species
        ax = axes[2]
        top_n = 20
        top_idx = np.argsort(improvement)[-top_n:][::-1]
        ax.barh(range(top_n), improvement[top_idx], color="#E45756", height=0.6)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f"species_{i}" for i in top_idx], fontsize=7)
        ax.set_xlabel("MSE Improvement")
        ax.set_title(f"(c) Top {top_n} Most Improved Species")
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(OUTPUT / "fig3_per_species.pdf", bbox_inches="tight")
        plt.savefig(OUTPUT / "fig3_per_species.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✅ fig3_per_species.pdf")

        # Statistical significance
        t_stat, p_val = stats.ttest_rel(base_mse, best_mse)
        print(f"\n  Paired t-test ({baseline_name} vs {best_name}):")
        print(f"    t = {t_stat:.3f}, p = {p_val:.2e}")
        print(f"    {'Significant' if p_val < 0.05 else 'Not significant'} (α=0.05)")
    else:
        print("  ⚠️ Prediction files not found — skipping per-species analysis")
else:
    print("  ⚠️ Baseline/best config not found")

# ══════════════════════════════════════════════════════
# LaTeX table
# ══════════════════════════════════════════════════════
print("\n% LaTeX table:")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Ablation study results on BATIS Kenya test set.}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Configuration & Parameters & MSE $\downarrow$ & Top-30 Recall $\uparrow$ \\")
print(r"\midrule")

for _, row in results_df.sort_values("test_mse").iterrows():
    name_tex = row["name"].replace("_", r"\_").replace("+", " + ")
    params_k = row["n_params"] / 1000
    print(f"{name_tex} & {params_k:.0f}K & {row['test_mse']:.5f} & {row['test_top30']:.4f} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print(f"\n✅ All figures saved to {OUTPUT}")