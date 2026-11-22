#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 4: Feature Importance & Shapley Analysis
==============================================
Run: python -I scripts/04_shap_analysis.py

Uses ablation results to compute modality-level Shapley values,
per-species multimodal benefit, and feature-group contributions.

Generates: fig4_shapley.pdf, fig5_species_benefit.pdf,
           shapley_values.csv
"""

import json
import itertools
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

BASE = Path(__file__).resolve().parent.parent   # scripts/ -> project root
RESULTS = BASE / "results"
OUTPUT = RESULTS

# ══════════════════════════════════════════════════════
# Load ablation results
# ══════════════════════════════════════════════════════
results_df = pd.read_csv(RESULTS / "ablation_results.csv")
with open(RESULTS / "ablation_full.json") as f:
    full_results = json.load(f)

# Build lookup: config_name -> test_mse
mse_lookup = dict(zip(results_df["name"], results_df["test_mse"]))
top30_lookup = dict(zip(results_df["name"], results_df["test_top30"]))

print("Loaded ablation results:")
print(results_df[["name", "test_mse", "test_top30"]].to_string(index=False))

# ══════════════════════════════════════════════════════
# Part 1: Modality Shapley Values (approximate)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 1: Modality-level Shapley Values")
print("=" * 60)

# We have 4 modalities: satellite (S), environment (E), audio (A), movement (M)
# Available coalitions from our ablation configs:
#   {S}       -> satellite_only
#   {E}       -> env_only
#   {A}       -> audio_only
#   {S,E}     -> sat+env
#   {E,A}     -> env+audio
#   {E,M}     -> env+move
#   {S,E,A,M} -> sat+env+audio+move
#
# Missing: {M}, {S,A}, {S,M}, {A,M}, {S,E,A}, {S,E,M}, {S,A,M}, {E,A,M}, {}
#
# Strategy: compute marginal contributions from available pairs,
# then average to get approximate Shapley values.

modalities = ["satellite", "environment", "audio", "movement"]

# Map coalition frozensets to their MSE/Top30 (using negative MSE so higher = better)
coalition_map = {
    frozenset(["satellite"]):                           "satellite_only",
    frozenset(["environment"]):                         "env_only",
    frozenset(["audio"]):                               "audio_only",
    frozenset(["satellite", "environment"]):             "sat+env",
    frozenset(["environment", "audio"]):                 "env+audio",
    frozenset(["environment", "movement"]):              "env+move",
    frozenset(["satellite", "environment", "audio", "movement"]): "sat+env+audio+move",
}

# For each modality, find all pairs (S_without, S_with) where adding the modality
# transitions S_without -> S_with, and both are in our coalition_map
def compute_marginal_contributions(metric_lookup, higher_is_better=True):
    """Compute marginal contributions for each modality from available coalitions."""
    contributions = {m: [] for m in modalities}

    for modality in modalities:
        # Check all possible coalitions that include this modality
        for config_set, config_name in coalition_map.items():
            if modality not in config_set:
                continue
            # S_with = config_set, S_without = config_set - {modality}
            s_without = config_set - {modality}

            if len(s_without) == 0:
                # Adding modality to empty set — use a baseline
                # We approximate empty-set performance as mean prediction (worst case)
                # For MSE with mean prediction, baseline ≈ variance of targets
                # For Top-30 with random, baseline ≈ 30/1054 ≈ 0.028
                if higher_is_better:
                    baseline = 0.028  # random Top-30
                else:
                    baseline = 0.006  # approximate MSE of predicting mean
                marginal = metric_lookup[config_name] - baseline
            elif s_without in coalition_map:
                without_name = coalition_map[s_without]
                marginal = metric_lookup[config_name] - metric_lookup[without_name]
            else:
                continue  # can't compute this marginal

            if not higher_is_better:
                marginal = -marginal  # for MSE, reduction is good

            # Weight by 1/(|N| * C(|N|-1, |S_without|)) for Shapley
            n = len(modalities)
            s = len(s_without)
            import math
            weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            contributions[modality].append((marginal, weight))

    # Weighted average
    shapley = {}
    for m in modalities:
        if contributions[m]:
            total_weight = sum(w for _, w in contributions[m])
            shapley[m] = sum(v * w for v, w in contributions[m]) / total_weight
        else:
            shapley[m] = 0.0
    return shapley

shapley_mse = compute_marginal_contributions(mse_lookup, higher_is_better=False)
shapley_top30 = compute_marginal_contributions(top30_lookup, higher_is_better=True)

print("\nApproximate Shapley Values (from available coalitions):")
print(f"  {'Modality':<15} {'φ(MSE↓)':<12} {'φ(Top30↑)':<12}")
print("  " + "-" * 38)
for m in modalities:
    print(f"  {m:<15} {shapley_mse[m]:>+.6f}   {shapley_top30[m]:>+.4f}")

# Save to CSV
shapley_df = pd.DataFrame({
    "modality": modalities,
    "shapley_mse_reduction": [shapley_mse[m] for m in modalities],
    "shapley_top30_gain": [shapley_top30[m] for m in modalities],
})
shapley_df.to_csv(OUTPUT / "shapley_values.csv", index=False)
print(f"\n  Saved → shapley_values.csv")

# ══════════════════════════════════════════════════════
# Figure 4: Shapley values bar chart
# ══════════════════════════════════════════════════════
print("\nGenerating Figure 4...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

colors_shap = ["#4C78A8", "#54A24B", "#E45756", "#F58518"]
labels = ["Satellite", "Environment", "Audio", "Movement"]

# MSE reduction (Shapley)
vals_mse = [shapley_mse[m] for m in modalities]
bars = ax1.bar(labels, vals_mse, color=colors_shap, edgecolor="white", width=0.6)
ax1.set_ylabel("Shapley Value (MSE reduction)")
ax1.set_title("(a) Modality Contribution to MSE ↓")
ax1.axhline(0, color="k", linewidth=0.5)
for bar, v in zip(bars, vals_mse):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00002,
             f"{v:+.5f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Top-30 gain (Shapley)
vals_top30 = [shapley_top30[m] for m in modalities]
bars = ax2.bar(labels, vals_top30, color=colors_shap, edgecolor="white", width=0.6)
ax2.set_ylabel("Shapley Value (Top-30 Recall gain)")
ax2.set_title("(b) Modality Contribution to Top-30 ↑")
ax2.axhline(0, color="k", linewidth=0.5)
for bar, v in zip(bars, vals_top30):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.002 if v >= 0 else bar.get_height() - 0.012,
             f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT / "fig4_shapley.pdf", bbox_inches="tight")
plt.savefig(OUTPUT / "fig4_shapley.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✅ fig4_shapley.pdf")

# ══════════════════════════════════════════════════════
# Part 2: Per-species multimodal benefit
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: Per-Species Multimodal Benefit")
print("=" * 60)

# Compare full model vs satellite_only and env_only
configs_to_compare = [
    ("sat+env+audio+move", "satellite_only", "Audio+Env+Move benefit over Satellite"),
    ("sat+env+audio+move", "env_only", "Full multimodal benefit over Env-only"),
    ("sat+env", "satellite_only", "Environment benefit over Satellite"),
    ("env+audio", "env_only", "Audio benefit over Env-only"),
]

pred_files = {}
for name in results_df["name"]:
    fp = RESULTS / f"preds_{name}.npz"
    if fp.exists():
        data = np.load(fp)
        pred_files[name] = {
            "preds": data["test_preds"],
            "targets": data["test_targets"],
        }

if not pred_files:
    print("  ⚠️ No prediction files found, skipping per-species analysis")
else:
    targets = list(pred_files.values())[0]["targets"]
    n_hotspots, n_species = targets.shape
    print(f"  Test set: {n_hotspots} hotspots × {n_species} species")

    # Species prevalence (mean encounter rate)
    species_prevalence = targets.mean(axis=0)

    # Per-species MSE for each config
    per_species_mse = {}
    for name, data in pred_files.items():
        per_species_mse[name] = ((data["preds"] - targets) ** 2).mean(axis=0)

    # ── Figure 5: Per-species benefit analysis ──
    print("\nGenerating Figure 5...")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for idx, (better, worse, title) in enumerate(configs_to_compare):
        ax = axes[idx // 2, idx % 2]

        if better not in per_species_mse or worse not in per_species_mse:
            ax.text(0.5, 0.5, f"Missing: {better} or {worse}",
                    transform=ax.transAxes, ha="center")
            ax.set_title(title)
            continue

        improvement = per_species_mse[worse] - per_species_mse[better]
        pct_improved = (improvement > 0).mean() * 100

        # Color by improvement direction
        colors_scatter = np.where(improvement > 0, "#54A24B", "#E45756")

        ax.scatter(species_prevalence, improvement, c=colors_scatter,
                   alpha=0.3, s=6, rasterized=True)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Species Prevalence (mean encounter rate)")
        ax.set_ylabel("MSE Improvement (>0 = multimodal better)")
        ax.set_title(f"({chr(97+idx)}) {title}")
        ax.set_xscale("symlog", linthresh=1e-4)

        # Annotation
        ax.text(0.02, 0.98, f"{pct_improved:.0f}% species improved",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Correlation
        mask = species_prevalence > 0
        if mask.sum() > 10:
            r, p = stats.spearmanr(species_prevalence[mask], improvement[mask])
            ax.text(0.02, 0.88, f"ρ = {r:.3f} (p = {p:.1e})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT / "fig5_species_benefit.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT / "fig5_species_benefit.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✅ fig5_species_benefit.pdf")

    # ── Summary statistics ──
    print("\n  Per-species improvement summary:")
    for better, worse, title in configs_to_compare:
        if better not in per_species_mse or worse not in per_species_mse:
            continue
        imp = per_species_mse[worse] - per_species_mse[better]
        print(f"\n  {title}:")
        print(f"    Mean improvement:  {imp.mean():+.6f}")
        print(f"    Median:            {np.median(imp):+.6f}")
        print(f"    Species improved:  {(imp > 0).sum()}/{len(imp)} "
              f"({(imp > 0).mean()*100:.1f}%)")
        print(f"    Species worsened:  {(imp < 0).sum()}/{len(imp)}")

        # Paired t-test
        t, p = stats.ttest_rel(per_species_mse[worse], per_species_mse[better])
        print(f"    Paired t-test:     t={t:.2f}, p={p:.2e} "
              f"({'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'})")

    # ── Top/bottom species ──
    if "sat+env+audio+move" in per_species_mse and "env_only" in per_species_mse:
        imp_full = per_species_mse["env_only"] - per_species_mse["sat+env+audio+move"]

        print("\n  Top 10 species MOST improved by full multimodal (vs env_only):")
        top_idx = np.argsort(imp_full)[-10:][::-1]
        for rank, i in enumerate(top_idx, 1):
            print(f"    {rank:2d}. species_{i:04d}  "
                  f"prev={species_prevalence[i]:.4f}  "
                  f"Δmse={imp_full[i]:+.6f}")

        print("\n  Top 10 species MOST hurt by full multimodal (vs env_only):")
        bot_idx = np.argsort(imp_full)[:10]
        for rank, i in enumerate(bot_idx, 1):
            print(f"    {rank:2d}. species_{i:04d}  "
                  f"prev={species_prevalence[i]:.4f}  "
                  f"Δmse={imp_full[i]:+.6f}")

# ══════════════════════════════════════════════════════
# Part 3: Marginal contribution table
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3: Marginal Contributions Table")
print("=" * 60)

print("\nAdding modality to env_only baseline:")
if "env_only" in mse_lookup:
    base_mse = mse_lookup["env_only"]
    base_top = top30_lookup["env_only"]

    additions = [
        ("+ satellite",  "sat+env"),
        ("+ audio",      "env+audio"),
        ("+ movement",   "env+move"),
        ("+ all three",  "sat+env+audio+move"),
    ]

    print(f"  {'Addition':<15} {'MSE':<10} {'ΔMSE':<12} {'Top30':<8} {'ΔTop30':<10}")
    print("  " + "-" * 55)
    print(f"  {'env_only':<15} {base_mse:<10.5f} {'—':<12} {base_top:<8.4f} {'—':<10}")
    for label, name in additions:
        if name in mse_lookup:
            dm = mse_lookup[name] - base_mse
            dt = top30_lookup[name] - base_top
            print(f"  {label:<15} {mse_lookup[name]:<10.5f} {dm:<+12.5f} "
                  f"{top30_lookup[name]:<8.4f} {dt:<+10.4f}")

print("\nAdding modality to satellite_only baseline:")
if "satellite_only" in mse_lookup:
    base_mse = mse_lookup["satellite_only"]
    base_top = top30_lookup["satellite_only"]

    additions = [
        ("+ environment", "sat+env"),
    ]

    print(f"  {'Addition':<15} {'MSE':<10} {'ΔMSE':<12} {'Top30':<8} {'ΔTop30':<10}")
    print("  " + "-" * 55)
    print(f"  {'sat_only':<15} {base_mse:<10.5f} {'—':<12} {base_top:<8.4f} {'—':<10}")
    for label, name in additions:
        if name in mse_lookup:
            dm = mse_lookup[name] - base_mse
            dt = top30_lookup[name] - base_top
            print(f"  {label:<15} {mse_lookup[name]:<10.5f} {dm:<+12.5f} "
                  f"{top30_lookup[name]:<8.4f} {dt:<+10.4f}")

# ══════════════════════════════════════════════════════
# LaTeX Shapley table
# ══════════════════════════════════════════════════════
print("\n% LaTeX Shapley table:")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Approximate Shapley values for each modality.}")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"Modality & $\phi$(MSE $\downarrow$) & $\phi$(Top-30 $\uparrow$) \\")
print(r"\midrule")
for m, label in zip(modalities, ["Satellite", "Environment", "Audio", "Movement"]):
    print(f"{label} & {shapley_mse[m]:+.5f} & {shapley_top30[m]:+.4f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print(f"\n✅ All figures saved to {OUTPUT}")
