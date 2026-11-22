#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 8: Model Ensemble & Uncertainty Analysis
===============================================
Run: python -I scripts/08_ensemble_uncertainty.py

Analyzes prediction agreement and uncertainty across the 7 ablation
configurations (v2 BirdNET models). Quantifies where models agree,
which species have highest prediction uncertainty, and how modality
combinations affect confidence.

Outputs:
  - fig7_ensemble_uncertainty.pdf / ensemble_uncertainty.png
  - ensemble_analysis.json
  - species_uncertainty.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if __name__ != "__main__":
    import sys; sys.exit(0)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FEATURES_DIR = BASE / "working"
OUTPUT = BASE / "results"

# ══════════════════════════════════════════════════════
# 1. Load all model predictions
# ══════════════════════════════════════════════════════

print("=" * 60)
print("Model Ensemble & Uncertainty Analysis")
print("=" * 60)

# Load species info
species_df = pd.read_csv(DATA / "species_list_kenya.csv")
n_real = len(species_df)
species_names = (species_df["common_name"].values if "common_name" in species_df.columns
                 else species_df.iloc[:, 1].values)

# Load splits
with open(FEATURES_DIR / "satbird_splits.json") as f:
    splits = json.load(f)
test_ids = splits["test"]

# Collect all v2 predictions
CONFIG_LABELS = {
    "sat+env+audio+move": "Full (S+E+A+M)",
    "env+audio": "Env+Audio",
    "audio_only": "Audio only",
    "sat+env": "Sat+Env",
    "env+move": "Env+Move",
    "satellite_only": "Sat only",
    "env_only": "Env only",
}

print("\nLoading predictions...")
preds_dict = {}
targets = None

for config in CONFIG_LABELS:
    p = OUTPUT / f"preds_v2_{config}.npz"
    if not p.exists():
        print(f"  ⚠️ Missing: {p.name}")
        continue
    data = np.load(p)
    test_preds = data["test_preds"][:, :n_real]
    if targets is None:
        targets = data["test_targets"][:, :n_real]
    preds_dict[config] = test_preds
    print(f"  ✅ {config}: {test_preds.shape}")

n_models = len(preds_dict)
n_test = targets.shape[0]
print(f"\nLoaded {n_models} models, {n_test} test hotspots, {n_real} species")

config_names = list(preds_dict.keys())
all_preds = np.stack([preds_dict[c] for c in config_names])  # (n_models, n_test, n_species)

# ══════════════════════════════════════════════════════
# 2. Ensemble statistics
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Ensemble Statistics")
print("=" * 60)

ensemble_mean = all_preds.mean(axis=0)   # (n_test, n_species)
ensemble_std = all_preds.std(axis=0)     # prediction uncertainty
ensemble_range = all_preds.max(axis=0) - all_preds.min(axis=0)

# Per-model MSE
print(f"\n  {'Config':<25} {'MSE':>8} {'Top-30 Acc':>10} {'Corr w/ ensemble':>18}")
print("  " + "-" * 65)
model_mses = {}
for config in config_names:
    mse = ((preds_dict[config] - targets) ** 2).mean()
    # Top-30 accuracy
    top30_acc_list = []
    for i in range(n_test):
        pred_top = set(np.argsort(preds_dict[config][i])[-30:])
        true_top = set(np.argsort(targets[i])[-30:])
        top30_acc_list.append(len(pred_top & true_top) / 30)
    top30 = np.mean(top30_acc_list)
    # Correlation with ensemble mean
    corr = np.corrcoef(preds_dict[config].flatten(), ensemble_mean.flatten())[0, 1]
    model_mses[config] = mse
    label = CONFIG_LABELS[config]
    print(f"  {label:<25} {mse:.5f} {top30:>10.4f} {corr:>18.4f}")

# Ensemble MSE
ens_mse = ((ensemble_mean - targets) ** 2).mean()
ens_top30 = []
for i in range(n_test):
    pred_top = set(np.argsort(ensemble_mean[i])[-30:])
    true_top = set(np.argsort(targets[i])[-30:])
    ens_top30.append(len(pred_top & true_top) / 30)
ens_top30 = np.mean(ens_top30)
print(f"  {'ENSEMBLE MEAN':<25} {ens_mse:.5f} {ens_top30:>10.4f} {'—':>18}")

# ══════════════════════════════════════════════════════
# 3. Species-level uncertainty
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Species Uncertainty Analysis")
print("=" * 60)

# Mean uncertainty (std across models) per species
species_mean_std = ensemble_std.mean(axis=0)       # avg uncertainty across hotspots
species_mean_range = ensemble_range.mean(axis=0)
species_mean_rate = targets.mean(axis=0)            # actual mean encounter rate

# Coefficient of variation of predictions across models
species_cv = np.where(ensemble_mean.mean(axis=0) > 0.001,
                      species_mean_std / ensemble_mean.mean(axis=0), 0)

# Top uncertain species (by std)
top_uncertain = np.argsort(species_mean_std)[-15:][::-1]
print(f"\n  Top 15 most uncertain species (highest prediction spread):")
print(f"  {'Species':<35} {'Mean rate':>10} {'Pred std':>10} {'Pred range':>12} {'CV':>8}")
print("  " + "-" * 78)
for idx in top_uncertain:
    name = species_names[idx] if idx < len(species_names) else f"sp_{idx}"
    print(f"  {name:<35} {species_mean_rate[idx]:>10.4f} {species_mean_std[idx]:>10.4f} "
          f"{species_mean_range[idx]:>12.4f} {species_cv[idx]:>8.2f}")

# Most agreed-upon species (lowest std among common species)
common_mask = species_mean_rate > 0.01
if common_mask.sum() > 0:
    common_std = species_mean_std.copy()
    common_std[~common_mask] = 999
    top_agreed = np.argsort(common_std)[:15]
    print(f"\n  Top 15 most agreed-upon species (common, lowest spread):")
    print(f"  {'Species':<35} {'Mean rate':>10} {'Pred std':>10} {'CV':>8}")
    print("  " + "-" * 65)
    for idx in top_agreed:
        name = species_names[idx] if idx < len(species_names) else f"sp_{idx}"
        print(f"  {name:<35} {species_mean_rate[idx]:>10.4f} {species_mean_std[idx]:>10.4f} "
              f"{species_cv[idx]:>8.2f}")

# ══════════════════════════════════════════════════════
# 4. Modality contribution to disagreement
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Modality Contribution to Model Disagreement")
print("=" * 60)

# Compare models with/without each modality
modality_configs = {
    "Satellite": {
        "with": [c for c in config_names if "sat" in c],
        "without": [c for c in config_names if "sat" not in c],
    },
    "Audio": {
        "with": [c for c in config_names if "audio" in c],
        "without": [c for c in config_names if "audio" not in c and c != "satellite_only"],
    },
    "Environment": {
        "with": [c for c in config_names if "env" in c],
        "without": [c for c in config_names if "env" not in c],
    },
    "Movement": {
        "with": [c for c in config_names if "move" in c],
        "without": [c for c in config_names if "move" not in c],
    },
}

print(f"\n  {'Modality':<15} {'With (mean MSE)':>16} {'Without (mean MSE)':>19} {'Δ MSE':>10} {'Impact':>10}")
print("  " + "-" * 72)
modality_impact = {}
for mod, groups in modality_configs.items():
    if groups["with"] and groups["without"]:
        mse_with = np.mean([model_mses[c] for c in groups["with"]])
        mse_without = np.mean([model_mses[c] for c in groups["without"]])
        delta = mse_with - mse_without
        impact = "improves" if delta < 0 else "hurts"
        print(f"  {mod:<15} {mse_with:>16.5f} {mse_without:>19.5f} {delta:>+10.5f} {impact:>10}")
        modality_impact[mod] = {"mse_with": round(float(mse_with), 5),
                                "mse_without": round(float(mse_without), 5),
                                "delta": round(float(delta), 5)}

# ══════════════════════════════════════════════════════
# 5. Hotspot-level uncertainty
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Hotspot-Level Uncertainty")
print("=" * 60)

hotspot_mean_std = ensemble_std.mean(axis=1)   # avg uncertainty per hotspot
hotspot_mean_range = ensemble_range.mean(axis=1)

# Hotspots where models disagree most
top_disagree = np.argsort(hotspot_mean_std)[-10:][::-1]
print(f"\n  Top 10 most uncertain hotspots:")
print(f"  {'Hotspot ID':<15} {'Mean std':>10} {'Mean range':>12} {'True richness':>15}")
print("  " + "-" * 55)
for idx in top_disagree:
    hid = test_ids[idx] if idx < len(test_ids) else f"idx_{idx}"
    richness = (targets[idx] > 0.01).sum()
    print(f"  {hid:<15} {hotspot_mean_std[idx]:>10.4f} {hotspot_mean_range[idx]:>12.4f} "
          f"{richness:>15}")

# Correlation: uncertainty vs actual richness
from scipy import stats as sp_stats
corr, p = sp_stats.pearsonr(hotspot_mean_std, targets.mean(axis=1))
print(f"\n  Correlation (uncertainty vs mean encounter rate): r={corr:.3f}, p={p:.4f}")

richness_true = (targets > 0.01).sum(axis=1)
corr_r, p_r = sp_stats.pearsonr(hotspot_mean_std, richness_true)
print(f"  Correlation (uncertainty vs species richness): r={corr_r:.3f}, p={p_r:.4f}")

# ══════════════════════════════════════════════════════
# 6. Pairwise model agreement
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Pairwise Model Agreement (Top-30 overlap)")
print("=" * 60)

n_configs = len(config_names)
agreement_matrix = np.zeros((n_configs, n_configs))

for i in range(n_configs):
    for j in range(n_configs):
        overlaps = []
        for h in range(n_test):
            top_i = set(np.argsort(preds_dict[config_names[i]][h])[-30:])
            top_j = set(np.argsort(preds_dict[config_names[j]][h])[-30:])
            overlaps.append(len(top_i & top_j) / 30)
        agreement_matrix[i, j] = np.mean(overlaps)

short_labels = [CONFIG_LABELS[c].split(" (")[0] if "(" in CONFIG_LABELS[c]
                else CONFIG_LABELS[c] for c in config_names]

print(f"\n  {'':>15}", end="")
for sl in short_labels:
    print(f" {sl[:8]:>8}", end="")
print()
for i, sl in enumerate(short_labels):
    print(f"  {sl:>13}", end="  ")
    for j in range(n_configs):
        print(f" {agreement_matrix[i, j]:>7.3f}", end="")
    print()

# ══════════════════════════════════════════════════════
# 7. Visualizations
# ══════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Model Ensemble & Uncertainty Analysis", fontsize=16, fontweight="bold", y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# (a) Model comparison — MSE
ax = fig.add_subplot(gs[0, 0])
labels = [CONFIG_LABELS[c] for c in config_names]
mses = [model_mses[c] for c in config_names]
sort_idx = np.argsort(mses)
colors_a = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_models))
bars = ax.barh(range(n_models), [mses[i] for i in sort_idx],
               color=[colors_a[rank] for rank in range(n_models)])
ax.set_yticks(range(n_models))
ax.set_yticklabels([labels[i] for i in sort_idx], fontsize=8)
ax.axvline(ens_mse, color="red", linestyle="--", linewidth=2, label=f"Ensemble: {ens_mse:.5f}")
ax.set_xlabel("MSE")
ax.set_title("(a) Model MSE Comparison")
ax.legend(fontsize=7)
ax.grid(axis="x", alpha=0.2)

# (b) Species uncertainty distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(species_mean_std[species_mean_rate > 0.001], bins=50, color="#3498db",
        alpha=0.7, edgecolor="white")
ax.axvline(np.median(species_mean_std[species_mean_rate > 0.001]),
           color="red", linestyle="--", label="Median")
ax.set_xlabel("Mean Prediction Std (across models)")
ax.set_ylabel("Number of Species")
ax.set_title("(b) Species Prediction Uncertainty")
ax.legend()

# (c) Uncertainty vs encounter rate
ax = fig.add_subplot(gs[0, 2])
mask = species_mean_rate > 0.001
ax.scatter(species_mean_rate[mask], species_mean_std[mask],
           s=8, alpha=0.4, c="#e74c3c")
ax.set_xlabel("Mean Encounter Rate (ground truth)")
ax.set_ylabel("Prediction Std (across models)")
ax.set_title("(c) Uncertainty vs Abundance")
ax.text(0.95, 0.95, f"r = {np.corrcoef(species_mean_rate[mask], species_mean_std[mask])[0,1]:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# (d) Pairwise agreement heatmap
ax = fig.add_subplot(gs[1, 0:2])
im = ax.imshow(agreement_matrix, cmap="YlOrRd", vmin=0.3, vmax=1.0)
ax.set_xticks(range(n_configs))
ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")
ax.set_yticks(range(n_configs))
ax.set_yticklabels(short_labels, fontsize=8)
for i in range(n_configs):
    for j in range(n_configs):
        ax.text(j, i, f"{agreement_matrix[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if agreement_matrix[i,j] > 0.7 else "black")
plt.colorbar(im, ax=ax, shrink=0.7, label="Top-30 Overlap")
ax.set_title("(d) Pairwise Model Agreement (Top-30 Species Overlap)")

# (e) Modality impact
ax = fig.add_subplot(gs[1, 2])
mod_names = list(modality_impact.keys())
mod_deltas = [modality_impact[m]["delta"] for m in mod_names]
colors_e = ["#2ecc71" if d < 0 else "#e74c3c" for d in mod_deltas]
ax.barh(range(len(mod_names)), [-d * 1000 for d in mod_deltas], color=colors_e)
ax.set_yticks(range(len(mod_names)))
ax.set_yticklabels(mod_names, fontsize=9)
ax.set_xlabel("MSE Improvement (×10³)")
ax.set_title("(e) Modality Impact on MSE")
ax.axvline(0, color="black", linewidth=0.5)
ax.grid(axis="x", alpha=0.2)

# (f) Top uncertain species heatmap
ax = fig.add_subplot(gs[2, 0:2])
n_show = min(15, len(top_uncertain))
show_idx = top_uncertain[:n_show]
heatmap_data = np.zeros((n_show, n_models))
for j, config in enumerate(config_names):
    for i, sp_idx in enumerate(show_idx):
        heatmap_data[i, j] = preds_dict[config][:, sp_idx].mean()

im2 = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(n_models))
ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")
ax.set_yticks(range(n_show))
ax.set_yticklabels([species_names[i][:30] if i < len(species_names) else f"sp_{i}"
                    for i in show_idx], fontsize=7)
plt.colorbar(im2, ax=ax, shrink=0.7, label="Mean predicted rate")
ax.set_title("(f) Top Uncertain Species — Predictions Across Models")

# (g) Hotspot uncertainty distribution
ax = fig.add_subplot(gs[2, 2])
ax.hist(hotspot_mean_std, bins=40, color="#9b59b6", alpha=0.7, edgecolor="white")
ax.axvline(np.median(hotspot_mean_std), color="red", linestyle="--", label="Median")
ax.set_xlabel("Mean Prediction Std (across models)")
ax.set_ylabel("Number of Hotspots")
ax.set_title("(g) Hotspot Prediction Uncertainty")
ax.legend()

plt.savefig(OUTPUT / "ensemble_uncertainty.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT / "fig7_ensemble_uncertainty.pdf", dpi=300, bbox_inches="tight")
print(f"\n✅ Saved figures")

# ── Save CSV ──
uncertainty_df = pd.DataFrame({
    "species_idx": range(n_real),
    "species_name": [species_names[i] if i < len(species_names) else f"sp_{i}" for i in range(n_real)],
    "mean_encounter_rate": np.round(species_mean_rate, 5),
    "pred_std": np.round(species_mean_std, 5),
    "pred_range": np.round(species_mean_range, 5),
    "pred_cv": np.round(species_cv, 4),
})
uncertainty_df.to_csv(OUTPUT / "species_uncertainty.csv", index=False)

# ── Save JSON ──
results = {
    "n_models": n_models,
    "n_test_hotspots": n_test,
    "n_species": n_real,
    "ensemble_mse": round(float(ens_mse), 5),
    "ensemble_top30": round(float(ens_top30), 4),
    "per_model_mse": {CONFIG_LABELS[c]: round(float(model_mses[c]), 5) for c in config_names},
    "modality_impact": modality_impact,
    "mean_species_uncertainty": round(float(species_mean_std.mean()), 5),
    "uncertainty_vs_rate_corr": round(float(np.corrcoef(
        species_mean_rate[mask], species_mean_std[mask])[0, 1]), 4),
    "uncertainty_vs_richness_corr": round(float(corr_r), 4),
}

with open(OUTPUT / "ensemble_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved: ensemble_analysis.json, species_uncertainty.csv")
print(f"\n→ Done! Check results/ensemble_uncertainty.png")
