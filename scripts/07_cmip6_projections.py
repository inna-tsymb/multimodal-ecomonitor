#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 7: CMIP6 Climate Projections (Multi-GCM)
==============================================
Run: python -I scripts/07_cmip6_projections.py

Uses locally downloaded WorldClim CMIP6 downscaled bioclimatic variables,
extracts features at hotspot locations for future scenarios,
runs trained model predictions, and compares current vs future distributions.

Auto-discovers all GCM .tif files in data/cmip6/ and runs projections
for each GCM separately, then computes a multi-GCM ensemble average.

Requires:
  - data/cmip6/wc2.1_*_bioc_*_ssp*_2041-2060.tif  (one or more GCMs)
  - trained model (results/model_v2_*.pt or model_*.pt)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torchvision.models as models
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
CMIP6_DIR = DATA / "cmip6"

# ══════════════════════════════════════════════════════
# 1. Auto-discover GCM files
# ══════════════════════════════════════════════════════
print("=" * 60)
print("CMIP6 Climate Projections — Multi-GCM")
print("=" * 60)

# Pattern: wc2.1_<res>_bioc_<GCM>_<ssp>_<period>.tif
TIF_PATTERN = re.compile(
    r"wc2\.1_(\d+\.?\d*m)_bioc_(.+?)_(ssp\d+)_(\d{4}-\d{4})\.tif$"
)

discovered = {}  # {(gcm, ssp_key): {"file": Path, "gcm": str, "res": str, ...}}
for f in sorted(CMIP6_DIR.glob("wc2.1_*_bioc_*_ssp*_*.tif")):
    m = TIF_PATTERN.match(f.name)
    if not m:
        continue
    res, gcm, ssp, period = m.groups()
    if f.stat().st_size < 1_000_000:
        print(f"  ⚠️  Skipping {f.name} — too small ({f.stat().st_size} bytes)")
        continue
    # Normalize ssp key
    ssp_label = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}.get(ssp, ssp.upper())
    year_mid = str((int(period[:4]) + int(period[5:])) // 2)
    ssp_key = f"{ssp}_{year_mid}"
    key = (gcm, ssp_key)
    discovered[key] = {
        "file": f, "gcm": gcm, "res": res, "ssp": ssp,
        "period": period, "ssp_key": ssp_key,
        "label": f"{ssp_label} ({year_mid})",
    }

gcm_names = sorted(set(gcm for gcm, _ in discovered))
ssp_keys = sorted(set(sk for _, sk in discovered))

print(f"\n  Found {len(discovered)} files across {len(gcm_names)} GCM(s):")
for (gcm, sk), info in sorted(discovered.items()):
    print(f"    {gcm:20s}  {info['label']:16s}  res={info['res']:>6s}  {info['file'].name}")

if not discovered:
    print("\n⚠️ No CMIP6 .tif files found in data/cmip6/")
    import sys; sys.exit(1)


# ══════════════════════════════════════════════════════
# 2. Load current features & normalization stats
# ══════════════════════════════════════════════════════
print("\nLoading current features...")

with open(FEATURES_DIR / "satbird_splits.json") as f:
    splits = json.load(f)
with open(FEATURES_DIR / "species_metadata.json") as f:
    metadata = json.load(f)

species_df = pd.read_csv(DATA / "species_list_kenya.csv")
n_real = len(species_df)
n_species = 1054  # with padding

all_ids = splits["train"] + splits["val"] + splits["test"]
all_idx_map = {hid: i for i, hid in enumerate(all_ids)}

bioclim_current = np.load(FEATURES_DIR / "bioclim_features.npy")
landcover = np.load(FEATURES_DIR / "landcover_features.npy")
terrain = np.load(FEATURES_DIR / "terrain_features.npy")
inat = np.load(FEATURES_DIR / "inat_features.npy")

n_bio = bioclim_current.shape[1]
print(f"  Bioclim dims: {n_bio}, Hotspots: {len(all_ids)}")

# Audio
birdnet_file = FEATURES_DIR / "audio_embeddings_birdnet.npz"
if birdnet_file.exists():
    audio_data = np.load(birdnet_file)
    AUDIO_DIM = 1024
else:
    audio_data = np.load(FEATURES_DIR / "audio_embeddings.npz")
    AUDIO_DIM = 128
audio_embeddings = audio_data["embeddings"]
audio_mask = audio_data["mask"].astype(bool)

# Movement
move_df = pd.read_csv(FEATURES_DIR / "movement_features.csv")
MOVE_COLS = ["mean_step_length", "std_step_length", "mean_speed", "max_speed",
             "mean_turning_angle", "displacement", "straightness", "home_range_km2"]
move_features = np.zeros((n_species, len(MOVE_COLS)), dtype=np.float32)
for _, row in move_df.iterrows():
    idx = int(row["species_idx"])
    if idx < n_species:
        move_features[idx] = [row.get(c, 0) for c in MOVE_COLS]

# Normalize
def safe_normalize(arr):
    mu = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return (arr - mu) / std, mu, std

bioclim_norm, bio_mu, bio_std = safe_normalize(bioclim_current)
landcover_norm = landcover
terrain_norm, ter_mu, ter_std = safe_normalize(terrain)
inat_norm, inat_mu, inat_std = safe_normalize(inat)

env_current = np.concatenate([bioclim_norm, landcover_norm, terrain_norm, inat_norm], axis=1)
env_dim = env_current.shape[1]
print(f"  Env dim: {env_dim}")


# ══════════════════════════════════════════════════════
# 3. Extract future bioclim from multi-band GeoTIFFs
# ══════════════════════════════════════════════════════

def load_hotspot_coords():
    """Try to get coordinates — first from cached JSON, then from per-hotspot metadata."""
    coords = {}
    # 1. Try cached hotspot_coordinates.json (fastest)
    coord_file = FEATURES_DIR / "hotspot_coordinates.json"
    if coord_file.exists():
        with open(coord_file) as f:
            raw = json.load(f)
        for hid, c in raw.items():
            if hid in all_idx_map:
                coords[hid] = (float(c["lat"]), float(c["lon"]))
        if coords:
            return coords
    # 2. Fallback: per-hotspot JSON files
    for search_dir in [DATA / "batis", DATA / "satbird_targets"]:
        if not search_dir.exists():
            continue
        for hid in all_ids:
            for subdir in ["", "train", "val", "test"]:
                p = search_dir / subdir / f"{hid}.json" if subdir else search_dir / f"{hid}.json"
                if p.exists():
                    with open(p) as f:
                        m = json.load(f)
                    lat = m.get("lat", m.get("latitude"))
                    lon = m.get("lon", m.get("longitude"))
                    if lat is not None and lon is not None:
                        coords[hid] = (float(lat), float(lon))
                    break
        if coords:
            break
    return coords


def extract_bioclim_from_multiband(tif_path, coords_dict):
    """Extract bioclim values from a multi-band GeoTIFF (19 bands = bio1..bio19)."""
    if not coords_dict:
        return None

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        print(f"    Raster: {n_bands} bands, {src.width}x{src.height}, CRS={src.crs}")

        hotspot_ids = [hid for hid in all_ids if hid in coords_dict]
        features = np.full((len(hotspot_ids), min(n_bands, n_bio)), np.nan, dtype=np.float32)

        all_data = src.read()  # shape: (n_bands, height, width)

        extracted = 0
        for i, hid in enumerate(hotspot_ids):
            lat, lon = coords_dict[hid]
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    for b in range(min(n_bands, n_bio)):
                        val = all_data[b, row, col]
                        if val != src.nodata and not np.isnan(val):
                            features[i, b] = val
                    extracted += 1
            except (IndexError, ValueError):
                pass

        print(f"    Extracted values for {extracted}/{len(hotspot_ids)} hotspots")

        # WorldClim bioclim temperature values are stored as deg_C * 10
        temp_cols = features[:, :min(11, features.shape[1])]
        valid_temp = temp_cols[~np.isnan(temp_cols)]
        if len(valid_temp) > 0 and np.abs(valid_temp).mean() > 100:
            print(f"    Scaling temperature variables (div 10): mean was {valid_temp.mean():.1f}")
            features[:, :min(11, features.shape[1])] /= 10.0

    return features, hotspot_ids


print("\nLoading hotspot coordinates...")
coords = load_hotspot_coords()
print(f"  Found coordinates for {len(coords)}/{len(all_ids)} hotspots")

# Build future env for each (GCM, SSP) combination
future_env = {}  # key: (gcm, ssp_key) -> env array

if coords:
    print("\nExtracting future bioclim features...")
    for (gcm, ssp_key), info in sorted(discovered.items()):
        print(f"\n  {gcm} / {info['label']} ({info['file'].name}):")
        result = extract_bioclim_from_multiband(info["file"], coords)
        if result is not None:
            feats, hids = result

            # Handle NaN: fill with current values where future is missing
            nan_mask = np.isnan(feats)
            if nan_mask.any():
                n_nan = nan_mask.sum()
                print(f"    Filling {n_nan} NaN values with current bioclim")
                for i, hid in enumerate(hids):
                    if hid in all_idx_map:
                        idx = all_idx_map[hid]
                        for b in range(feats.shape[1]):
                            if np.isnan(feats[i, b]):
                                feats[i, b] = bioclim_current[idx, b]

            # Normalize with current-climate stats
            feats_norm = (feats - bio_mu[:feats.shape[1]]) / bio_std[:feats.shape[1]]

            # Build future env: replace bioclim portion, keep rest
            env_fut = env_current.copy()
            for i, hid in enumerate(hids):
                if hid in all_idx_map:
                    idx = all_idx_map[hid]
                    env_fut[idx, :feats_norm.shape[1]] = feats_norm[i]
            future_env[(gcm, ssp_key)] = env_fut
            print(f"    ✅ Updated env features for {len(hids)} hotspots")
else:
    # Synthetic fallback
    print("  ⚠️ No coordinates found — using synthetic delta approach")
    deltas = {
        "ssp245_2050": {"temp_shift": 2.0, "precip_factor": 1.05},
        "ssp585_2050": {"temp_shift": 3.5, "precip_factor": 1.10},
    }
    for (gcm, ssp_key), info in sorted(discovered.items()):
        d = deltas.get(ssp_key, {"temp_shift": 2.0, "precip_factor": 1.05})
        env_fut = env_current.copy()
        temp_delta_norm = d["temp_shift"] / (bio_std[:min(11, n_bio)] + 1e-8)
        env_fut[:, :min(11, n_bio)] += temp_delta_norm
        for b in range(11, min(19, n_bio)):
            precip_shift = bio_mu[b] * (d["precip_factor"] - 1.0) / bio_std[b]
            env_fut[:, b] = env_fut[:, b] * d["precip_factor"] + precip_shift
        future_env[(gcm, ssp_key)] = env_fut
        print(f"    {gcm} {info['label']}: T+{d['temp_shift']} C, P x{d['precip_factor']}")


# ══════════════════════════════════════════════════════
# 4. Model definition (must match training)
# ══════════════════════════════════════════════════════

class SatelliteEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalSDMv3(nn.Module):
    def __init__(self, n_species, env_dim=33, audio_dim=1024, move_dim=8,
                 sat_hidden=256, env_hidden=64, audio_hidden=128, move_hidden=32,
                 use_satellite=True, use_env=True, use_audio=True, use_movement=True):
        super().__init__()
        self.use_satellite = use_satellite
        self.use_env = use_env
        self.use_audio = use_audio
        self.use_movement = use_movement

        fusion_dim = 0
        if use_satellite:
            self.sat_encoder = SatelliteEncoder(sat_hidden)
            fusion_dim += sat_hidden
        if use_env:
            self.env_encoder = MLPEncoder(env_dim, 128, env_hidden)
            fusion_dim += env_hidden
        if use_audio:
            audio_encoder_hidden = 256 if audio_dim >= 512 else 128
            self.audio_encoder = MLPEncoder(audio_dim, audio_encoder_hidden, audio_hidden)
            fusion_dim += audio_hidden
        if use_movement:
            self.move_encoder = MLPEncoder(move_dim, 64, move_hidden)
            fusion_dim += move_hidden

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_species),
            nn.Sigmoid(),
        )

    def forward(self, image, env, audio, movement):
        parts = []
        if self.use_satellite:
            parts.append(self.sat_encoder(image))
        if self.use_env:
            parts.append(self.env_encoder(env))
        if self.use_audio:
            parts.append(self.audio_encoder(audio))
        if self.use_movement:
            parts.append(self.move_encoder(movement))
        return self.head(torch.cat(parts, dim=1))


# ══════════════════════════════════════════════════════
# 5. Run projections
# ══════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Find best model
model_candidates = [
    ("v2_full", OUTPUT / "model_v2_sat+env+audio+move.pt"),
    ("v2_env+audio", OUTPUT / "model_v2_env+audio.pt"),
    ("v2_sat+env", OUTPUT / "model_v2_sat+env.pt"),
    ("v1_full", OUTPUT / "model_sat+env+audio+move.pt"),
    ("v1_env+audio", OUTPUT / "model_env+audio.pt"),
    ("v1_sat+env", OUTPUT / "model_sat+env.pt"),
]

model_name = None
model_path = None
for name, path in model_candidates:
    if path.exists():
        model_name = name
        model_path = path
        break

if model_path is None:
    print("⚠️ No trained model found — run training first")
    import sys; sys.exit(1)

print(f"Using model: {model_name} ({model_path.name})")

# Detect config from file stem (model_v2_sat+env+audio+move)
_stem = model_path.stem
use_sat = "sat" in _stem or "full" in model_name
use_env = "env" in _stem or "full" in model_name
use_audio = "audio" in _stem or "full" in model_name
use_move = "move" in _stem or "full" in model_name

model = MultimodalSDMv3(
    n_species=n_species, env_dim=env_dim, audio_dim=AUDIO_DIM,
    use_satellite=use_sat, use_env=use_env,
    use_audio=use_audio, use_movement=use_move,
).to(device)

state = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

test_ids = splits["test"]
print(f"Projecting for {len(test_ids)} test hotspots...")

mean_audio = audio_embeddings[audio_mask].mean(axis=0) if audio_mask.any() else np.zeros(AUDIO_DIM)

def predict_batch(env_feats, hotspot_ids):
    preds = []
    with torch.no_grad():
        for hid in hotspot_ids:
            idx = all_idx_map[hid]
            env_t = torch.from_numpy(env_feats[idx:idx+1]).float().to(device)
            img_t = torch.zeros(1, 4, 64, 64).to(device)
            audio_t = torch.from_numpy(mean_audio[None]).float().to(device)
            move_t = torch.zeros(1, 8).to(device)
            pred = model(img_t, env_t, audio_t, move_t)
            preds.append(pred.cpu().numpy()[0])
    return np.array(preds)

print("  Current climate predictions...")
current_preds = predict_batch(env_current, test_ids)

# Per-GCM predictions
gcm_scenario_preds = {}  # (gcm, ssp_key) -> preds array
for (gcm, ssp_key), env_fut in sorted(future_env.items()):
    info = discovered[(gcm, ssp_key)]
    print(f"  {gcm} / {info['label']} predictions...")
    gcm_scenario_preds[(gcm, ssp_key)] = predict_batch(env_fut, test_ids)


# ══════════════════════════════════════════════════════
# 6. Multi-GCM ensemble averaging
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MULTI-GCM ENSEMBLE")
print("=" * 60)

ensemble_preds = {}  # ssp_key -> averaged preds
for ssp_key in ssp_keys:
    gcms_for_ssp = [(gcm, sk) for (gcm, sk) in gcm_scenario_preds if sk == ssp_key]
    if len(gcms_for_ssp) > 1:
        stacked = np.stack([gcm_scenario_preds[k] for k in gcms_for_ssp])
        ensemble_preds[ssp_key] = stacked.mean(axis=0)
        gcm_std = stacked.std(axis=0)
        print(f"  {ssp_key}: ensemble of {len(gcms_for_ssp)} GCMs "
              f"({', '.join(g for g, _ in gcms_for_ssp)}), "
              f"mean inter-GCM std = {gcm_std[:, :n_real].mean():.5f}")
    else:
        # Only one GCM — use it directly
        ensemble_preds[ssp_key] = gcm_scenario_preds[gcms_for_ssp[0]]
        print(f"  {ssp_key}: single GCM ({gcms_for_ssp[0][0]})")


# ══════════════════════════════════════════════════════
# 7. Analysis & results
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CLIMATE PROJECTION RESULTS")
print("=" * 60)

species_names = (species_df["common_name"].values if "common_name" in species_df.columns
                 else species_df.iloc[:, 1].values)

def analyze_scenario(preds_fut, label):
    """Compute richness, gaining/losing species, top affected."""
    richness_current = (current_preds[:, :n_real] > 0.01).sum(axis=1).mean()
    richness_future = (preds_fut[:, :n_real] > 0.01).sum(axis=1).mean()
    mean_rate_current = current_preds[:, :n_real].mean()
    mean_rate_future = preds_fut[:, :n_real].mean()

    delta = preds_fut[:, :n_real].mean(axis=0) - current_preds[:, :n_real].mean(axis=0)
    n_gaining = int((delta > 0.005).sum())
    n_losing = int((delta < -0.005).sum())
    n_stable = n_real - n_gaining - n_losing

    top_losers = np.argsort(delta)[:5]
    top_gainers = np.argsort(delta)[-5:][::-1]

    return {
        "label": label,
        "richness_current": round(float(richness_current), 1),
        "richness_future": round(float(richness_future), 1),
        "richness_delta": round(float(richness_future - richness_current), 1),
        "mean_rate_current": round(float(mean_rate_current), 5),
        "mean_rate_future": round(float(mean_rate_future), 5),
        "n_gaining": n_gaining, "n_losing": n_losing, "n_stable": n_stable,
        "top_gainers": [species_names[i] for i in top_gainers if i < len(species_names)],
        "top_losers": [species_names[i] for i in top_losers if i < len(species_names)],
    }

# Per-GCM results
per_gcm_results = []
for (gcm, ssp_key), preds in sorted(gcm_scenario_preds.items()):
    info = discovered[(gcm, ssp_key)]
    label = f"{gcm} {info['label']}"
    r = analyze_scenario(preds, label)
    r["gcm"] = gcm
    r["ssp_key"] = ssp_key
    r["source"] = "single_gcm"
    per_gcm_results.append(r)

    print(f"\n  {label}:")
    print(f"    Richness: {r['richness_current']:.1f} → {r['richness_future']:.1f} "
          f"(Δ{r['richness_delta']:+.1f})")
    print(f"    Gaining: {r['n_gaining']}  Losing: {r['n_losing']}  Stable: {r['n_stable']}")
    print(f"    Top gainers: {', '.join(r['top_gainers'][:3])}")
    print(f"    Top losers:  {', '.join(r['top_losers'][:3])}")

# Ensemble results
ensemble_results = []
if len(gcm_names) > 1:
    print(f"\n  {'─' * 50}")
    print(f"  ENSEMBLE ({' + '.join(gcm_names)}):")
    for ssp_key in ssp_keys:
        if ssp_key in ensemble_preds:
            ssp_label = {"ssp245_2050": "SSP2-4.5 (2050)", "ssp585_2050": "SSP5-8.5 (2050)"}.get(
                ssp_key, ssp_key)
            label = f"Ensemble {ssp_label}"
            r = analyze_scenario(ensemble_preds[ssp_key], label)
            r["gcm"] = "ensemble"
            r["ssp_key"] = ssp_key
            r["source"] = "ensemble"
            ensemble_results.append(r)

            print(f"\n    {label}:")
            print(f"      Richness: {r['richness_current']:.1f} → {r['richness_future']:.1f} "
                  f"(Δ{r['richness_delta']:+.1f})")
            print(f"      Gaining: {r['n_gaining']}  Losing: {r['n_losing']}  Stable: {r['n_stable']}")

# Inter-GCM agreement
if len(gcm_names) > 1:
    print(f"\n  {'─' * 50}")
    print(f"  INTER-GCM COMPARISON:")
    for ssp_key in ssp_keys:
        gcms_for_ssp = [(gcm, sk) for (gcm, sk) in gcm_scenario_preds if sk == ssp_key]
        if len(gcms_for_ssp) < 2:
            continue
        deltas = {}
        for (gcm, sk) in gcms_for_ssp:
            d = gcm_scenario_preds[(gcm, sk)][:, :n_real].mean(axis=0) - \
                current_preds[:, :n_real].mean(axis=0)
            deltas[gcm] = d

        gcm_list = sorted(deltas.keys())
        # Direction agreement: how many species both GCMs agree on direction
        d1 = deltas[gcm_list[0]]
        d2 = deltas[gcm_list[1]]
        agree_sign = ((d1 > 0.005) & (d2 > 0.005)) | ((d1 < -0.005) & (d2 < -0.005)) | \
                     ((np.abs(d1) <= 0.005) & (np.abs(d2) <= 0.005))
        agreement = agree_sign.mean()

        # Correlation of deltas
        corr = np.corrcoef(d1, d2)[0, 1]

        ssp_label = {"ssp245_2050": "SSP2-4.5", "ssp585_2050": "SSP5-8.5"}.get(ssp_key, ssp_key)
        print(f"\n    {ssp_label}:")
        print(f"      Direction agreement: {agreement:.1%}")
        print(f"      Delta correlation (r): {corr:.3f}")
        print(f"      {gcm_list[0]} richness Δ: "
              f"{per_gcm_results[[i for i,(g,s) in enumerate([(r['gcm'],r['ssp_key']) for r in per_gcm_results]) if g==gcm_list[0] and s==ssp_key][0]]['richness_delta']:+.1f}")
        print(f"      {gcm_list[1]} richness Δ: "
              f"{per_gcm_results[[i for i,(g,s) in enumerate([(r['gcm'],r['ssp_key']) for r in per_gcm_results]) if g==gcm_list[1] and s==ssp_key][0]]['richness_delta']:+.1f}")


all_results = per_gcm_results + ensemble_results

# Save CSV
results_df = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ("top_gainers", "top_losers")}
                            for r in all_results])
results_df.to_csv(OUTPUT / "cmip6_projections.csv", index=False)

# Save per-species deltas
for (gcm, ssp_key), preds_fut in gcm_scenario_preds.items():
    delta = preds_fut[:, :n_real].mean(axis=0) - current_preds[:, :n_real].mean(axis=0)
    np.savez(OUTPUT / f"cmip6_delta_{gcm}_{ssp_key}.npz",
             delta=delta, current=current_preds[:, :n_real].mean(axis=0),
             future=preds_fut[:, :n_real].mean(axis=0))

for ssp_key, preds_ens in ensemble_preds.items():
    delta = preds_ens[:, :n_real].mean(axis=0) - current_preds[:, :n_real].mean(axis=0)
    np.savez(OUTPUT / f"cmip6_delta_ensemble_{ssp_key}.npz",
             delta=delta, current=current_preds[:, :n_real].mean(axis=0),
             future=preds_ens[:, :n_real].mean(axis=0))


# ══════════════════════════════════════════════════════
# 8. Visualization
# ══════════════════════════════════════════════════════
n_gcms = len(gcm_names)
has_ensemble = len(gcm_names) > 1

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

GCM_COLORS = {gcm_names[i]: c for i, c in enumerate(["#3498db", "#e67e22", "#9b59b6", "#1abc9c"][:n_gcms])}
ENS_COLOR = "#2c3e50"

# (a) Richness delta comparison — per GCM + ensemble
ax = fig.add_subplot(gs[0, 0])
bar_labels = []
bar_vals = []
bar_colors = []
for ssp_key in ssp_keys:
    ssp_label = {"ssp245_2050": "SSP2-4.5", "ssp585_2050": "SSP5-8.5"}.get(ssp_key, ssp_key)
    for gcm in gcm_names:
        r = [x for x in per_gcm_results if x["gcm"] == gcm and x["ssp_key"] == ssp_key]
        if r:
            bar_labels.append(f"{gcm[:12]}\n{ssp_label}")
            bar_vals.append(r[0]["richness_delta"])
            bar_colors.append(GCM_COLORS[gcm])
    if has_ensemble:
        r = [x for x in ensemble_results if x["ssp_key"] == ssp_key]
        if r:
            bar_labels.append(f"Ensemble\n{ssp_label}")
            bar_vals.append(r[0]["richness_delta"])
            bar_colors.append(ENS_COLOR)

y_pos = range(len(bar_labels))
face_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in bar_vals]
bars = ax.barh(y_pos, bar_vals, color=face_colors, edgecolor=bar_colors, linewidth=2)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(bar_labels, fontsize=8)
ax.set_xlabel("Δ Species Richness")
ax.set_title("(a) Richness Change by GCM")
ax.axvline(0, color="black", linewidth=0.5)
ax.invert_yaxis()

# (b) Gaining vs Losing — grouped by GCM
ax = fig.add_subplot(gs[0, 1])
x_labels = []
gain_vals = []
lose_vals = []
edge_colors = []
for ssp_key in ssp_keys:
    ssp_label = {"ssp245_2050": "2-4.5", "ssp585_2050": "5-8.5"}.get(ssp_key, ssp_key)
    for gcm in gcm_names:
        r = [x for x in per_gcm_results if x["gcm"] == gcm and x["ssp_key"] == ssp_key]
        if r:
            x_labels.append(f"{gcm[:10]}\n{ssp_label}")
            gain_vals.append(r[0]["n_gaining"])
            lose_vals.append(r[0]["n_losing"])
            edge_colors.append(GCM_COLORS[gcm])
    if has_ensemble:
        r = [x for x in ensemble_results if x["ssp_key"] == ssp_key]
        if r:
            x_labels.append(f"Ens\n{ssp_label}")
            gain_vals.append(r[0]["n_gaining"])
            lose_vals.append(r[0]["n_losing"])
            edge_colors.append(ENS_COLOR)

x = np.arange(len(x_labels))
w = 0.35
ax.bar(x - w/2, gain_vals, w, label="Gaining", color="#2ecc71", alpha=0.8)
ax.bar(x + w/2, lose_vals, w, label="Losing", color="#e74c3c", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=7)
ax.set_ylabel("Species Count")
ax.set_title("(b) Gaining vs Losing Species")
ax.legend(fontsize=8)

# (c) Per-species delta distributions overlaid
ax = fig.add_subplot(gs[0, 2])
for gcm in gcm_names:
    for ssp_key in ssp_keys:
        k = (gcm, ssp_key)
        if k in gcm_scenario_preds:
            delta = gcm_scenario_preds[k][:, :n_real].mean(axis=0) - current_preds[:, :n_real].mean(axis=0)
            ssp_short = {"ssp245_2050": "2-4.5", "ssp585_2050": "5-8.5"}.get(ssp_key, ssp_key)
            ax.hist(delta, bins=50, alpha=0.4, label=f"{gcm[:10]} {ssp_short}",
                    color=GCM_COLORS[gcm],
                    linestyle="--" if "585" in ssp_key else "-",
                    edgecolor=GCM_COLORS[gcm])
ax.axvline(0, color="red", linewidth=1, linestyle="--")
ax.set_xlabel("Δ Mean Encounter Rate")
ax.set_ylabel("Species Count")
ax.set_title("(c) Per-Species Change Distribution")
ax.legend(fontsize=7)

# (d, e) Top affected species — per GCM for worst-case SSP
worst_ssp = "ssp585_2050" if "ssp585_2050" in ssp_keys else ssp_keys[-1]
for gi, gcm in enumerate(gcm_names[:2]):
    ax = fig.add_subplot(gs[1, gi])
    k = (gcm, worst_ssp)
    if k in gcm_scenario_preds:
        delta = gcm_scenario_preds[k][:, :n_real].mean(axis=0) - current_preds[:, :n_real].mean(axis=0)
        top_affected = np.argsort(np.abs(delta))[-15:]
        names = [species_names[i][:25] if i < len(species_names) else f"sp_{i}"
                 for i in top_affected]
        vals = delta[top_affected]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ssp_label = {"ssp245_2050": "SSP2-4.5", "ssp585_2050": "SSP5-8.5"}.get(worst_ssp, worst_ssp)
        ax.set_title(f"({chr(100+gi)}) Top Affected — {gcm}\n{ssp_label}", fontsize=10)
        ax.set_xlabel("Δ Encounter Rate")

# (f) Inter-GCM scatter (if 2+ GCMs)
if has_ensemble and len(gcm_names) >= 2:
    ax = fig.add_subplot(gs[1, 2])
    for ssp_key in ssp_keys:
        k1 = (gcm_names[0], ssp_key)
        k2 = (gcm_names[1], ssp_key)
        if k1 in gcm_scenario_preds and k2 in gcm_scenario_preds:
            d1 = gcm_scenario_preds[k1][:, :n_real].mean(axis=0) - current_preds[:, :n_real].mean(axis=0)
            d2 = gcm_scenario_preds[k2][:, :n_real].mean(axis=0) - current_preds[:, :n_real].mean(axis=0)
            ssp_short = {"ssp245_2050": "SSP2-4.5", "ssp585_2050": "SSP5-8.5"}.get(ssp_key, ssp_key)
            ax.scatter(d1, d2, alpha=0.3, s=8, label=ssp_short)
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel(f"Δ Encounter Rate ({gcm_names[0][:15]})")
    ax.set_ylabel(f"Δ Encounter Rate ({gcm_names[1][:15]})")
    ax.set_title("(f) Inter-GCM Agreement")
    ax.legend(fontsize=8)
else:
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

# (g) Summary table
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
txt = f"CMIP6 Multi-GCM Climate Projections\n"
txt += f"GCMs: {', '.join(gcm_names)}    Model: {model_name}\n"
txt += f"Species: {n_real}    Test hotspots: {len(test_ids)}\n"
txt += f"Coord source: {'raster extraction' if coords else 'synthetic deltas'}\n\n"

# Table header
header = f"{'Source':<22s} {'Scenario':<14s} {'Richness Δ':>12s} {'Gaining':>8s} {'Losing':>8s} {'Stable':>8s}"
txt += header + "\n" + "─" * len(header) + "\n"
for r in all_results:
    src = r["gcm"] if r["source"] == "single_gcm" else "ENSEMBLE"
    ssp_label = {"ssp245_2050": "SSP2-4.5", "ssp585_2050": "SSP5-8.5"}.get(r["ssp_key"], r["ssp_key"])
    txt += (f"{src:<22s} {ssp_label:<14s} {r['richness_delta']:>+12.1f} "
            f"{r['n_gaining']:>8d} {r['n_losing']:>8d} {r['n_stable']:>8d}\n")

ax.text(0.02, 0.95, txt, transform=ax.transAxes,
        fontsize=9, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

fig.suptitle("Species Distribution Under Climate Change — Multi-GCM Comparison", fontsize=14, y=0.99)
plt.savefig(OUTPUT / "cmip6_projections.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT / "fig6_cmip6_projections.pdf", bbox_inches="tight")
print(f"\n✅ Saved: cmip6_projections.png, fig6_cmip6_projections.pdf")

# Full JSON results
full_results = {
    "gcms": gcm_names,
    "model": model_name,
    "n_species": n_real,
    "n_test": len(test_ids),
    "has_coordinates": bool(coords),
    "per_gcm": per_gcm_results,
    "ensemble": ensemble_results,
}
with open(OUTPUT / "cmip6_projections.json", "w") as f:
    json.dump(full_results, f, indent=2, default=str)

print(f"\n→ Done! Check results/cmip6_projections.png")