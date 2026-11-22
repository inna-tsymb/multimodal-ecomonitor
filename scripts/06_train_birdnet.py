#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 6: Training with BirdNET Embeddings (1024-dim)
====================================================
Run: python -I scripts/06_train_birdnet.py

Improvement over 02_train.py:
  - BirdNET pretrained embeddings (1024-dim) instead of mel-spectrogram (128-dim)
  - Larger audio encoder hidden layer (256 instead of 128)
  - Compares results against previous ablation baseline

Requires: working/audio_embeddings_birdnet.npz (from 05a)
"""

import json
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ != "__main__":
    import sys; sys.exit(0)

BASE = Path(__file__).resolve().parent.parent  # scripts/ → project root
FEATURES_DIR = BASE / "working"
DATA = BASE / "data"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ══════════════════════════════════════════════════════
# 1. Load features (BirdNET embeddings)
# ══════════════════════════════════════════════════════
print("\nLoading features...")

with open(FEATURES_DIR / "satbird_splits.json") as f:
    splits = json.load(f)
with open(FEATURES_DIR / "species_metadata.json") as f:
    metadata = json.load(f)

# ── BirdNET embeddings (1024-dim) ──
birdnet_file = FEATURES_DIR / "audio_embeddings_birdnet.npz"
old_audio_file = FEATURES_DIR / "audio_embeddings.npz"

if birdnet_file.exists():
    audio_data = np.load(birdnet_file)
    AUDIO_DIM = 1024
    audio_source = "BirdNET"
    print(f"✅ Using BirdNET embeddings ({AUDIO_DIM}-dim)")
else:
    print(f"⚠️  BirdNET embeddings not found at {birdnet_file}")
    print(f"    Falling back to mel-spectrogram embeddings")
    audio_data = np.load(old_audio_file)
    AUDIO_DIM = 128
    audio_source = "mel-spectrogram"

audio_embeddings = audio_data["embeddings"]
audio_mask = audio_data["mask"]

move_df = pd.read_csv(FEATURES_DIR / "movement_features.csv")
MOVE_COLS = ["mean_step_length", "std_step_length", "mean_speed", "max_speed",
             "mean_turning_angle", "displacement", "straightness", "home_range_km2"]

bioclim = np.load(FEATURES_DIR / "bioclim_features.npy")
landcover = np.load(FEATURES_DIR / "landcover_features.npy")
terrain = np.load(FEATURES_DIR / "terrain_features.npy")
inat = np.load(FEATURES_DIR / "inat_features.npy")

n_species = audio_embeddings.shape[0]
print(f"Species: {n_species}")
print(f"Audio ({audio_source}): {audio_mask.sum():.0f}/{n_species} species, {AUDIO_DIM}-dim")
print(f"Bioclim: {bioclim.shape}, Landcover: {landcover.shape}")
print(f"Terrain: {terrain.shape}, iNat: {inat.shape}")

# ══════════════════════════════════════════════════════
# 2. Prepare features
# ══════════════════════════════════════════════════════

move_features = np.zeros((n_species, len(MOVE_COLS)), dtype=np.float32)
for _, row in move_df.iterrows():
    idx = int(row["species_idx"])
    if idx < n_species:
        move_features[idx] = [row.get(c, 0) for c in MOVE_COLS]


def safe_normalize(arr):
    mu = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return (arr - mu) / std, mu, std


bioclim_norm, _, _ = safe_normalize(bioclim)
landcover_norm = landcover
terrain_norm, _, _ = safe_normalize(terrain)
inat_norm, _, _ = safe_normalize(inat)

env_features = np.concatenate([bioclim_norm, landcover_norm, terrain_norm, inat_norm], axis=1)
env_dim = env_features.shape[1]
print(f"Environmental dim: {env_dim}")

# ══════════════════════════════════════════════════════
# 3. Check satellite + targets
# ══════════════════════════════════════════════════════

sat_image_dir = DATA / "satbird_images"
has_satellite = sat_image_dir.exists() and len(list(sat_image_dir.glob("*.tif"))) > 100

targets_dir = DATA / "satbird_targets"
has_targets = targets_dir.exists() and len(list(targets_dir.glob("*.json"))) > 100

all_ids = splits["train"] + splits["val"] + splits["test"]
hotspot_index_map = {hid: i for i, hid in enumerate(all_ids)}

print(f"Satellite: {'✅ ' + str(len(list(sat_image_dir.glob('*.tif')))) + ' images' if has_satellite else '❌'}")
print(f"Targets: {'✅ ' + str(len(list(targets_dir.glob('*.json')))) + ' files' if has_targets else '❌'}")

if not has_targets:
    print("\n⚠️ No real targets — generating synthetic for pipeline test")
    targets_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    for hid in all_ids:
        rng = np.random.RandomState(int(abs(hash(hid)) % 10000))
        rates = rng.exponential(0.01, n_species).astype(np.float32)
        rates = np.clip(rates, 0, 1)
        rates[rng.random(n_species) > 0.05] = 0
        with open(targets_dir / f"{hid}.json", "w") as f:
            json.dump({"encounter_rates": rates.tolist()}, f)
    print(f"  Generated {len(all_ids)} synthetic targets")

# ══════════════════════════════════════════════════════
# 4. Dataset
# ══════════════════════════════════════════════════════


class MultimodalSDMDataset(Dataset):
    def __init__(self, hotspot_ids, env_features_all, audio_emb, audio_mask,
                 move_feat, hotspot_index_map, targets_dir,
                 sat_dir=None, img_size=64):
        self.ids = hotspot_ids
        self.env_all = env_features_all
        self.audio_emb = audio_emb
        self.audio_mask = audio_mask
        self.move_feat = move_feat
        self.idx_map = hotspot_index_map
        self.targets_dir = Path(targets_dir)
        self.sat_dir = Path(sat_dir) if sat_dir else None
        self.img_size = img_size
        self.has_satellite = sat_dir is not None and Path(sat_dir).exists()

        self.sat_mean = np.array([1353, 1116, 1041, 945], dtype=np.float32).reshape(4, 1, 1)
        self.sat_std = np.array([245, 357, 593, 672], dtype=np.float32).reshape(4, 1, 1)

        self._targets = {}
        for hid in self.ids:
            tp = self.targets_dir / f"{hid}.json"
            with open(tp) as f:
                tdata = json.load(f)
            rates = np.array(
                tdata.get("encounter_rates", tdata.get("probs", tdata.get("rates", []))),
                dtype=np.float32)
            self._targets[hid] = rates

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        hid = self.ids[idx]
        row_idx = self.idx_map[hid]

        # Satellite
        if self.has_satellite:
            import rasterio
            img_path = self.sat_dir / f"{hid}.tif"
            if img_path.exists():
                with rasterio.open(img_path) as src:
                    img = src.read([1, 2, 3, 4]).astype(np.float32)
                img = (img - self.sat_mean) / self.sat_std
                if img.shape[1] != self.img_size or img.shape[2] != self.img_size:
                    from skimage.transform import resize
                    img = resize(img, (4, self.img_size, self.img_size),
                                 preserve_range=True).astype(np.float32)
            else:
                img = np.zeros((4, self.img_size, self.img_size), dtype=np.float32)
        else:
            img = np.zeros((4, self.img_size, self.img_size), dtype=np.float32)

        rates = self._targets[hid]
        env = self.env_all[row_idx].astype(np.float32)

        # Audio (encounter-rate weighted)
        weights = rates.copy()
        weights[~self.audio_mask[:len(weights)].astype(bool)] = 0
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
            audio_vec = (weights[:, None] * self.audio_emb[:len(weights)]).sum(axis=0)
        else:
            audio_vec = np.zeros(self.audio_emb.shape[1], dtype=np.float32)

        # Movement (encounter-rate weighted)
        move_mask = (self.move_feat[:len(rates)].sum(axis=1) != 0)
        mw = rates.copy()
        mw[~move_mask[:len(mw)]] = 0
        mw_sum = mw.sum()
        if mw_sum > 0:
            mw /= mw_sum
            move_vec = (mw[:, None] * self.move_feat[:len(mw)]).sum(axis=0)
        else:
            move_vec = np.zeros(self.move_feat.shape[1], dtype=np.float32)

        return {
            "image": torch.from_numpy(img),
            "env": torch.from_numpy(env),
            "audio": torch.from_numpy(audio_vec),
            "movement": torch.from_numpy(move_vec),
            "target": torch.from_numpy(rates),
        }


# ══════════════════════════════════════════════════════
# 5. Model (scaled for BirdNET)
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
    """v3: scaled audio encoder for BirdNET 1024-dim embeddings."""
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
            # Larger hidden for 1024-dim input (was 128 for mel-128)
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

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════
# 6. Training
# ══════════════════════════════════════════════════════

def top_k_accuracy(preds, targets, k=30):
    scores = []
    for p, t in zip(preds, targets):
        pred_topk = set(np.argsort(p)[-k:])
        true_topk = set(np.argsort(t)[-k:])
        if len(true_topk) > 0:
            scores.append(len(pred_topk & true_topk) / k)
    return np.mean(scores) if scores else 0.0


def run_experiment(name, config, splits, n_epochs=50, lr=1e-4, patience=10):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    use_sat = has_satellite and config.get("use_satellite", False)
    sat_dir = str(sat_image_dir) if use_sat else None
    batch_size = 64 if use_sat else 128
    n_workers = 8

    train_ds = MultimodalSDMDataset(
        splits["train"], env_features, audio_embeddings, audio_mask,
        move_features, hotspot_index_map, targets_dir, sat_dir)
    val_ds = MultimodalSDMDataset(
        splits["val"], env_features, audio_embeddings, audio_mask,
        move_features, hotspot_index_map, targets_dir, sat_dir)
    test_ds = MultimodalSDMDataset(
        splits["test"], env_features, audio_embeddings, audio_mask,
        move_features, hotspot_index_map, targets_dir, sat_dir)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=n_workers, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                        num_workers=n_workers, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         num_workers=n_workers, pin_memory=True, persistent_workers=True)

    model = MultimodalSDMv3(
        n_species=n_species, env_dim=env_dim, audio_dim=AUDIO_DIM, **config
    ).to(device)
    print(f"Params: {model.count_params():,}  |  Batch: {batch_size}  |  Audio: {AUDIO_DIM}-dim")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    use_amp = use_sat and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    history = []
    best_val_mse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0
        for batch in train_dl:
            img = batch["image"].to(device, non_blocking=True)
            env = batch["env"].to(device, non_blocking=True)
            aud = batch["audio"].to(device, non_blocking=True)
            mov = batch["movement"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast("cuda"):
                    pred = model(img, env, aud, mov)
                    loss = criterion(pred, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(img, env, aud, mov)
                loss = criterion(pred, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item() * img.size(0)

        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(device, non_blocking=True)
                env = batch["env"].to(device, non_blocking=True)
                aud = batch["audio"].to(device, non_blocking=True)
                mov = batch["movement"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                if use_amp:
                    with autocast("cuda"):
                        pred = model(img, env, aud, mov)
                        loss = criterion(pred, target)
                else:
                    pred = model(img, env, aud, mov)
                    loss = criterion(pred, target)

                val_loss += loss.item() * img.size(0)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(target.cpu().numpy())

        val_loss /= len(val_ds)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_top30 = top_k_accuracy(val_preds, val_targets, k=30)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_mse": round(train_loss, 6),
            "val_mse": round(val_loss, 6),
            "val_top30": round(val_top30, 4),
        })

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | train {train_loss:.5f} | "
                  f"val {val_loss:.5f} | top30 {val_top30:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_loss = 0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_dl:
            img = batch["image"].to(device, non_blocking=True)
            env = batch["env"].to(device, non_blocking=True)
            aud = batch["audio"].to(device, non_blocking=True)
            mov = batch["movement"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            if use_amp:
                with autocast("cuda"):
                    pred = model(img, env, aud, mov)
                    loss = criterion(pred, target)
            else:
                pred = model(img, env, aud, mov)
                loss = criterion(pred, target)

            test_loss += loss.item() * img.size(0)
            test_preds.append(pred.cpu().numpy())
            test_targets.append(target.cpu().numpy())

    test_loss /= len(test_ds)
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_top30 = top_k_accuracy(test_preds, test_targets, k=30)

    print(f"  TEST — MSE: {test_loss:.5f}, Top-30: {test_top30:.4f}")

    # Save with v2 suffix to distinguish from 02_train results
    torch.save(best_state, OUTPUT / f"model_v2_{name}.pt")
    np.savez(OUTPUT / f"preds_v2_{name}.npz",
             test_preds=test_preds, test_targets=test_targets)

    return {
        "name": name, "config": config, "n_params": model.count_params(),
        "test_mse": round(test_loss, 6), "test_top30": round(test_top30, 4),
        "best_epoch": len(history) - wait, "history": history,
    }


# ══════════════════════════════════════════════════════
# 7. Ablation Study (same configs as 02_train)
# ══════════════════════════════════════════════════════

if has_satellite:
    ablation_configs = OrderedDict([
        ("satellite_only",      {"use_satellite": True,  "use_env": False, "use_audio": False, "use_movement": False}),
        ("env_only",            {"use_satellite": False, "use_env": True,  "use_audio": False, "use_movement": False}),
        ("audio_only",          {"use_satellite": False, "use_env": False, "use_audio": True,  "use_movement": False}),
        ("sat+env",             {"use_satellite": True,  "use_env": True,  "use_audio": False, "use_movement": False}),
        ("env+audio",           {"use_satellite": False, "use_env": True,  "use_audio": True,  "use_movement": False}),
        ("env+move",            {"use_satellite": False, "use_env": True,  "use_audio": False, "use_movement": True}),
        ("sat+env+audio+move",  {"use_satellite": True,  "use_env": True,  "use_audio": True,  "use_movement": True}),
    ])
else:
    ablation_configs = OrderedDict([
        ("env_only",            {"use_satellite": False, "use_env": True,  "use_audio": False, "use_movement": False}),
        ("audio_only",          {"use_satellite": False, "use_env": False, "use_audio": True,  "use_movement": False}),
        ("env+audio",           {"use_satellite": False, "use_env": True,  "use_audio": True,  "use_movement": False}),
        ("env+move",            {"use_satellite": False, "use_env": True,  "use_audio": False, "use_movement": True}),
        ("env+audio+move",      {"use_satellite": False, "use_env": True,  "use_audio": True,  "use_movement": True}),
    ])

print(f"\nAblation: {len(ablation_configs)} configs")
print(f"Satellite: {has_satellite}")
print(f"Audio source: {audio_source} ({AUDIO_DIM}-dim)")

all_results = []
total_t0 = time.time()

for name, config in ablation_configs.items():
    try:
        result = run_experiment(name, config, splits, n_epochs=50, patience=10)
        all_results.append(result)
    except Exception as e:
        print(f"⚠️ {name} failed: {e}")
        import traceback; traceback.print_exc()

total_time = time.time() - total_t0

# ══════════════════════════════════════════════════════
# 8. Results + comparison with baseline
# ══════════════════════════════════════════════════════

results_df = pd.DataFrame([{
    "name": r["name"], "n_params": r["n_params"],
    "test_mse": r["test_mse"], "test_top30": r["test_top30"],
    "best_epoch": r["best_epoch"],
} for r in all_results]).sort_values("test_mse")

print("\n" + "=" * 70)
print(f"ABLATION RESULTS — {audio_source} {AUDIO_DIM}-dim (sorted by MSE)")
print("=" * 70)
print(results_df.to_string(index=False))
print(f"\nTotal training time: {total_time/60:.1f} min")

results_df.to_csv(OUTPUT / "ablation_results_v2.csv", index=False)
with open(OUTPUT / "ablation_full_v2.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# ── Compare with baseline (02_train) ──
baseline_csv = OUTPUT / "ablation_results.csv"
if baseline_csv.exists():
    baseline_df = pd.read_csv(baseline_csv)
    print("\n" + "=" * 70)
    print("COMPARISON: mel-spectrogram (v1) vs BirdNET (v2)")
    print("=" * 70)
    print(f"{'Config':<25} {'v1 MSE':>10} {'v2 MSE':>10} {'Δ MSE':>10} "
          f"{'v1 Top30':>10} {'v2 Top30':>10} {'Δ Top30':>10}")
    print("-" * 85)

    for _, row_v2 in results_df.iterrows():
        name = row_v2["name"]
        match = baseline_df[baseline_df["name"] == name]
        if len(match) > 0:
            v1_mse = match.iloc[0]["test_mse"]
            v1_top30 = match.iloc[0]["test_top30"]
            d_mse = row_v2["test_mse"] - v1_mse
            d_top30 = row_v2["test_top30"] - v1_top30
            marker_mse = "✅" if d_mse < 0 else "❌"
            marker_top30 = "✅" if d_top30 > 0 else "❌"
            print(f"{name:<25} {v1_mse:>10.5f} {row_v2['test_mse']:>10.5f} "
                  f"{d_mse:>+10.5f}{marker_mse} "
                  f"{v1_top30:>10.4f} {row_v2['test_top30']:>10.4f} "
                  f"{d_top30:>+10.4f}{marker_top30}")
        else:
            print(f"{name:<25} {'—':>10} {row_v2['test_mse']:>10.5f} "
                  f"{'':>11} {'—':>10} {row_v2['test_top30']:>10.4f}")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))

ax = axes[0]
bars = ax.barh(range(len(results_df)), results_df["test_mse"], color=colors)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df["name"], fontsize=9)
ax.set_xlabel("Test MSE ↓")
ax.set_title(f"(a) Test MSE — {audio_source}")
ax.invert_yaxis()
for bar, val in zip(bars, results_df["test_mse"]):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
            f" {val:.5f}", va="center", fontsize=8)

ax = axes[1]
bars = ax.barh(range(len(results_df)), results_df["test_top30"], color=colors)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df["name"], fontsize=9)
ax.set_xlabel("Top-30 Recall ↑")
ax.set_title(f"(b) Top-30 Recall — {audio_source}")
ax.invert_yaxis()

ax = axes[2]
for res in all_results:
    hist = pd.DataFrame(res["history"])
    ax.plot(hist["epoch"], hist["val_mse"], label=res["name"], linewidth=1.5)
ax.set_xlabel("Epoch"); ax.set_ylabel("Val MSE")
ax.set_title("(c) Learning Curves")
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT / "ablation_comparison_v2.png", dpi=200, bbox_inches="tight")

print(f"\n✅ Results saved to {OUTPUT}")
print(f"   ablation_results_v2.csv, ablation_full_v2.json, ablation_comparison_v2.png")
print("→ Next: python -I scripts/03_analysis.py (update to compare v1 vs v2)")
