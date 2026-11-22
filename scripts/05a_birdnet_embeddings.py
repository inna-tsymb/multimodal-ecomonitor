#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 5a: BirdNET Pretrained Embeddings
=======================================
Replace mel-spectrogram (128-dim) with BirdNET embeddings (1024-dim).

Install: uv pip install birdnet
Run:     python -I scripts/05a_birdnet_embeddings.py

Reads audio files from:
  - data/birdclef/train_audio/<ebird_code>/*.ogg
  - data/birdclef2024/train_audio/<ebird_code>/*.ogg  (if exists)
  - data/birdclef2025/train_audio/<ebird_code>/*.ogg  (if exists)
  - data/xenocanto/<ebird_code>/*.mp3

Outputs: working/audio_embeddings_birdnet.npz
         (keys: "embeddings" shape [n_species, 1024], "mask" shape [n_species])
"""

import sys
import json
import time
import signal
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
WORKING = BASE / "working"
WORKING.mkdir(exist_ok=True)

CHECKPOINT = WORKING / "birdnet_embed_checkpoint.npz"
OUTPUT = WORKING / "audio_embeddings_birdnet.npz"
EMBED_DIM = 1024
MAX_FILES_PER_SOURCE = 5   # max audio files per species per source
MAX_FILES_TOTAL = 10       # max total audio files per species

if __name__ != "__main__":
    import sys
    sys.exit(0)
    
# ── Load species list ──
species_csv = DATA / "species_list_kenya.csv"
if not species_csv.exists():
    # Try alternative location
    for p in [DATA / "batis" / "species_list_kenya.csv",
              WORKING / "species_list_kenya.csv"]:
        if p.exists():
            species_csv = p
            break

species_df = pd.read_csv(species_csv)
n_real_species = len(species_df)
# The model uses 1054-dim output (755 real + 299 padding), match that
N_TOTAL = 1054
print(f"Loaded {n_real_species} real species from {species_csv.name} (padded to {N_TOTAL})")

# Ensure we have ebird_code and scientific_name
assert "ebird_code" in species_df.columns, "Need ebird_code column"
assert "scientific_name" in species_df.columns, "Need scientific_name column"

# ── Load BirdNET model ──
print("Loading BirdNET model...")
try:
    import birdnet
    # Try ONNX backend first (no TF needed), then TF
    try:
        model = birdnet.load("acoustic", "2.4", "tf", library="litert")
        print("  Loaded BirdNET v2.4 (LiteRT backend)")
    except Exception:
        try:
            model = birdnet.load("acoustic", "2.4", "tf")
            print("  Loaded BirdNET v2.4 (TF backend)")
        except Exception:
            model = birdnet.load("acoustic", "3.0", "onnx")
            print("  Loaded BirdNET v3.0 (ONNX backend)")
except ImportError:
    print("ERROR: birdnet not installed. Run: pip install birdnet")
    print("  Or for TF backend: pip install 'birdnet[tf]'")
    sys.exit(1)

# ── Check if model supports embeddings ──
# The birdnet library's predict() returns predictions; for embeddings
# we need to use the model's internal embedding extraction
# BirdNET v2.4 produces 1024-dim embeddings from 3-second chunks at 48kHz

def extract_embedding(audio_path, timeout_sec=60):
    """Extract BirdNET embedding from an audio file using model.encode().
    Returns mean embedding across all 3-second chunks, or None on failure."""

    # Timeout handler for hung files
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timeout processing {audio_path}")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)

    try:
        # model.encode() returns AcousticFileEncodingResult
        # .embeddings shape: (1, n_chunks, 1024)
        result = model.encode(str(audio_path))
        emb = result.embeddings  # (1, n_chunks, 1024)
        if emb is None or emb.size == 0:
            return None
        # Mean over chunks → (1024,)
        return emb[0].mean(axis=0).astype(np.float32)

    except TimeoutError:
        print(f"    ⏱ Timeout: {audio_path.name}")
        return None
    except Exception as e:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Find audio files for each species ──
def find_audio_files(ebird_code, sci_name):
    """Find audio files from BirdCLEF and Xeno-Canto directories."""
    files = []

    # BirdCLEF directories (actual folder names on Mordor)
    for dirname in ["birdclef", "birdclef2024", "birdclef2025"]:
        bclef_dir = DATA / dirname / "train_audio" / ebird_code
        if bclef_dir.exists():
            audio_files = sorted(bclef_dir.glob("*.*"))[:MAX_FILES_PER_SOURCE]
            files.extend(audio_files)

    # Xeno-Canto directory (uses ebird_code)
    xc_dir = DATA / "xenocanto" / ebird_code
    if xc_dir.exists():
        xc_files = sorted(xc_dir.glob("*.*"))[:MAX_FILES_PER_SOURCE]
        files.extend(xc_files)

    return files[:MAX_FILES_TOTAL]


# ── Load checkpoint if exists ──
all_embeddings = np.zeros((N_TOTAL, EMBED_DIM), dtype=np.float32)
mask = np.zeros(N_TOTAL, dtype=np.float32)
start_idx = 0

if CHECKPOINT.exists():
    ckpt = np.load(CHECKPOINT)
    all_embeddings = ckpt["embeddings"]
    mask = ckpt["mask"]
    start_idx = int(ckpt.get("last_idx", 0)) + 1
    print(f"Resuming from checkpoint: {start_idx}/{n_real_species} "
          f"({mask.sum():.0f} species done)")

# ── Process species (only real 755, padding stays zero) ──
print(f"\nExtracting BirdNET embeddings for {n_real_species} species...")
t0 = time.time()
n_found = int(mask.sum())
n_errors = 0

for i in range(start_idx, n_real_species):
    row = species_df.iloc[i]
    ebird_code = str(row["ebird_code"])
    sci_name = str(row["scientific_name"])

    audio_files = find_audio_files(ebird_code, sci_name)

    if not audio_files:
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{n_real_species}] {sci_name}: no audio files "
                  f"({n_found} done, {elapsed:.0f}s)")
        continue

    # Extract embeddings from all available files
    file_embeddings = []
    for af in audio_files:
        emb = extract_embedding(af)
        if emb is not None and len(emb) == EMBED_DIM:
            file_embeddings.append(emb)

    if file_embeddings:
        all_embeddings[i] = np.mean(file_embeddings, axis=0)
        mask[i] = 1.0
        n_found += 1

    # Progress
    if (i + 1) % 50 == 0 or i == n_real_species - 1:
        elapsed = time.time() - t0
        rate = (i - start_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (n_real_species - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{n_real_species}] {n_found} species with embeddings, "
              f"{n_errors} errors, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    # Checkpoint every 50 species
    if (i + 1) % 50 == 0:
        np.savez(CHECKPOINT,
                 embeddings=all_embeddings, mask=mask,
                 last_idx=i)

# ── Save final output ──
np.savez(OUTPUT,
         embeddings=all_embeddings,
         mask=mask)

elapsed = time.time() - t0
print(f"\n✅ BirdNET embeddings saved to {OUTPUT}")
print(f"   {n_found}/{n_real_species} species with embeddings ({EMBED_DIM}-dim)")
print(f"   Time: {elapsed:.1f}s")

# ── Compare with old mel embeddings ──
old_file = WORKING / "audio_embeddings.npz"
if old_file.exists():
    old = np.load(old_file)
    old_mask = old["mask"]
    old_dim = old["embeddings"].shape[1] if old["embeddings"].ndim == 2 else 128
    print(f"\n   Old mel embeddings: {old_mask.sum():.0f}/{len(old_mask)} species, {old_dim}-dim")
    print(f"   New BirdNET:        {mask.sum():.0f}/{len(mask)} species, {EMBED_DIM}-dim")
    # Handle different array lengths
    min_len = min(len(old_mask), len(mask))
    overlap = ((old_mask[:min_len] > 0) & (mask[:min_len] > 0)).sum()
    new_only = (mask[:min_len] > 0).sum() - overlap
    print(f"   Overlap:            {overlap} species")
    if new_only > 0:
        print(f"   New coverage:       +{new_only} species (BirdNET only)")

print(f"\n→ Next: update 02_train.py to use audio_embeddings_birdnet.npz (1024-dim)")
print(f"  Change: audio_input_dim from 128 to {EMBED_DIM}")
print(f"  Change: FEATURES_DIR / 'audio_embeddings.npz' → 'audio_embeddings_birdnet.npz'")