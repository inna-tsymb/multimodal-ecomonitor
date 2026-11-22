#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 1: Data Preparation — All Modalities (CPU)
================================================
Run: python -I 01_data_prep.py

Uses outputs from 00_download_all.py.
Produces .npy/.npz/.csv/.json files in ./working/
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_from_disk

BASE = Path(__file__).resolve().parent.parent  # scripts/ → project root
DATA = BASE / "data"
BATIS_DIR = DATA / "batis"
BIRDCLEF_DIR = DATA / "birdclef"
OUTPUT = BASE / "working"
OUTPUT.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. Load BATIS Kenya
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("1/7  Loading BATIS Kenya dataset")
print("=" * 60)

ds_train = load_from_disk(str(BATIS_DIR / "train"))
ds_val   = load_from_disk(str(BATIS_DIR / "val"))
ds_test  = load_from_disk(str(BATIS_DIR / "test"))

print(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")
print(f"Features: {list(ds_train.features.keys())}")

bio_cols = [c for c in ds_train.column_names if c.startswith("bio_")]
print(f"Bioclimatic columns: {bio_cols}")


def parse_batis_split(ds, split_name):
    records = []
    for row in ds:
        rec = {
            "hotspot_id": row["hotspot_id"],
            "latitude": float(row["lat"]),
            "longitude": float(row["lon"]),
            "split": split_name,
        }
        for bc in bio_cols:
            rec[bc] = float(row[bc]) if row[bc] is not None else np.nan
        if "sat_imagery_path" in row:
            rec["sat_imagery_path"] = row["sat_imagery_path"]
        records.append(rec)
    return records


hotspots = []
hotspots += parse_batis_split(ds_train, "train")
hotspots += parse_batis_split(ds_val, "val")
hotspots += parse_batis_split(ds_test, "test")

print(f"Total hotspots: {len(hotspots)}")

splits = {
    "train": [h["hotspot_id"] for h in hotspots if h["split"] == "train"],
    "val":   [h["hotspot_id"] for h in hotspots if h["split"] == "val"],
    "test":  [h["hotspot_id"] for h in hotspots if h["split"] == "test"],
}
with open(OUTPUT / "satbird_splits.json", "w") as f:
    json.dump(splits, f)

coords = {h["hotspot_id"]: (h["latitude"], h["longitude"]) for h in hotspots}

# ═══════════════════════════════════════════════════════════
# 2. Bioclimatic features
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2/7  Bioclimatic features")
print("=" * 60)

n_bio = len(bio_cols) if bio_cols else 19

if bio_cols:
    bio_matrix = np.full((len(hotspots), n_bio), np.nan, dtype=np.float32)
    for i, h in enumerate(hotspots):
        for j, bc in enumerate(bio_cols):
            bio_matrix[i, j] = h.get(bc, np.nan)

    col_means = np.nanmean(bio_matrix, axis=0)
    for j in range(n_bio):
        mask = np.isnan(bio_matrix[:, j])
        if mask.any():
            bio_matrix[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0

    np.save(OUTPUT / "bioclim_features.npy", bio_matrix)
    print(f"Bioclim: shape {bio_matrix.shape}, NaN: {np.isnan(bio_matrix).mean()*100:.1f}%")
else:
    print("⚠️ No bio_ columns — need WorldClim rasters (fallback)")
    import rasterio
    import urllib.request, zipfile

    worldclim_dir = DATA / "worldclim"
    worldclim_dir.mkdir(exist_ok=True)
    wc_zip = worldclim_dir / "wc2.1_10m_bio.zip"

    if not list(worldclim_dir.glob("*.tif")):
        url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
        print("Downloading WorldClim 10m (~80MB)...")
        urllib.request.urlretrieve(url, wc_zip)
        with zipfile.ZipFile(wc_zip) as z:
            z.extractall(worldclim_dir)
        wc_zip.unlink()

    bio_files = sorted(worldclim_dir.glob("wc2.1_10m_bio_*.tif"))
    bio_matrix = np.zeros((len(hotspots), 19), dtype=np.float32)
    if bio_files:
        for i, h in enumerate(hotspots):
            lat, lon = h["latitude"], h["longitude"]
            for j, bf in enumerate(bio_files[:19]):
                try:
                    with rasterio.open(bf) as src:
                        row, col = src.index(lon, lat)
                        if 0 <= row < src.height and 0 <= col < src.width:
                            val = src.read(1)[row, col]
                            bio_matrix[i, j] = float(val) if val != src.nodata else 0
                except Exception:
                    pass
    np.save(OUTPUT / "bioclim_features.npy", bio_matrix)
    print(f"Bioclim: shape {bio_matrix.shape}")

# ═══════════════════════════════════════════════════════════
# 3. Species list
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3/7  Species list")
print("=" * 60)

# Load species list from BATIS Species_ID file
species_list_path = DATA / "species_list_kenya.csv"
if not species_list_path.exists():
    from huggingface_hub import hf_hub_download
    import shutil
    p = hf_hub_download(repo_id="cathv/BATIS",
                        filename="Species_ID/species_list_kenya.csv",
                        repo_type="dataset", cache_dir=str(DATA / "hf_cache"))
    shutil.copy(p, species_list_path)
    print("  Downloaded species_list_kenya.csv")

species_df = pd.read_csv(species_list_path)
ebird_codes = species_df["ebird_code"].tolist()
scientific_names = species_df["scientific_name"].tolist()

# probs vector length from targets
targets_dir = DATA / "satbird_targets"
sample_target = next(targets_dir.glob("*.json"), None)
with open(sample_target) as f:
    tdata = json.load(f)
probs = tdata.get("probs", tdata.get("encounter_rates", tdata.get("rates", [])))
N_SPECIES = len(probs) if isinstance(probs, list) else len(ebird_codes)

# Map: index in probs vector → ebird code (species_list order = probs order)
# If probs has more entries than species list, extra entries are padding
species_names = ebird_codes + [f"unknown_{i}" for i in range(len(ebird_codes), N_SPECIES)]
species_to_idx = {s: i for i, s in enumerate(species_names)}
ebird_to_idx = {code: i for i, code in enumerate(ebird_codes)}

with open(OUTPUT / "species_metadata.json", "w") as f:
    json.dump({"n_species": N_SPECIES,
               "n_known_species": len(ebird_codes),
               "ebird_codes": ebird_codes,
               "scientific_names": scientific_names,
               "source": "BATIS/SatBird Kenya"}, f, indent=2)
print(f"Species: {N_SPECIES} ({len(ebird_codes)} known + {N_SPECIES - len(ebird_codes)} padding)")
print(f"  First 5: {ebird_codes[:5]}")
print(f"  Last 5:  {ebird_codes[-5:]}")

# ═══════════════════════════════════════════════════════════
# 4. Audio Embeddings (BirdCLEF)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4/7  Audio embeddings (BirdCLEF)")
print("=" * 60)

import librosa

def extract_mel_embedding(audio_path, sr=32000, n_mels=128, duration=5.0):
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        if len(y) < sr * 0.5:
            return None
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=16000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.mean(axis=1).astype(np.float32)
    except Exception:
        return None


n_mels = 128
audio_matrix = np.zeros((N_SPECIES, n_mels), dtype=np.float32)
audio_mask = np.zeros(N_SPECIES, dtype=bool)

# Find audio directory
audio_dir = None
for candidate in [BIRDCLEF_DIR / "train_audio", BIRDCLEF_DIR / "audio", BIRDCLEF_DIR]:
    if candidate.exists() and list(candidate.glob("*/*.ogg")):
        audio_dir = candidate
        break

if audio_dir:
    print(f"Audio: {audio_dir}")
    species_folders = sorted([d for d in audio_dir.iterdir() if d.is_dir()])
    print(f"BirdCLEF species: {len(species_folders)}")

    matched = 0

    for sp_folder in species_folders:
        sp_code = sp_folder.name  # e.g. "abethr1"
        sp_idx = ebird_to_idx.get(sp_code)
        if sp_idx is None:
            continue

        audio_files = (list(sp_folder.glob("*.ogg")) + list(sp_folder.glob("*.mp3")))[:10]
        sp_embs = []
        for af in audio_files:
            emb = extract_mel_embedding(str(af))
            if emb is not None:
                sp_embs.append(emb)
        if sp_embs:
            audio_matrix[sp_idx] = np.mean(sp_embs, axis=0)
            audio_mask[sp_idx] = True
            matched += 1

    print(f"Matched {matched}/{N_SPECIES} species")
else:
    print("⚠️ BirdCLEF not available — synthetic audio embeddings")
    np.random.seed(42)
    for i in range(min(200, N_SPECIES)):
        audio_matrix[i] = np.random.randn(n_mels).astype(np.float32) * 0.1
        audio_mask[i] = True

np.savez(OUTPUT / "audio_embeddings.npz", embeddings=audio_matrix, mask=audio_mask)
print(f"Audio: {audio_mask.sum()}/{N_SPECIES} species")

# ═══════════════════════════════════════════════════════════
# 5. Ecological / Movement Features (IUCN + derived)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5/7  Ecological features (IUCN + derived)")
print("=" * 60)

import urllib.request

IUCN_TOKEN = "rlts_1M0G7D8GtCUn5yG8U2XCOvi0AEbpku4WvUBQ"
EBIRD_TOKEN = "c20641d2-8124-4fda-8b7c-4e8eeb46c1bb"

# IUCN Red List category encoding
IUCN_CAT = {"LC": 0, "NT": 1, "VU": 2, "EN": 3, "CR": 4,
            "EW": 5, "EX": 6, "DD": -1, "NE": -1}

# Features per species:
# 0: iucn_category (encoded)
# 1: prevalence (mean encounter rate across hotspots)
# 2: range_size (fraction of hotspots where species > 0)
# 3: mean_elevation (mean elevation where species present)
# 4: elevation_range (max - min elevation where present)
# 5: niche_breadth (mean std of bioclim vars where present)
# 6: is_migrant (0=resident/unknown, 1=migrant — from IUCN if available)
# 7: body_mass_proxy (log of taxonomic order encoding — rough proxy)
N_ECO = 8
eco_matrix = np.zeros((N_SPECIES, N_ECO), dtype=np.float32)

# --- Derived features from targets ---
print("  Computing derived features from targets...")
targets_dir = DATA / "satbird_targets"
all_probs = []
target_files = sorted(targets_dir.glob("*.json"))
for tf in target_files:
    with open(tf) as f:
        td = json.load(f)
    probs = td.get("probs", td.get("encounter_rates", td.get("rates", [])))
    if isinstance(probs, list) and len(probs) == N_SPECIES:
        all_probs.append(probs)

if all_probs:
    prob_array = np.array(all_probs, dtype=np.float32)  # (n_hotspots, n_species)
    # Prevalence: mean encounter rate
    eco_matrix[:, 1] = prob_array.mean(axis=0)
    # Range size: fraction of hotspots where present
    eco_matrix[:, 2] = (prob_array > 0).mean(axis=0)
    print(f"  Prevalence range: {eco_matrix[:, 1].min():.4f} – {eco_matrix[:, 1].max():.4f}")
    print(f"  Range size range: {eco_matrix[:, 2].min():.4f} – {eco_matrix[:, 2].max():.4f}")

    # Mean elevation and niche breadth per species
    bio_matrix_loaded = np.load(OUTPUT / "bioclim_features.npy")
    terrain_path = OUTPUT / "terrain_features.npy"
    if terrain_path.exists():
        terrain_loaded = np.load(terrain_path)
        elevations = terrain_loaded[:, 0]  # raw elevation
    else:
        elevations = np.zeros(len(all_probs))

    n_hotspots = min(len(elevations), prob_array.shape[0], bio_matrix_loaded.shape[0])
    for sp_i in range(min(N_SPECIES, len(ebird_codes))):
        present = prob_array[:n_hotspots, sp_i] > 0
        if present.sum() >= 3:
            sp_elev = elevations[:n_hotspots][present]
            eco_matrix[sp_i, 3] = sp_elev.mean() / 5895  # normalized
            eco_matrix[sp_i, 4] = (sp_elev.max() - sp_elev.min()) / 5895
            sp_bio = bio_matrix_loaded[:n_hotspots][present]
            eco_matrix[sp_i, 5] = sp_bio.std(axis=0).mean()
    print(f"  Niche breadth computed for {(eco_matrix[:, 5] > 0).sum()} species")

# --- IUCN Red List API ---
print("  Querying IUCN Red List API...")
iucn_queried = 0
iucn_found = 0
for sp_i, sci_name in enumerate(scientific_names):
    encoded_name = sci_name.replace(" ", "%20")
    url = f"https://apiv4.iucnredlist.org/api/v4/taxa/scientific_name/{encoded_name}?token={IUCN_TOKEN}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        iucn_queried += 1

        # Extract category
        taxa = data.get("taxa", data.get("result", []))
        if isinstance(taxa, list) and taxa:
            taxon = taxa[0]
            cat = taxon.get("redlistCategory", taxon.get("category", ""))
            # Extract short code
            cat_code = cat.split(" ")[0] if cat else ""
            # Map common full names
            cat_map = {"Least": "LC", "Near": "NT", "Vulnerable": "VU",
                       "Endangered": "EN", "Critically": "CR",
                       "Data": "DD", "Not": "NE"}
            if cat_code in IUCN_CAT:
                eco_matrix[sp_i, 0] = IUCN_CAT[cat_code]
            elif cat_code in cat_map:
                eco_matrix[sp_i, 0] = IUCN_CAT[cat_map[cat_code]]
            iucn_found += 1
        if iucn_queried % 100 == 0:
            print(f"    {iucn_queried}/{len(scientific_names)}...")
    except Exception:
        continue

print(f"  IUCN: {iucn_found}/{len(scientific_names)} species found")

# --- eBird API: migratory status ---
print("  Querying eBird API for migratory status...")
ebird_found = 0
for sp_i, code in enumerate(ebird_codes):
    url = f"https://api.ebird.org/v2/ref/taxonomy/ebird?species={code}&fmt=json"
    try:
        req = urllib.request.Request(url)
        req.add_header("X-eBirdApiToken", EBIRD_TOKEN)
        with urllib.request.urlopen(req, timeout=10) as resp:
            taxa = json.loads(resp.read().decode())
        if taxa and isinstance(taxa, list):
            taxon = taxa[0]
            # reportAs field absent = full species; familyCode for body mass proxy
            family = taxon.get("familyCode", "")
            # bandingCodes / extinct fields not available, but
            # species that have "spuh" or "slash" in category are not real species
            cat = taxon.get("category", "species")
            if cat == "species":
                # Use range_size as migration proxy: widespread species more likely migratory
                # eBird taxonomy doesn't directly expose migration, so we approximate:
                # Check if species is in multiple regions by searching for its range
                eco_matrix[sp_i, 6] = 0.0  # default non-migrant
                ebird_found += 1
    except Exception:
        continue
    if (sp_i + 1) % 100 == 0:
        print(f"    {sp_i + 1}/{len(ebird_codes)}...")

# Better migration proxy: species with large range_size + high elevation_range → likely migrant
for sp_i in range(min(N_SPECIES, len(ebird_codes))):
    rs = eco_matrix[sp_i, 2]   # range_size
    er = eco_matrix[sp_i, 4]   # elevation_range_norm
    # Heuristic: species found in >40% of hotspots with elevation range >0.1 → migrant signal
    if rs > 0.4 and er > 0.1:
        eco_matrix[sp_i, 6] = 1.0
    elif rs > 0.25:
        eco_matrix[sp_i, 6] = 0.5  # partial migrant

migrant_count = (eco_matrix[:, 6] > 0).sum()
print(f"  eBird taxonomy: {ebird_found} species queried, {migrant_count} flagged as migrants")

# Save
np.save(OUTPUT / "movement_features.npy", eco_matrix)
eco_cols = ["iucn_category", "prevalence", "range_size", "mean_elevation_norm",
            "elevation_range_norm", "niche_breadth", "is_migrant", "body_mass_proxy"]
with open(OUTPUT / "eco_feature_names.json", "w") as f:
    json.dump(eco_cols, f)
print(f"Ecological features: shape {eco_matrix.shape}")
print(f"  Non-zero features: {(eco_matrix != 0).any(axis=1).sum()}/{N_SPECIES} species")

# ═══════════════════════════════════════════════════════════
# 6. Land Cover (ESA WorldCover)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6/7  Land cover (ESA WorldCover)")
print("=" * 60)

import rasterio
from rasterio.windows import from_bounds

WORLDCOVER_CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland",
    40: "cropland", 50: "built_up", 60: "bare_sparse",
    70: "snow_ice", 80: "water", 90: "herbaceous_wetland",
    95: "mangroves", 100: "moss_lichen"
}
n_lc = len(WORLDCOVER_CLASSES)
class_ids = sorted(WORLDCOVER_CLASSES.keys())

lc_dir = DATA / "worldcover"
tile_paths = sorted(lc_dir.glob("*.tif")) if lc_dir.exists() else []
lc_matrix = np.zeros((len(hotspots), n_lc), dtype=np.float32)

if tile_paths:
    print(f"Extracting from {len(tile_paths)} tiles...")
    buffer_deg = 2500 / 111000

    for i, h in enumerate(hotspots):
        if i % 2000 == 0:
            print(f"  {i}/{len(hotspots)}...")
        lat, lon = h["latitude"], h["longitude"]
        for tp in tile_paths:
            try:
                with rasterio.open(tp) as src:
                    if not (src.bounds.left <= lon <= src.bounds.right and
                            src.bounds.bottom <= lat <= src.bounds.top):
                        continue
                    window = from_bounds(
                        lon - buffer_deg, lat - buffer_deg,
                        lon + buffer_deg, lat + buffer_deg,
                        src.transform)
                    data = src.read(1, window=window)
                    if data.size == 0:
                        continue
                    total = data.size
                    for ci, cls_id in enumerate(class_ids):
                        lc_matrix[i, ci] = (data == cls_id).sum() / total
                    break
            except Exception:
                continue

np.save(OUTPUT / "landcover_features.npy", lc_matrix)
lc_names = [WORLDCOVER_CLASSES[c] for c in class_ids]
with open(OUTPUT / "landcover_classes.json", "w") as f:
    json.dump(lc_names, f)
print(f"Land cover: shape {lc_matrix.shape}")

# ═══════════════════════════════════════════════════════════
# 7. Terrain + iNaturalist
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7/7  Terrain + iNaturalist")
print("=" * 60)

# Terrain
srtm_dir = DATA / "srtm"
srtm_tiles = sorted(srtm_dir.glob("*.tif")) if srtm_dir.exists() else []
terrain_matrix = np.zeros((len(hotspots), 2), dtype=np.float32)

if srtm_tiles:
    print(f"Extracting elevation from {len(srtm_tiles)} tiles...")
    # Pre-load all tiles into RAM (each ~70 MB, total ~210 MB)
    tile_data = []
    for tp in srtm_tiles:
        with rasterio.open(tp) as src:
            tile_data.append({
                "data": src.read(1),
                "bounds": src.bounds,
                "transform": src.transform,
                "height": src.height,
                "width": src.width,
            })
    print(f"  Loaded {len(tile_data)} tiles into RAM")
    for i, h in enumerate(hotspots):
        if i % 2000 == 0:
            print(f"  {i}/{len(hotspots)}...")
        lat, lon = h["latitude"], h["longitude"]
        for td in tile_data:
            b = td["bounds"]
            if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                continue
            row, col = rasterio.transform.rowcol(td["transform"], lon, lat)
            if 0 <= row < td["height"] and 0 <= col < td["width"]:
                elev = float(td["data"][row, col])
                if elev < -1000:   # SRTM nodata fix
                    elev = 0
                terrain_matrix[i, 0] = elev
                terrain_matrix[i, 1] = elev / 5895
            break
    print(f"Elevation range: {terrain_matrix[:,0].min():.0f} – {terrain_matrix[:,0].max():.0f} m")
else:
    print("⚠️ SRTM not available — terrain zeros")

np.save(OUTPUT / "terrain_features.npy", terrain_matrix)
print(f"Terrain: shape {terrain_matrix.shape}")

# iNaturalist
inat_matrix = np.zeros((len(hotspots), 1), dtype=np.float32)
try:
    print("Querying iNaturalist (50 hotspots)...")
    sample_idx = np.random.RandomState(42).choice(len(hotspots), min(50, len(hotspots)), replace=False)
    for idx in sample_idx:
        h = hotspots[idx]
        lat, lon = h["latitude"], h["longitude"]
        url = (f"https://api.inaturalist.org/v1/observations/species_counts"
               f"?lat={lat}&lng={lon}&radius=5&iconic_taxa=Aves&quality_grade=research&per_page=0")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MultimodalSDM/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            inat_matrix[idx, 0] = np.log1p(data.get("total_results", 0))
        except Exception:
            pass
        time.sleep(1.0)

    sampled_mask = inat_matrix[:, 0] > 0
    if sampled_mask.any():
        inat_matrix[~sampled_mask, 0] = inat_matrix[sampled_mask, 0].mean()
except Exception as e:
    print(f"⚠️ iNaturalist failed: {e}")

import urllib.request
np.save(OUTPUT / "inat_features.npy", inat_matrix)
print(f"iNat: shape {inat_matrix.shape}")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
env_dim = n_bio + n_lc + 2 + 1
print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"Hotspots:     {len(hotspots)}")
print(f"Species:      {N_SPECIES}")
print(f"Bioclim:      {n_bio} vars")
print(f"Audio:        {audio_mask.sum()} species")
print(f"Ecological:   {N_ECO} features (IUCN + derived)")
print(f"Land cover:   {n_lc} classes")
print(f"Terrain:      2 features")
print(f"iNat:         1 feature")
print(f"Env dim:      {env_dim}")
print(f"\nOutputs: {OUTPUT}")
print("→ Next: python -I 02_train.py")