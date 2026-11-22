#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 5b: IUCN Red List Features via API v4
==========================================
Fetch real ecological data from IUCN for Kenya bird species:
  - Conservation status (CR/EN/VU/NT/LC/DD → numeric)
  - Habitats (forest, wetland, grassland, etc.)
  - Threats count
  - Population trend

Install: pip install requests
Run:     python -I scripts/05b_iucn_features.py

Outputs: working/iucn_features.csv
         working/movement_features_v2.npz  (updated with IUCN data)
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
WORKING = BASE / "working"
WORKING.mkdir(exist_ok=True)

# ── IUCN API v4 config ──
IUCN_TOKEN = "rlts_XUFqy90Qh3JNuyE54skjIEd0ao9yqK0QPgQm"
IUCN_BASE = "https://apiv4.iucnredlist.org/api/v4"
HEADERS = {
    "Authorization": f"Bearer {IUCN_TOKEN}",
    "Accept": "application/json",
}

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests
CHECKPOINT_FILE = WORKING / "iucn_checkpoint.json"

# ── Load species list ──
species_csv = DATA / "species_list_kenya.csv"
if not species_csv.exists():
    for p in [DATA / "batis" / "species_list_kenya.csv",
              WORKING / "species_list_kenya.csv"]:
        if p.exists():
            species_csv = p
            break

species_df = pd.read_csv(species_csv)
n_species = len(species_df)
print(f"Loaded {n_species} species from {species_csv.name}")

# ── Test API connection ──
print("\nTesting IUCN API v4 connection...")
try:
    r = requests.get(f"{IUCN_BASE}/version", headers=HEADERS, timeout=10)
    if r.status_code == 200:
        print(f"  ✅ Connected! API version: {r.json()}")
    else:
        print(f"  ⚠️ Status {r.status_code}: {r.text[:200]}")
        # Try without /api/v4 prefix
        r2 = requests.get("https://api.iucnredlist.org/api/v4/version",
                          headers=HEADERS, timeout=10)
        if r2.status_code == 200:
            IUCN_BASE = "https://api.iucnredlist.org/api/v4"
            print(f"  ✅ Using alternate base URL: {IUCN_BASE}")
except Exception as e:
    print(f"  ❌ Connection failed: {e}")
    print("  Continuing anyway — will retry per species...")

# ── Conservation status encoding ──
STATUS_MAP = {
    "EX": 7, "EW": 6, "CR": 5, "EN": 4, "VU": 3,
    "NT": 2, "LC": 1, "DD": 0, "NE": -1,
}

TREND_MAP = {
    "decreasing": -1, "stable": 0, "increasing": 1, "unknown": 0,
}

# Major habitat categories for one-hot encoding
HABITAT_CATEGORIES = [
    "Forest", "Savanna", "Shrubland", "Grassland",
    "Wetlands", "Rocky areas", "Desert",
    "Artificial/Terrestrial", "Marine", "Artificial/Aquatic",
]


def fetch_species_data(genus, species):
    """Fetch IUCN data for a species. Returns dict or None."""
    # Step 1: Get taxon info
    try:
        r = requests.get(
            f"{IUCN_BASE}/taxa/scientific_name",
            headers=HEADERS,
            params={"genus_name": genus, "species_name": species},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}

    data = r.json()

    # Parse response — structure may vary
    taxa = None
    if isinstance(data, dict):
        if "assessments" in data:
            taxa = data
        elif "result" in data:
            results = data["result"]
            if isinstance(results, list) and results:
                taxa = results[0]
            elif isinstance(results, dict):
                taxa = results
        elif "taxonid" in data or "taxon_id" in data:
            taxa = data

    if not taxa:
        return None

    result = {
        "found": True,
        "taxon_id": taxa.get("taxonid") or taxa.get("taxon_id"),
    }

    # Extract conservation status
    assessments = taxa.get("assessments", [])
    if assessments:
        latest = assessments[0] if isinstance(assessments, list) else assessments
        code = (latest.get("red_list_category_code")
                or latest.get("category")
                or latest.get("redlistCategory"))
        result["status_code"] = code
        result["status_numeric"] = STATUS_MAP.get(code, -1)
        result["year_published"] = latest.get("year_published")
        trend = latest.get("population_trend", "unknown")
        if isinstance(trend, str):
            result["population_trend"] = trend.lower()
            result["trend_numeric"] = TREND_MAP.get(trend.lower(), 0)
    elif "category" in taxa:
        code = taxa["category"]
        result["status_code"] = code
        result["status_numeric"] = STATUS_MAP.get(code, -1)

    # Step 2: Get habitats (if we have taxon_id)
    taxon_id = result.get("taxon_id")
    if taxon_id:
        time.sleep(REQUEST_DELAY)
        try:
            rh = requests.get(
                f"{IUCN_BASE}/habitats",
                headers=HEADERS,
                params={"taxonid": taxon_id},
                timeout=15,
            )
            if rh.status_code == 200:
                hab_data = rh.json()
                habs = hab_data if isinstance(hab_data, list) else hab_data.get("result", [])
                result["habitats"] = []
                if isinstance(habs, list):
                    for h in habs:
                        name = h.get("habitat") or h.get("name") or ""
                        result["habitats"].append(name)
                result["n_habitats"] = len(result["habitats"])
        except Exception:
            pass

        # Step 3: Get threats
        time.sleep(REQUEST_DELAY)
        try:
            rt = requests.get(
                f"{IUCN_BASE}/threats",
                headers=HEADERS,
                params={"taxonid": taxon_id},
                timeout=15,
            )
            if rt.status_code == 200:
                thr_data = rt.json()
                threats = thr_data if isinstance(thr_data, list) else thr_data.get("result", [])
                result["n_threats"] = len(threats) if isinstance(threats, list) else 0
        except Exception:
            pass

    return result


# ── Load checkpoint ──
species_data = {}
start_idx = 0

if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE) as f:
        ckpt = json.load(f)
    species_data = ckpt.get("data", {})
    start_idx = ckpt.get("last_idx", 0) + 1
    n_found = sum(1 for v in species_data.values()
                  if isinstance(v, dict) and v.get("found"))
    print(f"Resuming from checkpoint: {start_idx}/{n_species} "
          f"({n_found} found)")

# ── Fetch data for each species ──
print(f"\nFetching IUCN data for {n_species} species...")
t0 = time.time()
n_found = sum(1 for v in species_data.values()
              if isinstance(v, dict) and v.get("found"))
n_errors = 0
n_not_found = 0

for i in range(start_idx, n_species):
    row = species_df.iloc[i]
    sci_name = str(row["scientific_name"])
    parts = sci_name.split()

    if len(parts) < 2:
        n_errors += 1
        continue

    genus, sp = parts[0], parts[1]

    # Skip if already fetched
    if sci_name in species_data:
        continue

    result = fetch_species_data(genus, sp)
    time.sleep(REQUEST_DELAY)

    if result is None:
        species_data[sci_name] = {"found": False}
        n_not_found += 1
    elif "error" in result:
        species_data[sci_name] = {"found": False, "error": result["error"]}
        n_errors += 1
    else:
        species_data[sci_name] = result
        n_found += 1

    # Progress
    if (i + 1) % 25 == 0 or i == n_species - 1:
        elapsed = time.time() - t0
        rate = (i - start_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (n_species - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{n_species}] found={n_found}, "
              f"not_found={n_not_found}, errors={n_errors}, "
              f"{elapsed:.0f}s, ETA {eta/60:.1f}min")

    # Checkpoint every 25 species
    if (i + 1) % 25 == 0:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"last_idx": i, "data": species_data}, f)

# ── Save final checkpoint ──
with open(CHECKPOINT_FILE, "w") as f:
    json.dump({"last_idx": n_species - 1, "data": species_data}, f)

# ── Build features DataFrame ──
print("\nBuilding feature matrix...")

records = []
for i, row in species_df.iterrows():
    sci_name = str(row["scientific_name"])
    d = species_data.get(sci_name, {})

    rec = {
        "species_idx": i,
        "scientific_name": sci_name,
        "iucn_found": 1 if d.get("found") else 0,
        "status_numeric": d.get("status_numeric", -1),
        "trend_numeric": d.get("trend_numeric", 0),
        "n_habitats": d.get("n_habitats", 0),
        "n_threats": d.get("n_threats", 0),
    }

    # Habitat one-hot
    habitats = d.get("habitats", [])
    for cat in HABITAT_CATEGORIES:
        rec[f"hab_{cat.lower().split('/')[0].replace(' ', '_')}"] = (
            1 if any(cat.lower() in h.lower() for h in habitats) else 0
        )

    records.append(rec)

features_df = pd.DataFrame(records)
features_df.to_csv(WORKING / "iucn_features.csv", index=False)

# ── Summary ──
n_total_found = features_df["iucn_found"].sum()
print(f"\n✅ IUCN features saved to {WORKING / 'iucn_features.csv'}")
print(f"   {n_total_found}/{n_species} species found in IUCN Red List")

status_counts = features_df[features_df["iucn_found"] == 1]["status_numeric"].value_counts()
inv_status = {v: k for k, v in STATUS_MAP.items()}
print("\n   Conservation status distribution:")
for val, count in status_counts.sort_index(ascending=False).items():
    label = inv_status.get(val, "?")
    print(f"     {label}: {count}")

trend_counts = features_df[features_df["iucn_found"] == 1]["trend_numeric"].value_counts()
inv_trend = {v: k for k, v in TREND_MAP.items()}
print("\n   Population trend distribution:")
for val, count in trend_counts.sort_index().items():
    for k, v in TREND_MAP.items():
        if v == val:
            print(f"     {k}: {count}")
            break

# ── Merge with existing movement features ──
old_move = WORKING / "movement_features.npz"
if old_move.exists():
    old_data = np.load(old_move)
    old_features = old_data["features"]  # [n_species, 8]
    n_sp, old_dim = old_features.shape
    print(f"\n   Existing movement features: {old_dim}-dim for {n_sp} species")

    # Add IUCN columns: status, trend, n_habitats, n_threats, hab_forest..hab_marine
    iucn_cols = ["status_numeric", "trend_numeric", "n_habitats", "n_threats"]
    iucn_cols += [c for c in features_df.columns if c.startswith("hab_")]
    iucn_matrix = features_df[iucn_cols].values.astype(np.float32)

    # Normalize
    for j in range(iucn_matrix.shape[1]):
        col = iucn_matrix[:, j]
        std = col.std()
        if std > 0:
            iucn_matrix[:, j] = (col - col.mean()) / std

    # Concatenate
    if n_sp == len(features_df):
        combined = np.concatenate([old_features, iucn_matrix], axis=1)
        np.savez(WORKING / "movement_features_v2.npz",
                 features=combined)
        print(f"   ✅ Combined features: {old_dim} + {iucn_matrix.shape[1]} "
              f"= {combined.shape[1]}-dim → movement_features_v2.npz")
    else:
        print(f"   ⚠️ Size mismatch: movement={n_sp}, iucn={len(features_df)}")

print(f"\n→ Next: update 02_train.py to use movement_features_v2.npz")
print(f"  Change: move_input_dim accordingly")
