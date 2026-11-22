#!/usr/bin/env python -I
# -*- coding: utf-8 -*-
"""
Step 0: Download ALL datasets — Local PC version
=================================================
Run: python -I 00_download_all.py

Requirements:
  pip install datasets huggingface_hub rasterio geopandas librosa kaggle

Datasets:
  1. BATIS Kenya (HuggingFace) — metadata + bioclim + targets + satellite
  2. BirdCLEF 2023 (Kaggle) — bird audio
  3. ESA WorldCover — land cover
  4. SRTM — elevation
  5. iNaturalist — API
"""

import os
import json
import subprocess
import zipfile
import tarfile
import urllib.request
from pathlib import Path

# ── All data goes here ──
BASE = Path(__file__).resolve().parent.parent  # scripts/ → project root
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. BATIS Kenya — metadata CSV + targets + satellite images
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("1/5  BATIS Kenya (HuggingFace)")
print("=" * 60)

batis_dir = DATA / "batis"
batis_dir.mkdir(exist_ok=True)

# 1a. Load CSV metadata (bioclim, coordinates, splits)
if not (batis_dir / "done_meta.flag").exists():
    from datasets import load_dataset
    print("Loading BATIS Kenya metadata from HuggingFace...")
    cache = str(DATA / "hf_cache")
    ds_train = load_dataset("cathv/BATIS", name="Kenya", split="train", cache_dir=cache)
    ds_val   = load_dataset("cathv/BATIS", name="Kenya", split="val",   cache_dir=cache)
    ds_test  = load_dataset("cathv/BATIS", name="Kenya", split="test",  cache_dir=cache)

    print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")
    print(f"  Features: {list(ds_train.features.keys())}")

    ds_train.save_to_disk(str(batis_dir / "train"))
    ds_val.save_to_disk(str(batis_dir / "val"))
    ds_test.save_to_disk(str(batis_dir / "test"))

    feat_info = {k: str(v) for k, v in ds_train.features.items()}
    with open(batis_dir / "features.json", "w") as f:
        json.dump(feat_info, f, indent=2)

    (batis_dir / "done_meta.flag").touch()
    print("✅ BATIS metadata saved")
else:
    print("✅ BATIS metadata already exists")

# 1b. Download targets.tar.gz and images.tar.gz
from huggingface_hub import hf_hub_download

targets_dir = DATA / "satbird_targets"
sat_image_dir = DATA / "satbird_images"

# Targets
if not targets_dir.exists() or len(list(targets_dir.glob("*.json"))) < 100:
    print("Downloading targets.tar.gz (~50 MB)...")
    tar_path = hf_hub_download(
        repo_id="cathv/BATIS", filename="Kenya/targets.tar.gz",
        repo_type="dataset", cache_dir=str(DATA / "hf_cache"))
    targets_dir.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(targets_dir)
    # Clean macOS junk
    for junk in targets_dir.rglob("._*"):
        junk.unlink()
    # Flatten nested dirs — move all .json to targets_dir root
    for f in list(targets_dir.rglob("*.json")):
        if f.parent != targets_dir:
            dest = targets_dir / f.name
            f.rename(dest)
    # Remove empty subdirs
    for d in sorted(targets_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try: d.rmdir()
            except OSError: pass
    real = list(targets_dir.glob("*.json"))
    print(f"✅ Targets: {len(real)} JSON files")
else:
    print(f"✅ Targets already exist: {len(list(targets_dir.glob('*.json')))} files")

# Satellite images
if not sat_image_dir.exists() or len(list(sat_image_dir.glob("*.tif"))) < 100:
    print("Downloading images.tar.gz (~20 GB) — this will take a while...")
    tar_path = hf_hub_download(
        repo_id="cathv/BATIS", filename="Kenya/images.tar.gz",
        repo_type="dataset", cache_dir=str(DATA / "hf_cache"))
    sat_image_dir.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(sat_image_dir)
    for junk in sat_image_dir.rglob("._*"):
        junk.unlink()
    # Flatten nested dirs — move all .tif to sat_image_dir root
    for f in list(sat_image_dir.rglob("*.tif")):
        if f.parent != sat_image_dir:
            dest = sat_image_dir / f.name
            f.rename(dest)
    # Remove empty subdirs
    for d in sorted(sat_image_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try: d.rmdir()
            except OSError: pass
    tifs = list(sat_image_dir.glob("*.tif"))
    print(f"✅ Satellite images: {len(tifs)} TIF files")
else:
    print(f"✅ Satellite images already exist: {len(list(sat_image_dir.glob('*.tif')))} files")

# ═══════════════════════════════════════════════════════════
# 2. BirdCLEF 2023 (Kaggle) — bird audio
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2/5  BirdCLEF 2023 (Kaggle)")
print("=" * 60)

birdclef_dir = DATA / "birdclef"
birdclef_dir.mkdir(exist_ok=True)

# Set Kaggle API token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_82e7b192176fb85521f81f63b04bd10e"

if not list(birdclef_dir.rglob("*.ogg")):
    print("Downloading BirdCLEF 2023...")
    print("⚠️  First accept rules at: https://www.kaggle.com/competitions/birdclef-2023/rules")
    try:
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "birdclef-2023",
            "-p", str(birdclef_dir)
        ], check=True)
        # Unzip
        for zf in birdclef_dir.glob("*.zip"):
            print(f"  Extracting {zf.name}...")
            with zipfile.ZipFile(zf) as z:
                z.extractall(birdclef_dir)
            zf.unlink()
        n_ogg = len(list(birdclef_dir.rglob("*.ogg")))
        print(f"✅ BirdCLEF: {n_ogg} audio files")
    except Exception as e:
        print(f"⚠️ BirdCLEF download failed: {e}")
        print("  Make sure you accepted competition rules and kaggle CLI is installed")
else:
    n_ogg = len(list(birdclef_dir.rglob("*.ogg")))
    print(f"✅ BirdCLEF already exists ({n_ogg} audio files)")

# ═══════════════════════════════════════════════════════════
# 3. ESA WorldCover 2021
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3/5  ESA WorldCover (Land Cover)")
print("=" * 60)

lc_dir = DATA / "worldcover"
lc_dir.mkdir(exist_ok=True)

esa_tiles = [
    "N00E033", "N00E036", "N00E039",
    "N03E033", "N03E036", "N03E039",
    "S03E033", "S03E036", "S03E039",
]
base_url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"

if len(list(lc_dir.glob("*.tif"))) < 3:
    for tile in esa_tiles:
        fname = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        fpath = lc_dir / fname
        if fpath.exists():
            continue
        url = base_url + fname
        try:
            print(f"  Downloading {tile}...", end=" ", flush=True)
            urllib.request.urlretrieve(url, fpath)
            print(f"✅ ({fpath.stat().st_size // 1024 // 1024} MB)")
        except Exception:
            print("⚠️ not available")
    print(f"✅ ESA WorldCover: {len(list(lc_dir.glob('*.tif')))} tiles")
else:
    print(f"✅ ESA WorldCover already exists ({len(list(lc_dir.glob('*.tif')))} tiles)")

# ═══════════════════════════════════════════════════════════
# 4. SRTM Elevation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4/5  SRTM Elevation")
print("=" * 60)

srtm_dir = DATA / "srtm"
srtm_dir.mkdir(exist_ok=True)

srtm_tiles = ["srtm_44_13", "srtm_44_14", "srtm_45_13", "srtm_45_14"]
srtm_base = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"

if len(list(srtm_dir.glob("*.tif"))) < 2:
    for tile in srtm_tiles:
        fpath = srtm_dir / f"{tile}.tif"
        if fpath.exists():
            continue
        try:
            url = f"{srtm_base}{tile}.zip"
            zpath = srtm_dir / f"{tile}.zip"
            print(f"  Downloading {tile}...", end=" ", flush=True)
            urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath, 'r') as z:
                z.extractall(srtm_dir)
            zpath.unlink()
            print("✅")
        except Exception:
            print("⚠️ failed")
    print(f"✅ SRTM: {len(list(srtm_dir.glob('*.tif')))} tiles")
else:
    print(f"✅ SRTM already exists ({len(list(srtm_dir.glob('*.tif')))} tiles)")

# ═══════════════════════════════════════════════════════════
# 5. iNaturalist API check
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5/5  iNaturalist API check")
print("=" * 60)

try:
    url = "https://api.inaturalist.org/v1/observations?place_id=6878&iconic_taxa=Aves&per_page=1"
    req = urllib.request.Request(url, headers={"User-Agent": "MultimodalSDM/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    total = data.get("total_results", 0)
    print(f"✅ iNaturalist API accessible — {total:,} bird observations in Kenya")
except Exception as e:
    print(f"⚠️ iNaturalist API not reachable: {e}")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

sizes = {}
for name, d in [("batis", batis_dir), ("birdclef", birdclef_dir),
                ("worldcover", lc_dir), ("srtm", srtm_dir),
                ("targets", targets_dir), ("satellite", sat_image_dir)]:
    if d.exists():
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024 / 1024
        sizes[name] = sz
        print(f"  {name:12s}: {sz:,.0f} MB")

total_mb = sum(sizes.values())
print(f"\nTotal: {total_mb:,.0f} MB ({total_mb/1024:.1f} GB)")
print(f"\n→ Next: python -I 01_data_prep.py")