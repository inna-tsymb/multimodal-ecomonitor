#!/usr/bin/env python3
"""
01b_expand_audio.py — Expand audio embeddings coverage.

Sources:
  1. BirdCLEF 2024 (Kaggle) — adds ~150 more species
  2. Xeno-Canto API — fills remaining gaps (needs API key)

Run AFTER 01_data_prep.py. Overwrites audio_embeddings.npy and audio_mask.npy.
"""

import os, json, urllib.request, urllib.error, time
from pathlib import Path
import numpy as np

# ── Paths ──────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTPUT = BASE / "working"
BIRDCLEF_DIR = DATA / "birdclef"

# ── Config ─────────────────────────────────────────────────
N_SPECIES = 1054
N_MELS = 128
MAX_FILES_PER_SPECIES = 10  # max audio files to average per species
XC_MAX_DOWNLOADS = 5        # max Xeno-Canto recordings per species
XC_QUALITY = "A"            # quality filter for Xeno-Canto

# ── API keys ───────────────────────────────────────────────
os.environ["KAGGLE_API_TOKEN"] = "KGAT_82e7b192176fb85521f81f63b04bd10e"
XENO_CANTO_API_KEY = os.environ.get("XENO_CANTO_API_KEY", "69a2d49ea75a62fae7759bd6a26cf9d5125ddf19")  # set if available

# ── Load species list ──────────────────────────────────────
species_list_path = DATA / "species_list_kenya.csv"
if not species_list_path.exists():
    species_list_path = OUTPUT / "species_list_kenya.csv"
if not species_list_path.exists():
    print("ERROR: Run 01_data_prep.py first!")
    raise SystemExit(1)

import pandas as pd
species_df = pd.read_csv(species_list_path)
ebird_codes = species_df["ebird_code"].tolist()
scientific_names = species_df["scientific_name"].tolist()
ebird_to_idx = {code: i for i, code in enumerate(ebird_codes)}
sciname_to_idx = {name: i for i, name in enumerate(scientific_names)}

print(f"Species: {len(ebird_codes)} known + {N_SPECIES - len(ebird_codes)} padding = {N_SPECIES}")

# ── Audio embedding function ───────────────────────────────
def compute_embedding(audio_path):
    """Compute mean mel-spectrogram embedding for one audio file."""
    import signal
    def _timeout(signum, frame):
        raise TimeoutError()
    try:
        import librosa
        old_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(20)  # 20s max per file
        try:
            y, sr = librosa.load(str(audio_path), sr=22050, duration=30)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if len(y) < sr:  # less than 1 second
            return None
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.mean(axis=1).astype(np.float32)  # (n_mels,)
    except Exception:
        return None


# ── Step 1: Load existing embeddings ──────────────────────
audio_matrix = np.zeros((N_SPECIES, N_MELS), dtype=np.float32)
audio_mask = np.zeros(N_SPECIES, dtype=bool)

existing_emb = OUTPUT / "audio_embeddings.npy"
existing_mask = OUTPUT / "audio_mask.npy"
if existing_emb.exists() and existing_mask.exists():
    audio_matrix = np.load(existing_emb)
    audio_mask = np.load(existing_mask)
    print(f"Loaded existing: {audio_mask.sum()}/{N_SPECIES} species")
else:
    print("No existing embeddings — starting fresh")

initial_coverage = int(audio_mask.sum())
missing_codes = [code for code in ebird_codes if not audio_mask[ebird_to_idx[code]]]
print(f"Missing: {len(missing_codes)} species\n")


# ── Step 2: BirdCLEF 2024 ─────────────────────────────────
print("=" * 60)
print("STEP 1: BirdCLEF 2024 (Kaggle)")
print("=" * 60)

bc24_dir = DATA / "birdclef2024"
bc24_audio = bc24_dir / "train_audio"

if not bc24_audio.exists():
    print("  Downloading BirdCLEF 2024...")
    bc24_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("birdclef-2024", path=str(bc24_dir), quiet=False)
        # Extract
        import zipfile
        for zf in bc24_dir.glob("*.zip"):
            print(f"  Extracting {zf.name}...")
            with zipfile.ZipFile(zf) as z:
                z.extractall(bc24_dir)
            zf.unlink()
    except Exception as e:
        print(f"  ⚠️ BirdCLEF 2024 download failed: {e}")
        print("  Try: kaggle competitions download -c birdclef-2024")

# Find audio dir (may be nested)
for candidate in [bc24_audio, bc24_dir / "train_audio", bc24_dir]:
    if candidate.exists() and any(candidate.rglob("*.ogg")):
        bc24_audio = candidate
        break

bc24_matched = 0
if bc24_audio.exists():
    species_folders = sorted([d for d in bc24_audio.iterdir() if d.is_dir()])
    print(f"  BirdCLEF 2024 species: {len(species_folders)}")

    for sp_folder in species_folders:
        sp_code = sp_folder.name
        sp_idx = ebird_to_idx.get(sp_code)
        if sp_idx is None or audio_mask[sp_idx]:
            continue  # unknown or already covered

        audio_files = (list(sp_folder.glob("*.ogg")) + list(sp_folder.glob("*.mp3")))[:MAX_FILES_PER_SPECIES]
        embs = []
        for af in audio_files:
            emb = compute_embedding(af)
            if emb is not None:
                embs.append(emb)
        if embs:
            audio_matrix[sp_idx] = np.mean(embs, axis=0)
            audio_mask[sp_idx] = True
            bc24_matched += 1

    print(f"  +{bc24_matched} new species from BirdCLEF 2024")
else:
    print("  BirdCLEF 2024 not available")


# ── Step 3: BirdCLEF 2025 ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: BirdCLEF 2025 (Kaggle)")
print("=" * 60)

bc25_dir = DATA / "birdclef2025"
bc25_audio = bc25_dir / "train_audio"

if not bc25_audio.exists():
    print("  Downloading BirdCLEF 2025...")
    bc25_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("birdclef-2025", path=str(bc25_dir), quiet=False)
        import zipfile
        for zf in bc25_dir.glob("*.zip"):
            print(f"  Extracting {zf.name}...")
            with zipfile.ZipFile(zf) as z:
                z.extractall(bc25_dir)
            zf.unlink()
    except Exception as e:
        print(f"  ⚠️ BirdCLEF 2025 download failed: {e}")

for candidate in [bc25_audio, bc25_dir / "train_audio", bc25_dir]:
    if candidate.exists() and any(candidate.rglob("*.ogg")):
        bc25_audio = candidate
        break

bc25_matched = 0
if bc25_audio.exists():
    species_folders = sorted([d for d in bc25_audio.iterdir() if d.is_dir()])
    print(f"  BirdCLEF 2025 species: {len(species_folders)}")

    for sp_folder in species_folders:
        sp_code = sp_folder.name
        sp_idx = ebird_to_idx.get(sp_code)
        if sp_idx is None or audio_mask[sp_idx]:
            continue

        audio_files = (list(sp_folder.glob("*.ogg")) + list(sp_folder.glob("*.mp3")))[:MAX_FILES_PER_SPECIES]
        embs = []
        for af in audio_files:
            emb = compute_embedding(af)
            if emb is not None:
                embs.append(emb)
        if embs:
            audio_matrix[sp_idx] = np.mean(embs, axis=0)
            audio_mask[sp_idx] = True
            bc25_matched += 1

    print(f"  +{bc25_matched} new species from BirdCLEF 2025")
else:
    print("  BirdCLEF 2025 not available")


# ── Step 4: Xeno-Canto API ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Xeno-Canto API")
print("=" * 60)

still_missing = [code for code in ebird_codes if not audio_mask[ebird_to_idx[code]]]
print(f"  Still missing: {len(still_missing)} species")

xc_dir = DATA / "xenocanto"
xc_dir.mkdir(parents=True, exist_ok=True)
xc_matched = 0
xc_failed = 0

if still_missing:
    for i, sp_code in enumerate(still_missing):
        sp_idx = ebird_to_idx[sp_code]
        sci_name = scientific_names[sp_idx] if sp_idx < len(scientific_names) else None
        if not sci_name:
            continue

        # Check if already downloaded
        sp_dir = xc_dir / sp_code
        if sp_dir.exists() and list(sp_dir.glob("*.mp3")):
            # Already have files — compute embedding
            audio_files = list(sp_dir.glob("*.mp3"))[:MAX_FILES_PER_SPECIES]
            embs = []
            for af in audio_files:
                emb = compute_embedding(af)
                if emb is not None:
                    embs.append(emb)
            if embs:
                audio_matrix[sp_idx] = np.mean(embs, axis=0)
                audio_mask[sp_idx] = True
                xc_matched += 1
            continue

        # Query Xeno-Canto API v3
        query = f'sp:"{sci_name}" q:{XC_QUALITY}'
        encoded_query = urllib.request.quote(query)
        url = f"https://xeno-canto.org/api/3/recordings?query={encoded_query}"
        if XENO_CANTO_API_KEY:
            url += f"&key={XENO_CANTO_API_KEY}"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "MultimodalSDM/1.0 (academic research)")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            recordings = data.get("recordings", [])
            if not recordings:
                # Try without quality filter
                query2 = f'sp:"{sci_name}"'
                encoded_query2 = urllib.request.quote(query2)
                url2 = f"https://xeno-canto.org/api/3/recordings?query={encoded_query2}"
                if XENO_CANTO_API_KEY:
                    url2 += f"&key={XENO_CANTO_API_KEY}"
                req2 = urllib.request.Request(url2)
                req2.add_header("User-Agent", "MultimodalSDM/1.0 (academic research)")
                with urllib.request.urlopen(req2, timeout=15) as resp2:
                    data = json.loads(resp2.read().decode())
                recordings = data.get("recordings", [])

            # Debug first 3 species
            if i < 3:
                print(f"    DEBUG {sci_name}: {len(recordings)} recordings, url={url}")

            if not recordings:
                xc_failed += 1
                continue

            # Download up to XC_MAX_DOWNLOADS recordings
            sp_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            embs = []

            for rec in recordings[:XC_MAX_DOWNLOADS]:
                file_url = rec.get("file", "")
                if not file_url:
                    continue
                if file_url.startswith("//"):
                    file_url = "https:" + file_url

                fname = sp_dir / f"{rec['id']}.mp3"
                try:
                    dl_req = urllib.request.Request(file_url)
                    dl_req.add_header("User-Agent", "MultimodalSDM/1.0 (academic research)")
                    if XENO_CANTO_API_KEY:
                        dl_req.add_header("Cookie", f"xc_api_key={XENO_CANTO_API_KEY}")
                    with urllib.request.urlopen(dl_req, timeout=30) as dl_resp:
                        with open(fname, "wb") as f:
                            f.write(dl_resp.read())
                    downloaded += 1

                    emb = compute_embedding(fname)
                    if emb is not None:
                        embs.append(emb)
                except (urllib.error.HTTPError, urllib.error.URLError) as dl_err:
                    if i < 3:
                        print(f"    DEBUG download error: {dl_err}")
                    continue

            if embs:
                audio_matrix[sp_idx] = np.mean(embs, axis=0)
                audio_mask[sp_idx] = True
                xc_matched += 1

            # Rate limit
            time.sleep(0.5)

        except Exception as e:
            if i < 3:
                print(f"    DEBUG search error for {sci_name}: {e}")
            xc_failed += 1

        if (i + 1) % 50 == 0:
            # Checkpoint save every 50 species
            np.save(OUTPUT / "audio_embeddings.npy", audio_matrix)
            np.save(OUTPUT / "audio_mask.npy", audio_mask)
            print(f"    {i + 1}/{len(still_missing)} queried, +{xc_matched} found, saved checkpoint")

    print(f"  +{xc_matched} new species from Xeno-Canto")
    if xc_failed:
        print(f"  {xc_failed} species not found on Xeno-Canto")


# ── Save ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

final_coverage = int(audio_mask.sum())
print(f"  Before:  {initial_coverage}/{N_SPECIES}")
print(f"  After:   {final_coverage}/{N_SPECIES}")
print(f"  Added:   +{final_coverage - initial_coverage}")
print(f"    BirdCLEF 2024: +{bc24_matched}")
print(f"    BirdCLEF 2025: +{bc25_matched}")
print(f"    Xeno-Canto:    +{xc_matched}")

np.save(OUTPUT / "audio_embeddings.npy", audio_matrix)
np.save(OUTPUT / "audio_mask.npy", audio_mask)
print(f"\n✅ Saved to {OUTPUT}")
print(f"→ Next: python scripts/02_train.py")