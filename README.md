# Multimodal EcoMonitor

Multimodal Species Distribution Model — predicting bird encounter rates in Kenya using satellite imagery, audio, environmental, and movement data.

## Hardware

- GPU: NVIDIA GeForce RTX 5070 (12 GB VRAM)
- CPU: AMD Ryzen 5 5600X (6 cores)
- RAM: 32 GB
- OS: Ubuntu (WSL2) on Windows

## Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.1)
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Install dependencies
uv pip install datasets huggingface_hub rasterio librosa scikit-learn scikit-image kaggle pandas matplotlib scipy
```

## Run order

```bash
python -I scripts/00_download_all.py   # ~1-2 hours (20 GB satellite + BirdCLEF)
python -I scripts/01_data_prep.py      # ~15-30 min (CPU)
python -I scripts/02_train.py           # ~1-2 hours (RTX 5070, 7 ablation configs)
python -I scripts/03_analysis.py        # ~1 min (CPU, figures)
```

## Before first run

1. Accept BirdCLEF 2023 rules: https://www.kaggle.com/competitions/birdclef-2023/rules
2. Kaggle API token is embedded in scripts

## Directory structure after download

```
multimodal-ecomonitor/
├── scripts/         # 00-03 pipeline scripts
├── data/
│   ├── batis/           # BATIS metadata (HuggingFace)
│   ├── satbird_targets/ # 9763 JSON encounter rate files
│   ├── satbird_images/  # 9763 Sentinel-2 TIF (20 GB)
│   ├── birdclef/        # BirdCLEF 2023 audio (~50 GB)
│   ├── worldcover/      # ESA WorldCover tiles
│   └── srtm/            # SRTM elevation
├── working/             # Preprocessed features (.npy)
└── results/             # Models, predictions, figures
```