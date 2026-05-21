# Early Disaster Detection from Social Media Streams

MSc thesis project implementing an end-to-end pipeline for real-time
disaster detection from Twitter using transformer classification,
geoparsing, and temporal burst detection.

**Live Dashboard:** https://bazilali25-disaster-tweet-monitor.hf.space

---

## Project Overview

| Component | Method | Performance |
|---|---|---|
| Classification | DistilBERT fine-tuned on CrisisBench + HumAID | Macro F1 = 0.880 |
| Geoparsing | GeoNames gazetteer (Geoparser A) | Coverage = 53.8% |
| Burst Detection | CUSUM + Z-score detectors | Median latency = 0.4 min |

## Notebooks

| Notebook | Description |
|---|---|
| `Thesis_Stage_01_Upload.ipynb` | Data loading, merging CrisisBench + HumAID, preprocessing, EDA |
| `Thesis_Stage_02_Upload.ipynb` | DistilBERT and RoBERTa fine-tuning, threshold calibration, LOTO evaluation |
| `Thesis_Stage_03_(CPU)_Upload.ipynb` | Stream simulation, burst detection, geoparsing (CPU version) |
| `Thesis_Stage_03_(GPU)_Upload.ipynb` | Stream simulation, burst detection, geoparsing (GPU version) |

## How to Run

1. Open any notebook in Google Colab
2. Runtime → Change runtime type → **T4 GPU** (for Stage 2 and GPU Stage 3)
3. Run all cells in order (Ctrl+F9)
4. Models and data are loaded from Google Drive and HuggingFace

## Datasets

- [CrisisBench](https://huggingface.co/datasets/QCRI/CrisisBench-english) — 
  166,100 disaster tweets
- [HumAID](https://huggingface.co/datasets/QCRI/HumAID-all) — 
  77,000 humanitarian tweets
- [IDRISI-D](https://github.com/rsuwaileh/IDRISI) — 
  geoparsing benchmark

## Dashboard

Built with Streamlit, deployed permanently on Hugging Face Spaces.
Source code in `dashboard/app.py`.

**Live URL:** https://bazilali25-disaster-tweet-monitor.hf.space

## Requirements
pip install -r requirements.txt
