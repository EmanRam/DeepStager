# DeepStager: Automatic Multi-Stage Sleep Scoring from Raw Single-Channel EEG

## Overview

DeepStager develops and compares two approaches to automated sleep stage classification from a single EEG channel:

1. **XGBoost Baseline** — A 53-dimensional handcrafted feature pipeline paired with an XGBoost gradient-boosted classifier, providing an interpretable reference.
2. **CNN + BiLSTM (DeepSleepNet-inspired)** — A two-stage deep learning architecture combining a dual-stream CNN with a bidirectional LSTM for sequential context modelling.

Both systems target five-class AASM sleep staging (W, N1, N2, N3, REM) on the **Sleep-EDF Expanded (Cassette)** dataset using the **Fpz-Cz** EEG channel.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deepstager.git
cd deepstager
```

### 2. Create and Activate a Virtual Environment

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the libraries directly:

```bash
pip install mne>=1.12.1 PyWavelets xgboost>=2.0 torch scikit-learn numpy scipy matplotlib
```

### 4. Download the Dataset

Download the **Sleep-EDF Expanded (Cassette)** dataset from PhysioNet:

```bash
# Install the wget tool if needed, then download the EDF files
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```

Place the downloaded `.edf` files in the `data/` directory.

---

## Running the Notebooks

The project currently consists of two Jupyter notebooks:

| Notebook | Description |
|---|---|
| `01_xgboost_baseline.ipynb` | Feature extraction (CWT, time-domain, Welch) and XGBoost training & evaluation |
| `02_cnn_bilstm.ipynb` | Two-stage deep learning pipeline: CNN pre-training + BiLSTM fine-tuning |

Launch Jupyter and run them in order:

```bash
jupyter notebook
```

> **Note:** The deep learning notebook (`02_cnn_bilstm.ipynb`) requires a GPU for reasonable runtimes. If you don't have a local GPU, open it directly in [Google Colab](https://colab.research.google.com/) and enable a T4 GPU runtime under **Runtime → Change runtime type**.

---

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| XGBoost Baseline (76 subjects) | 79.45% | 72.49% |
| TwoStreamCNN — Stage 1 (20 subjects) | 84.2% | 76.3% |
| CNN + BiLSTM — Stage 2 (20 subjects) | **85.2%** | **77.5%** |

> **Note:** N1 remains the hardest class across all models (F1 ≈ 0.35), consistent with the broader sleep staging literature.

---

## Dataset

- **Source:** [Sleep-EDF Expanded (Cassette) — PhysioNet](https://physionet.org/content/sleep-edfx/)
- **Subjects used:** 76 recordings (first 38 subjects × 2 nights)
- **Channel:** Fpz-Cz EEG, sampled at 100 Hz
- **Epoch length:** 30 seconds (3,000 samples)
- **Total epochs (baseline):** 88,387 across 76 subjects
- **Classes:** Wake (W), N1, N2, N3, REM

### Class Distribution (76-subject set)
| Stage | Epochs | % |
|---|---|---|
| Wake | 23,851 | 27% |
| N1 | 7,534 | 8.5% |
| N2 | 35,262 | 35.3% |
| N3 | 7,930 | 9% |
| REM | 13,801 | 15.9% |

---

## Methodology

### Preprocessing (both models)
- Resample to 100 Hz
- Bandpass filter: 0.5–40 Hz (zero-phase Hamming-windowed FIR)
- Wake trimming: discard epochs >30 min before/after sleep onset
- Exclude Movement Time and Unknown epochs

### Model 1 — XGBoost with Handcrafted Features

Each 30-second epoch is represented by a **53-dimensional feature vector** from three domains:

| Domain | Features | Count |
|---|---|---|
| CWT Sub-band Statistics (Morlet, 5 bands × 8 stats) | Mean, Std, Energy, Spectral Entropy, Skewness, Kurtosis, Max, Min | 40 |
| Time-Domain | ZCR, RMS, MAV, Waveform Length, Log-Variance, Hjorth Activity/Mobility/Complexity | 8 |
| Welch Relative Spectral Power | Delta, Theta, Alpha, Sigma, Beta band ratios | 5 |

**Classifier:** XGBoost (200 trees, depth 6, lr 0.1, subsample 0.8, inverse-frequency class weighting)  
**Evaluation:** Subject-disjoint 80/20 split + 5-fold stratified cross-validation on training set

### Model 2 — CNN + BiLSTM (DeepSleepNet)

A two-stage training paradigm operating directly on raw EEG:

**Stage 1 — Dual-stream CNN Pre-training**
- Small-filter stream: kernel 50, stride 6 → captures sleep spindles & K-complexes
- Large-filter stream: kernel 400, stride 50 → captures slow-wave morphology
- Each stream: 3× ConvBlocks (Conv1D → BN → ReLU → MaxPool → Dropout 50%) → AdaptiveMaxPool → 128-d
- Concatenated to 256-d → Linear classifier (5 classes)
- Training: Adam (lr=1e-4), 30 epochs, batch 128, oversampled balanced training set

**Stage 2 — BiLSTM Fine-tuning**
- Input: sequences of 25 consecutive epochs (25 × 3,000)
- 2× stacked BiLSTM layers (512 units/direction → 1,024-d output)
- Residual shortcut: CNN output (256-d) projected to 1,024-d and added to BiLSTM output
- Two-LR strategy: CNN lr=1e-6, BiLSTM lr=1e-4; gradient clipping (norm=10)
- Training: 20 epochs, batch 16 sequences; evaluation: leave-one-subject-out (SC4001)

---

## Per-Class F1 Comparison

| Stage | XGBoost | CNN Only | CNN + BiLSTM |
|---|---|---|---|
| Wake | 0.914 | 0.90 | 0.91 |
| N1 | 0.351 | 0.35 | 0.36 |
| N2 | 0.830 | 0.83 | 0.87 |
| N3 | 0.824 | 0.93 | 0.90 |
| REM | 0.705 | 0.76 | 0.84 |

---

## Repository Structure

```
deepstager/
├── data/                            # EDF files from PhysioNet (not included)
├── 01_xgboost_baseline.ipynb        # Feature extraction + XGBoost pipeline
├── 02_cnn_bilstm.ipynb              # CNN pre-training + BiLSTM fine-tuning
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Implementation

### Libraries
```
mne >= 1.12.1       # EDF reading, bandpass filtering
pywavelets          # CWT feature extraction
xgboost >= 2.0      # Gradient-boosted baseline
torch               # CNN + BiLSTM implementation
scikit-learn        # Cross-validation, metrics, preprocessing
numpy, scipy        # Numerical computation
matplotlib          # Visualisation
```

## Planned Work

- Finalise XGBoost on full 76-subject dataset (add Cohen's kappa)
- Retrain CNN-only and CNN + BiLSTM on full dataset (76 subjects)
- Wavelet + LSTM hybrid (structured features → recurrent model)
- Transformer encoder replacing BiLSTM for long-range context
- Interpretability analysis (filter activations, saliency maps)
- Data augmentation for N1 (Gaussian noise, temporal shifting)

---

## References

1. Supratak et al. (2017). *DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG.* IEEE TNSRE, 25(11), 1998–2008.
2. Kemp et al. (2000). *Analysis of a sleep-dependent neuronal feedback loop.* IEEE-BME, 47(9), 1185–1194.
3. Goldberger et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23).
4. Zekriyapanah Gashti & Farjamnia (2025). *EEG Sleep Stage Classification with CWT and Deep Learning.* arXiv:2510.07524.
5. Danker-Hopfe et al. (2009). *Interrater reliability for sleep scoring.* Journal of Sleep Research, 18(1), 74–84.