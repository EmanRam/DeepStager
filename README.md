# DeepStager: Automatic Multi-Stage Sleep Scoring from Raw Single-Channel EEG

> **CISC 867 Final Project — Queen's University, School of Computing**
> Eman R. Ahmed · Salma M. El-sherif 

---

## Overview

DeepStager presents a systematic comparative evaluation of **five architectures** for automated five-class AASM sleep staging from a single EEG channel (Fpz-Cz), evaluated on the Sleep-EDF Expanded (Cassette) dataset.

| # | Model | Input | Temporal Modelling |
|---|-------|-------|--------------------|
| 1 | **XGBoost Baseline** | 53-D handcrafted features | None (epoch-independent) |
| 2 | **Two-Stream CNN** | Raw EEG | None (epoch-independent) |
| 3 | **CNN + BiLSTM** | Raw EEG | Bidirectional LSTM |
| 4 | **Wavelet + BiLSTM** | 53-D CWT features | Bidirectional LSTM |
| 5 | **CNN + Transformer** | Raw EEG | Multi-head self-attention |

**Target classes:** Wake (W), N1, N2, N3, REM — 5-class AASM-compatible scoring.

---

## Results Summary

### Held-Out Test Set Performance (31 recordings, ~38K epochs)

| Model | Test Accuracy | Macro F1 | Cohen's κ | κ Category | N1 F1 |
|-------|:---:|:---:|:---:|:---:|:---:|
| XGBoost (baseline) | 73.87% | 0.7026 | 0.6547 | Substantial | 0.37 |
| Two-Stream CNN | 82.32% | 0.7164 | 0.7508 | Substantial | 0.17 ▼ |
| CNN + BiLSTM | 83.30% | 0.7618 | 0.7650 | Substantial | 0.36 |
| Wavelet + BiLSTM | 79.74% | 0.7733 | 0.7341 | Substantial | **0.52 ★** |
| **CNN + Transformer** | **84.03%** | **0.7830** | **0.7778** | **Excellent** | 0.44 |

**Key finding:** N1 classification remains persistently challenging across all models (F1: 0.17–0.52), reflecting the absence of unambiguous spectral markers in single-channel EEG for this transitional stage.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [Sleep-EDF Expanded (Cassette) — PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) |
| **Subjects** | 78 (153 recordings, two nights per subject) |
| **Channel** | Fpz-Cz EEG, sampled at 100 Hz |
| **Epoch length** | 30 seconds (3,000 samples) |
| **Total epochs** | 195,469 labelled epochs after wake-trimming |
| **Split** | 80/20 subject-disjoint (122 train / 31 test recordings) |

### Class Distribution (153 recordings)

| Stage | Epochs | % |
|-------|-------:|--:|
| Wake (W) | ~53K | 27.1% |
| N1 | ~21.5K | 11.0% |
| N2 | ~69K | 35.4% |
| N3 | ~24K | 12.3% |
| REM | ~28K | 14.2% |

---

## Repository Structure

```
DeepStager/
├── data/                              # EDF files from PhysioNet (not tracked)
├── notebooks/
│   ├── 01_xgboost_baseline.ipynb      # Feature extraction + XGBoost pipeline
│   ├── 02_cnn_bilstm.ipynb            # CNN pre-training + BiLSTM fine-tuning
│   ├── 03_wavelet_bilstm.ipynb        # Wavelet + BiLSTM hybrid
│   └── 04_cnn_transformer.ipynb       # CNN + Transformer architecture
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/EmanRam/DeepStager.git
cd DeepStager
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

Or install directly:

```bash
pip install mne>=1.12.1 PyWavelets xgboost>=2.0 torch scikit-learn numpy scipy matplotlib seaborn
```

### 4. Download the Dataset

```bash
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```

Place the downloaded `.edf` and `.edf+` files in the `data/` directory.

> **Note:** The dataset is approximately 25 GB. Ensure sufficient disk space before downloading.

---

## Running the Notebooks

All notebooks are designed to run sequentially. Open them in Jupyter or directly in **Google Colab**.

| Notebook | Description | GPU Required |
|----------|-------------|:---:|
| `01_xgboost_baseline.ipynb` | CWT feature extraction (53-D) + XGBoost training & 5-fold CV | No |
| `02_cnn_bilstm.ipynb` | Stage 1 CNN pre-training + Stage 2 BiLSTM fine-tuning | Yes |
| `03_wavelet_bilstm.ipynb` | Wavelet + BiLSTM hybrid, hyperparameter trials, CWT-40 ablation | Yes |
| `04_cnn_transformer.ipynb` | CNN + Transformer with transfer learning + gradient clipping ablation | Yes |

```bash
jupyter notebook
```

> **Google Colab:** For deep learning notebooks, enable GPU under `Runtime → Change runtime type → T4 GPU`. All experiments in this project were conducted on an NVIDIA T4 (16 GB VRAM, 12 GB RAM).

---

## Methodology

### Preprocessing (all models)

1. Load PSG–Hypnogram pairs using MNE-Python
2. Resample to **100 Hz**
3. Apply zero-phase **Hamming-windowed FIR bandpass filter** (0.5–40 Hz)
4. **Wake-trimming:** discard epochs >30 min before first / after last non-wake stage; retain intra-night arousals
5. Exclude `Movement Time` and `Unknown (?)` epochs
6. Segment into non-overlapping **30-second epochs** (3,000 samples each)

### Model 1 — XGBoost Baseline

Each epoch is encoded as a **53-dimensional feature vector** from three domains:

| Domain | Features | Count |
|--------|----------|------:|
| CWT Sub-band Statistics (Morlet, 5 bands × 8 stats) | Mean, Std, Energy, Spectral Entropy, Skewness, Kurtosis, Max, Min | 40 |
| Time-Domain | ZCR, RMS, MAV, Waveform Length, Log-Variance, Hjorth Activity/Mobility/Complexity | 8 |
| Welch Relative Spectral Power | Delta, Theta, Alpha, Sigma, Beta band ratios | 5 |

- **Classifier:** XGBoost (200 trees, depth 6, lr 0.1, subsample 0.8, inverse-frequency class weighting)
- **Evaluation:** Subject-disjoint 80/20 split + 5-fold stratified cross-validation on training partition

### Models 2–3 — CNN + BiLSTM (DeepSleepNet-inspired)

**Stage 1 — Dual-stream CNN pre-training:**
- Small-filter stream: kernel 50, stride 6 → captures sleep spindles & K-complexes
- Large-filter stream: kernel 400, stride 50 → captures slow-wave morphology
- Each stream: 3× ConvBlocks (Conv1D → BN → ReLU → MaxPool → Dropout 50%) → AdaptiveMaxPool → 128-D
- Concatenated to 256-D → Linear classifier (5 classes)
- Training: Adam (lr=1×10⁻⁴), 30 epochs, batch 128

**Stage 2 — BiLSTM fine-tuning:**
- Input: sequences of 25 consecutive epochs
- 2× stacked BiLSTM layers (512 units/direction → 1,024-D output)
- Residual shortcut: CNN output (256-D) projected to 1,024-D, added to BiLSTM output
- Two-LR strategy: CNN lr=1×10⁻⁶, BiLSTM lr=1×10⁻⁴; gradient clipping (norm=1.0)
- Training: 30 epochs, batch 16 sequences; 5-fold subject-level CV

### Model 4 — Wavelet + BiLSTM Hybrid

- Input: 53-D CWT feature vectors (pre-computed), sequences of 25 epochs
- Linear projection → 64-D latent space (LayerNorm + ReLU + Dropout)
- 2× BiLSTM layers (128 units/direction → 256-D output)
- Best config: Adam (lr=1×10⁻³), StepLR (step=10, γ=0.5), dropout=0.3
- Ablation: 53-D full features vs. 40-D CWT-only (Weighted CE vs. Focal Loss)

### Model 5 — CNN + Transformer

- Identical dual-stream CNN encoder (weights transferred from best CNN-only fold)
- Sinusoidal positional encodings injected before self-attention
- 4-layer Transformer encoder (8 attention heads, d_model=256, FFN dim=1,024, dropout=0.1)
- Two-LR strategy: CNN lr=1×10⁻⁶, Transformer lr=1×10⁻⁴; gradient clipping (norm=1.0)
- Training: 30 epochs CV + 10 epochs full retrain; fold weights averaged for initialisation

---

## Ablation Studies

### 1. Wavelet + BiLSTM: Input Dimensionality

| Config | Input Dim | Loss Function | CV Accuracy | Test Accuracy | Macro F1 |
|--------|:---------:|---------------|:-----------:|:-------------:|:--------:|
| Full-53 (Trial 2) ★ | 53 | Weighted CrossEntropy | 79.33% | 79.74% | 0.7733 |
| CWT-40 Trial 1 | 40 | Weighted CrossEntropy | 75.52% | 77.28% | 0.7459 |
| CWT-40 Trial 2 | 40 | Focal Loss (γ=2.0) | 61.97% | 60.14% | 0.6173 |

> Time-domain descriptors and Welch features contribute non-redundant discriminative information. Focal Loss is destabilising in the low-dimensional feature regime.

### 2. CNN + Transformer: Gradient Clipping Norm

| Trial | Clip Norm | CV Accuracy | CV Macro F1 | Test Accuracy | Test κ |
|-------|:---------:|:-----------:|:-----------:|:-------------:|:------:|
| Trial 1 ★ | 1.0 | 82.22% ± 1.99% | 0.7430 | **84.03%** | **0.7778** |
| Trial 2 | 10.0 | 80.11% ± 2.34% | 0.7218 | 82.47% | 0.7581 |

> Tighter gradient clipping is essential to stabilise Transformer training when fine-tuning alongside a pretrained CNN backbone.

---

## Key Findings

1. **Temporal context modelling is essential.** Every model with a sequence encoder (BiLSTM or Transformer) outperforms the epoch-independent CNN by at least +0.98% accuracy and +0.45 macro F1.

2. **Self-attention provides marginal gains over recurrence** within a 12.5-minute window. CNN + Transformer outperforms CNN + BiLSTM by only 0.73% accuracy, suggesting that longer context windows spanning full ultradian cycles (~90 min) may be required for larger gains.

3. **Structured wavelet features offer a superior N1 recall trade-off.** The Wavelet + BiLSTM achieves the highest N1 F1 (0.52) of any model, at the cost of lower overall accuracy — demonstrating that explicit spectral decomposition retains N1-discriminative texture that end-to-end CNN embeddings suppress.

4. **Single-channel EEG cannot reliably resolve N1.** N1 F1 ranges from 0.17 to 0.52 across all five architectures, reflecting the absence of unambiguous spectral markers and the unavailability of EOG/EMG modalities used by human scorers.

---

## Implementation Details

### Libraries

| Library | Version | Role |
|---------|---------|------|
| `mne` | ≥ 1.12.1 | EDF loading, resampling, bandpass filtering |
| `pywavelets` | ≥ 1.6.0 | Morlet CWT feature extraction |
| `xgboost` | ≥ 2.1 | Gradient-boosted baseline classifier |
| `torch` | ≥ 2.2 | CNN, BiLSTM, Transformer implementations |
| `scikit-learn` | ≥ 1.4 | GroupShuffleSplit, StratifiedKFold, metrics |
| `numpy` / `scipy` | — | Numerical computation |
| `matplotlib` / `seaborn` | — | Visualisation |

## Future Work

- [ ] Multi-channel fusion: incorporate Pz-Oz EEG, EOG, and EMG channels
- [ ] Longer context windows (>25 epochs) for full ultradian cycle coverage
- [ ] Subject-adaptive fine-tuning via meta-learning / test-time adaptation
- [ ] Alternative minority-class loss functions (asymmetric loss, distribution-based re-weighting)
- [ ] Cross-dataset evaluation on SHHS and MASS benchmarks
- [ ] Attention map visualisation for clinical interpretability
- [ ] Edge deployment with ONNX quantisation (target: Raspberry Pi 5)

---

## References

1. Supratak et al. (2017). DeepSleepNet. *IEEE TNSRE*, 25(11), 1998–2008.
2. Lee-Iannotti (2023). Sleep disorders in neurologic disease. *Continuum*, 29(4).
3. Kemp et al. (2000). Sleep-EDF dataset. *IEEE-BME*, 47(9).
4. Rechtschaffen & Kales (1968). *Manual of Standardized Terminology for Sleep Staging*.
5. Danker-Hopfe et al. (2009). Interrater reliability for sleep scoring. *J. Sleep Res.*, 18(1).
6. Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
7. Zekriyapanah Gashti & Farjamnia (2025). EEG Sleep Stage Classification with CWT. *arXiv:2510.07524*.
8. Phan et al. (2022). SleepTransformer. *IEEE TBME*, 69(8), 2456–2467.
9. Tsinalis et al. (2016). Automatic sleep staging with CNNs. *arXiv:1610.01683*.
10. Mousavi et al. (2019). SleepEEGNet. *PLoS One*, 14(5).
11. Li et al. (2022). EEGSNet. *Int. J. Environ. Res. Public Health*, 19(10).
12. Vaswani et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

---
