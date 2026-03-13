# 🔊 SonicCity — Urban Sound Classification & Noise Robustness Analysis

A complete machine learning pipeline for classifying urban environmental sounds using the **UrbanSound8K** dataset. The project covers audio loading, feature extraction, visualization, preprocessing, multi-model classification, noise robustness testing, and a novel MFCC-only feature selection study.

---

## 📁 Dataset

**UrbanSound8K** — [Kaggle Link](https://www.kaggle.com/datasets/rupakroy/urban-sound-8k)

| Property | Details |
|----------|---------|
| Total Clips | 8,732 `.wav` audio files |
| Classes | 10 urban sound categories |
| Clip Duration | Up to 4 seconds each |
| Sampling Rate | 22,050 Hz |
| Folds | 10 (pre-defined cross-validation splits) |

**Sound Classes:**
`air_conditioner` · `car_horn` · `children_playing` · `dog_bark` · `drilling` · `engine_idling` · `gun_shot` · `jackhammer` · `siren` · `street_music`

---

## 🗂️ Project Structure

```
SonicCity/
│
├── urban-audio-classification-and-noise-analysis.ipynb   ← Main notebook
├── LICENSE
└── README.md

```

---

## 🔬 Pipeline Overview

### Part A — Audio Data Understanding
- Load `.wav` files using `librosa`
- Display **sampling rate** and **duration** for selected clips
- Plot **waveforms** for 3 different sound classes

### Part B — Feature Extraction
Extracted 4 core audio features from all 8,732 files:

| Feature | What It Captures |
|---------|-----------------|
| **MFCC** (13 coefficients) | Timbral texture — how the sound "feels" |
| **Spectral Centroid** | Brightness — where frequency energy is centered |
| **Zero Crossing Rate** | Noisiness — how rapidly the signal changes sign |
| **Chroma Features** | Pitch class content — musical/tonal presence |

### Part C — Feature Visualization
- **MFCC Heatmaps** — spectral texture over time
- **Spectrograms** — frequency vs time with dB intensity
- **Feature Distribution Plots** — histogram + KDE for each feature

### Part D — Preprocessing
- Converted features into numerical vectors (`DataFrame`)
- Applied **StandardScaler** (zero mean, unit variance)
- **80/20 Train-Test Split** using `train_test_split`

### Part E — Model Training
Trained 5 classical ML classifiers on combined features:

| Classifier | Type |
|-----------|------|
| K-Nearest Neighbors (KNN) | Instance-based |
| Decision Tree | Tree-based |
| Random Forest | Ensemble (Bagging) |
| Gaussian Naive Bayes | Probabilistic |
| Support Vector Machine (SVM) | Kernel-based |

### Part F — Evaluation
Each model evaluated on:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Confusion Matrix** heatmaps

**Combined Features Results:**

| Model | Accuracy |
|-------|----------|
| Random Forest | 0.6938 |
| KNN | 0.6463 |
| Decision Tree | 0.5810 |
| SVM | 0.4911 |
| Gaussian NB | 0.3429 |

### Part G — Noise Robustness Experiment
- Injected **white noise** at **SNR = 10 dB** into all audio files
- Re-extracted features from noisy signals
- Re-trained and evaluated all 5 classifiers

**Noisy Features Results:**

| Model | Accuracy (Noisy) |
|-------|-----------------|
| Random Forest | 0.6898 |
| KNN | 0.6651 |
| Decision Tree | 0.5695 |
| SVM | 0.5106 |
| Gaussian NB | 0.3320 |

### Part H — Novel Research Extension (Feature Selection Study)
**MFCC-only vs Combined Features comparison:**

Extracted full 13-dimensional MFCC vectors (instead of just mean) and retrained all classifiers.

**MFCC-Only Results:**

| Model | Accuracy (MFCC-only) | Accuracy (Combined) |
|-------|---------------------|---------------------|
| Random Forest | **0.8987** | 0.6938 |
| KNN | **0.8867** | 0.6463 |
| SVM | **0.8180** | 0.4911 |
| Decision Tree | **0.7310** | 0.5810 |
| Gaussian NB | **0.4740** | 0.3429 |

> **Key Finding:** MFCC-only features significantly outperformed combined features across all models, suggesting that detailed MFCC coefficients carry more discriminative information than aggregated multi-feature means.

---

## 📊 Key Results

- **Best Model Overall:** `RandomForestClassifier` with MFCC-only features — **89.87% accuracy**
- **Most Noise-Robust Model:** `RandomForestClassifier` — smallest accuracy drop under noise
- **Worst Performer:** `GaussianNB` — struggled with both clean and noisy data
- **Most Confused Classes:** `engine_idling` ↔ `air_conditioner` (both produce steady broadband hum)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading, feature extraction |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting waveforms, spectrograms |
| `seaborn` | Confusion matrix heatmaps, distributions |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `soundfile` | Audio I/O backend |

---

## ⚙️ How to Run

### On Kaggle (Recommended)
1. Open the notebook on Kaggle
2. Add the [UrbanSound8K dataset](https://www.kaggle.com/datasets/rupakroy/urban-sound-8k) as input
3. Run all cells top to bottom — all libraries are pre-installed

### Locally
```bash
pip install librosa numpy pandas matplotlib seaborn scikit-learn soundfile
jupyter notebook urban-audio-classification-and-noise-analysis.ipynb
```

> **Note:** Update the `DATASET_PATH`, `CSV_PATH`, and `AUDIO_BASE` variables in Cell 1 to match your local directory structure.

---

## 🚀 Future Improvements

- **Deep Learning:** CNN or RNN on raw spectrograms for higher accuracy
- **Hyperparameter Tuning:** GridSearchCV / RandomizedSearchCV
- **Data Augmentation:** Time stretching, pitch shifting, more noise types
- **Class Imbalance Handling:** SMOTE or class weighting
- **More Features:** Spectral contrast, spectral roll-off, tonnetz, tempogram
- **Cross-Validation:** Use the 10 pre-defined folds in UrbanSound8K properly

---

## 📄 References

- J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", ACM MM 2014
- [UrbanSound8K on Kaggle](https://www.kaggle.com/datasets/rupakroy/urban-sound-8k)
- [librosa documentation](https://librosa.org/doc/latest/index.html)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
