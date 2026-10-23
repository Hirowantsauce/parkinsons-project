# Parkinson's Disease Voice Biomarker Detection: Codebase Evolution & Architecture Analysis

This document provides a comprehensive technical comparison detailing:
1. **Original State ("What the code had earlier")**: Complete audit of original scripts, logic, data leakage, and architecture.
2. **Refactoring & Fixes ("How it was changed")**: Exact changes applied to resolve mathematical, data-leakage, and pathing bugs.
3. **New Features & Pipelines ("What's new & how it works")**: Complete walkthrough of newly built feature extractors, multimodal fusion models, and datasets.

---

## 1. What the Code Had Earlier

### A. Main Machine Learning Pipeline (`Codes/Parkinsons-project.py` & Notebooks)
* **Hardcoded Machine-Specific Paths**:
  ```python
  df = pd.read_csv("D:\\Rohit\\Parkinsons\\Databases\\park-database.csv")
  ```
  The script only ran on a specific local machine username/path and failed if relocated or cloned.
* **Severe Data Leakage Prior to Train/Test Split**:
  ```python
  # ORIGINAL CODE:
  randos = RandomOverSampler()
  x_randos, y_randos = randos.fit_resample(x, y)  # <-- Oversampled on ENTIRE dataset!
  scaler = MinMaxScaler((-1, 1))
  x = scaler.fit_transform(x_randos)              # <-- Scaled on ENTIRE dataset!
  pca = PCA(.95)
  x_pca = pca.fit_transform(x)                    # <-- PCA fitted on ENTIRE dataset!
  x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=14)
  ```
  * **Consequence**: `RandomOverSampler` duplicated instances before splitting. As a result, exact clones of test samples appeared in the training set. Furthermore, `MinMaxScaler` and `PCA` learned minimums, maximums, and eigenvectors from the test set. This produced an artificially inflated accuracy score (~98.3%) that could never generalize to real patients.
* **Lack of Subject-Level Partitioning**:
  * The dataset contained 195 rows from 31 distinct individuals (`phon_R01_S01_1` to `phon_R01_S01_6` were all Subject 1).
  * Random splitting scattered the same speaker's recordings across both training and test sets. The model simply memorized the speaker's vocal pitch and timbre rather than detecting disease pathology.
* **Defective ROC-AUC Curve Calculation**:
  ```python
  # ORIGINAL CODE:
  def plot_roc(rfe, x_test, y_test):
      probabilities = model.predict_proba(x_test)  # <-- Hardcoded global 'model' (BernoulliNB)!
  ```
  * Passing any classifier (e.g., Random Forest `rfe`) was ignored because the function always called the global variable `model` (`BernoulliNB`).
* **Code Smells & Duplicate Imports**:
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.neighbors import KNeighborsClassifier
  ```
* **Unutilized Audio Data**:
  * 81 raw audio recordings in `Voice data/` remained completely disconnected from the machine learning workflow.

---

### B. Feature Extraction Script (`Codes/mfcc-extraction.py`)
* **Hardcoded Paths & Redundant Declarations**:
  ```python
  file_path = r"D:\Rohit\Parkinsons\Voice data\Combined"
  output_csv_path = r"D:\Rohit\Parkinsons\Databases\extracted_features.csv"
  directory = "D:\Rohit\Parkinsons\Voice data\Combined"
  ```
* **Unlabeled Output CSV**:
  * The script extracted 24 MFCC values but did not assign the ground-truth clinical diagnosis (`status`). Columns were unlabelled numerical indices (`0` to `23`), making the CSV unusable for classification without manual data joining.

---

## 2. How the Code Was Changed

### Comparison: Main Pipeline (`Codes/Parkinsons-project.py`)

| Aspect | Earlier State | Modified State |
| :--- | :--- | :--- |
| **Pathing** | Hardcoded `D:\Rohit\...` | Dynamic `Path(__file__).resolve().parent.parent` |
| **Imports** | Triplicated `KNeighborsClassifier` | Deduplicated, organized, added `Path`, `warnings` filter |
| **Data Splitting** | Split executed **after** oversampling & PCA | `train_test_split(stratify=y)` executed **first** on raw features |
| **Oversampling** | Fitted on entire dataset (data leakage) | `RandomOverSampler` fitted **strictly on training fold** |
| **Scaling & PCA** | Fitted on entire dataset (data leakage) | `MinMaxScaler` and `PCA` fitted **strictly on training fold** |
| **Validation Modes** | Single leaky 80/20 holdout | Both leak-free Holdout and **Subject-Wise Group Cross-Validation** (`GroupShuffleSplit`) |
| **ROC Plotting** | Hardcoded to global `model` | Dynamic: inspects `predict_proba` or `decision_function` of passed estimator |
| **Visual Outputs** | Displayed via blocking `plt.show()` | Automatically saved as PNGs to `Outputs/` with non-blocking execution |

#### Code Transformation Diff:
```python
# ==========================================
# REFACTORED LEAK-FREE PIPELINE:
# ==========================================
# 1. Split FIRST on raw data
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=14, stratify=y
)

# 2. Resample ONLY training fold
ros = RandomOverSampler(random_state=14)
X_train_res, y_train_res = ros.fit_resample(X_train_raw, y_train_raw)

# 3. Fit Scaler ONLY on training fold; transform test fold
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test_raw)

# 4. Fit PCA ONLY on training fold; transform test fold
pca = PCA(n_components=0.95, random_state=14)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

---

## 3. What's New and How It Works

Three major software systems, two clinical databases, and automated visualization pipelines were created from scratch:

```
                  ┌──────────────────────────────────────────────┐
                  │          Raw Voice Recordings (.wav)         │
                  │   41 Healthy Controls + 40 Parkinson's Pts   │
                  └──────────────────────┬───────────────────────┘
                                         │
                         Codes/extract_voice_features.py
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
┌───────────────────────────────┐                       ┌───────────────────────────────────┐
│  Biomedical MDVP Features     │                       │  Dynamic Cepstral Variances       │
│  Pitch, Jitter, Shimmer, HNR  │                       │  MFCC_vars, LAR_vars, Demographics│
└───────────────┬───────────────┘                       └─────────────────┬─────────────────┘
                │                                                         │
                ▼                                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                          Codes/multimodal_voice_model.py                                  │
│             Robust Scaling  ──►  SelectKBest (k=20)  ──►  Multimodal Soft Ensemble         │
│               Random Forest + Extra Trees + HistGradientBoosting + SVC (RBF)              │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
                   5-Fold Cross-Validation Accuracy: 81.5% | ROC-AUC: 0.896
```

---

### New Component 1: Clinical Acoustic Feature Extractor
* **File**: [Codes/extract_voice_features.py](file:///d:/AG/Parkinsons/Codes/extract_voice_features.py)
* **Technology**: Python with `praat-parselmouth` (C++ Praat engine bindings), `librosa`, and `scipy`.
* **How It Works**:
  1. Iterates over all 81 raw audio recordings in `Voice data/Combined/`.
  2. Extracts all acoustic measurements matching the Oxford/UCI Parkinson's dataset:
     * **Fundamental Frequency**: Mean pitch ($F_0$), Maximum ($F_{hi}$), Minimum ($F_{lo}$) using parabolic acoustic interpolation.
     * **5 Jitter Measures**: Local Jitter (%), Absolute Jitter, Relative Average Perturbation (RAP), 5-point Period Perturbation Quotient (PPQ5), and Difference of Differences (DDP).
     * **6 Shimmer Measures**: Local Shimmer, Local dB Shimmer, 3-point APQ3, 5-point APQ5, 11-point APQ11, and Difference of Differences (DDA).
     * **Harmonicity & Noise**: Harmonics-to-Noise Ratio (HNR in dB) and Noise-to-Harmonics Ratio (NHR).
     * **Non-Linear Dynamics**:
       * *RPDE (Recurrence Period Density Entropy)*: Normalized Shannon entropy of pitch period distributions.
       * *DFA (Detrended Fluctuation Analysis)*: Long-range fractal scaling exponent $\alpha$.
       * *PPE (Pitch Period Entropy)*: Semitone pitch period variability entropy.
       * *D2 (Correlation Dimension)*: Embedding phase-space attractor dimension.
       * *spread1 & spread2*: Logarithmic and interquartile fundamental frequency variation.
  3. Maps filenames to clinical cohort labels: Healthy Controls (`0`) vs Parkinson's (`1`).
  4. Cross-references demographic metadata (`Age`, `Sex`) from `Demographics_age_sex.xlsx`.
  5. Exports to:
     * [Databases/voice-extracted-database.xlsx](file:///d:/AG/Parkinsons/Databases/voice-extracted-database.xlsx) (with 3 formatted sheets: `Biomedical_Acoustic_Data`, `MFCC_24_Features`, and `Subject_Demographics`).
     * [Databases/voice-extracted-database.csv](file:///d:/AG/Parkinsons/Databases/voice-extracted-database.csv).

---

### New Component 2: Extracted Voice Evaluation Engine
* **File**: [Codes/evaluate_voice_dataset.py](file:///d:/AG/Parkinsons/Codes/evaluate_voice_dataset.py)
* **How It Works**:
  * Evaluates 8 classifiers + Voting ensembles strictly on the 81 newly extracted voice recordings.
  * Discovered the scientific limitation of 8 kHz mobile audio: because telephone-bandwidth audio has a $125\ \mu\text{s}$ sample resolution, micro-second vocal jitter ($10\text{--}40\ \mu\text{s}$) is obscured by digital quantization noise, causing pure jitter/shimmer models to reach ~58-64% accuracy.

---

### New Component 3: Multimodal Fusion Model
* **File**: [Codes/multimodal_voice_model.py](file:///d:/AG/Parkinsons/Codes/multimodal_voice_model.py)
* **How It Works**:
  1. **Feature Merging**: Combines 44 candidate biomarkers across 3 distinct modalities:
     * *Classical Acoustic Measures*: $F_0$, $F_{lo}$, DFA, RPDE.
     * *Dynamic Vocal Tract Variances*: `MFCC_vars` (frame-by-frame variance of Mel-frequency filterbanks) and `LAR_vars` (Log Area Ratio variances of vocal tract reflection coefficients).
     * *Clinical Covariates*: Patient `Age` and `Sex`.
  2. **Preprocessing Pipeline**:
     * `RobustScaler`: Outlier-resistant median and interquartile range scaling.
     * `SelectKBest(f_classif, k=20)`: Retains the top 20 most discriminatory biomarkers.
  3. **Soft-Voting Ensemble**:
     * Ensembles 4 high-capacity classifiers: Random Forest + Extra Trees + HistGradientBoosting + Support Vector Classifier (RBF kernel).
     * Probability predictions are averaged, stabilizing individual fold variances.
  4. **Performance Leap**:
     * Accuracy increased from **58.0%** to **81.5%** on 5-fold cross-validation.
     * ROC-AUC surged from **0.64** to **0.896** (nearly 0.90).
  5. **Outputs Saved**:
     * [Databases/fused-multimodal-database.xlsx](file:///d:/AG/Parkinsons/Databases/fused-multimodal-database.xlsx) & [Databases/fused-multimodal-database.csv](file:///d:/AG/Parkinsons/Databases/fused-multimodal-database.csv)
     * [Outputs/multimodal_model_comparison.png](file:///d:/AG/Parkinsons/Outputs/multimodal_model_comparison.png)
     * [Outputs/multimodal_roc_curve.png](file:///d:/AG/Parkinsons/Outputs/multimodal_roc_curve.png)
     * [Outputs/multimodal_feature_importance.png](file:///d:/AG/Parkinsons/Outputs/multimodal_feature_importance.png)

---

## 4. Comprehensive Performance Summary

| Pipeline / Dataset | Evaluation Method | Accuracy | ROC-AUC | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Original Code** (`park-database.csv`) | Holdout (with Data Leakage) | 98.3% | — | Artificially high due to pre-split oversampling |
| **Refactored Code** (`park-database.csv`) | Holdout (Strictly Leak-Free) | 92.3% | 0.960 | High performance on 44.1 kHz laboratory audio |
| **Refactored Code** (`park-database.csv`) | Subject-Wise Group CV | 83.6% | — | True generalization on unseen laboratory patients |
| **Pure Voice Extracted** (8 kHz audio) | Holdout Test (Leak-Free) | 64.7% | 0.708 | Jitter/shimmer attenuated by 8 kHz sampling rate |
| **Pure Voice Extracted** (8 kHz audio) | 5-Fold Stratified CV | 58.0% | 0.639 | Classical biomedical features alone |
| **MFCC Means Alone** (8 kHz audio) | 5-Fold Stratified CV | 56.8% | 0.580 | Static mean spectral features |
| **Multimodal Fused Model** (8 kHz audio) | 5-Fold Stratified CV | **81.5%** | **0.896** | **Fused dynamic cepstral variance + demographics** |
| **Multimodal Fused Model** (8 kHz audio) | Holdout Test (Leak-Free) | **76.5%** | **0.889** | **Precision: 75.0% \| Recall: 75.0%** |
