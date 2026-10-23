# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

# %%
# Resolve dynamic project paths
BASE_DIR = Path(__file__).resolve().parent.parent
COMBINED_AUDIO_DIR = BASE_DIR / "Voice data" / "Combined"
HC_AUDIO_DIR = BASE_DIR / "Voice data" / "seperated" / "HC_AH"
PD_AUDIO_DIR = BASE_DIR / "Voice data" / "seperated" / "PD_AH"
OUTPUT_CSV_PATH = BASE_DIR / "Databases" / "extracted_features.csv"

# %%
def get_cohort_labels():
    """Maps filenames to health status (1 = Parkinson's, 0 = Healthy Control)."""
    labels = {}
    if HC_AUDIO_DIR.exists():
        for f in os.listdir(HC_AUDIO_DIR):
            if f.lower().endswith(".wav"):
                labels[f] = 0
    if PD_AUDIO_DIR.exists():
        for f in os.listdir(PD_AUDIO_DIR):
            if f.lower().endswith(".wav"):
                labels[f] = 1
    return labels

# %%
def feature_extraction(audio_path, n_mfcc=24):
    """Loads a WAV audio file and computes mean MFCCs across time."""
    y, sample_rate = librosa.load(str(audio_path), res_type="kaiser_fast")
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
    return mfcc

# %%
def extract_all_features():
    if not COMBINED_AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {COMBINED_AUDIO_DIR}")

    cohort_labels = get_cohort_labels()
    audio_files = [f for f in os.listdir(COMBINED_AUDIO_DIR) if f.lower().endswith(".wav")]
    print(f"Found {len(audio_files)} audio files in: {COMBINED_AUDIO_DIR}")

    features = {}
    statuses = {}

    for audio_file in audio_files:
        audio_path = COMBINED_AUDIO_DIR / audio_file
        try:
            features[audio_file] = feature_extraction(audio_path)
            # Assign status if known from cohort separation
            statuses[audio_file] = cohort_labels.get(audio_file, np.nan)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    # Build DataFrame
    col_names = [f"mfcc_{i}" for i in range(1, 25)]
    df = pd.DataFrame.from_dict(features, orient="index", columns=col_names)
    df.index.name = "name"
    df["status"] = [statuses.get(idx, np.nan) for idx in df.index]

    # Ensure parent output directory exists
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH)
    print(f"Saved {len(df)} feature records to: {OUTPUT_CSV_PATH}")
    return df

# %%
if __name__ == "__main__":
    df_features = extract_all_features()
    print(df_features.head())
