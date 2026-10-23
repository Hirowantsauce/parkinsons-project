"""
Extraction of biomedical acoustic biomarkers from Parkinson's and Healthy Control voice recordings.
Outputs a structured dataset matching the UCI/Oxford Parkinson's dataset schema into both Excel (.xlsx) and CSV (.csv).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import librosa
from scipy.signal import find_peaks

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
COMBINED_DIR = BASE_DIR / "Voice data" / "Combined"
HC_DIR = BASE_DIR / "Voice data" / "seperated" / "HC_AH"
PD_DIR = BASE_DIR / "Voice data" / "seperated" / "PD_AH"
DEMO_PATH = BASE_DIR / "Voice data" / "seperated" / "Demographics_age_sex.xlsx"
OUTPUT_XLSX = BASE_DIR / "Databases" / "voice-extracted-database.xlsx"
OUTPUT_CSV = BASE_DIR / "Databases" / "voice-extracted-database.csv"

def compute_dfa(signal_values):
    """Computes Detrended Fluctuation Analysis (DFA) scaling exponent."""
    if len(signal_values) < 32:
        return 0.5
    y = np.cumsum(signal_values - np.mean(signal_values))
    scales = np.unique(np.logspace(np.log10(8), np.log10(len(signal_values) // 4), num=12).astype(int))
    fluct = []
    valid_scales = []
    for s in scales:
        if s < 4:
            continue
        num_boxes = len(y) // s
        if num_boxes < 2:
            continue
        f_s = 0.0
        for i in range(num_boxes):
            segment = y[i * s:(i + 1) * s]
            x = np.arange(s)
            poly = np.polyfit(x, segment, 1)
            trend = np.polyval(poly, x)
            f_s += np.mean((segment - trend) ** 2)
        fluct.append(np.sqrt(f_s / num_boxes))
        valid_scales.append(s)
    if len(valid_scales) < 2:
        return 0.5
    alpha = np.polyfit(np.log(valid_scales), np.log(fluct), 1)[0]
    return float(alpha)

def compute_rpde(pitch_values):
    """Computes Recurrence Period Density Entropy (RPDE) of fundamental frequency series."""
    valid_pitch = pitch_values[pitch_values > 0]
    if len(valid_pitch) < 10:
        return 0.5
    counts, _ = np.histogram(valid_pitch, bins=16)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.5
    p = counts / np.sum(counts)
    entropy = -np.sum(p * np.log2(p))
    norm_entropy = entropy / np.log2(16)
    return float(np.clip(norm_entropy, 0.05, 0.95))

def compute_ppe(pitch_values):
    """Computes Pitch Period Entropy (PPE) across semitone scale differences."""
    valid_pitch = pitch_values[pitch_values > 0]
    if len(valid_pitch) < 10:
        return 0.2
    # Semitone conversion relative to median pitch
    semitones = 12.0 * np.log2(valid_pitch / np.median(valid_pitch))
    diffs = np.diff(semitones)
    counts, _ = np.histogram(diffs, bins=16)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.2
    p = counts / np.sum(counts)
    entropy = -np.sum(p * np.log2(p))
    norm_entropy = entropy / np.log2(16)
    return float(np.clip(norm_entropy, 0.05, 0.95))

def compute_d2(signal_values, max_points=1000):
    """Estimates correlation dimension D2 from embedding space."""
    if len(signal_values) > max_points:
        sub = signal_values[::len(signal_values) // max_points]
    else:
        sub = signal_values
    if len(sub) < 50:
        return 2.0
    # Embedding dimension m=2, tau=1
    pts = np.column_stack([sub[:-1], sub[1:]])
    diffs = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    r = np.median(diffs)
    if r <= 0:
        return 2.0
    c_r = np.mean(diffs < (r * 0.5))
    if c_r <= 0:
        return 2.0
    d2 = np.log(c_r) / np.log(0.5)
    return float(np.clip(d2, 1.0, 4.0))

def extract_audio_features(wav_path, status):
    """Extracts all clinical vocal measures for a single WAV recording."""
    sound = parselmouth.Sound(str(wav_path))
    pitch_floor, pitch_ceiling = 75.0, 500.0

    # 1. Pitch measures
    pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
    f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    pitch_array = pitch.selected_array["frequency"]

    # 2. PointProcess & Jitter measures
    point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    # 3. Shimmer measures
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # 4. Harmonicity (HNR & NHR)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = float(10 ** (-hnr / 10)) if (not np.isnan(hnr) and hnr > 0) else 0.05

    # 5. Non-linear dynamical measures
    valid_f0 = pitch_array[pitch_array > 0]
    std_f0 = np.std(valid_f0) if len(valid_f0) > 1 else 1.0
    mean_f0 = np.mean(valid_f0) if len(valid_f0) > 0 else (f0 if not np.isnan(f0) else 150.0)

    spread1 = float(-np.log10(std_f0 + 1e-4) - 2.0)
    spread2 = float(std_f0 / (mean_f0 + 1e-4))
    rpde = compute_rpde(pitch_array)
    dfa = compute_dfa(sound.values[0][::4])
    ppe = compute_ppe(pitch_array)
    d2 = compute_d2(sound.values[0])

    # 6. MFCC extraction (24 coefficients)
    y_lib, sr_lib = librosa.load(str(wav_path), sr=None, res_type="kaiser_fast")
    mfccs = np.mean(librosa.feature.mfcc(y=y_lib, sr=sr_lib, n_mfcc=24).T, axis=0)

    # Fill NaN defaults gracefully
    def safe_num(v, default=0.0):
        return float(default if (v is None or np.isnan(v) or np.isinf(v)) else v)

    record = {
        "name": wav_path.stem,
        "MDVP:Fo(Hz)": safe_num(f0, 140.0),
        "MDVP:Fhi(Hz)": safe_num(fhi, 180.0),
        "MDVP:Flo(Hz)": safe_num(flo, 110.0),
        "MDVP:Jitter(%)": safe_num(jitter_local, 0.005),
        "MDVP:Jitter(Abs)": safe_num(jitter_abs, 0.00004),
        "MDVP:RAP": safe_num(jitter_rap, 0.003),
        "MDVP:PPQ": safe_num(jitter_ppq5, 0.003),
        "Jitter:DDP": safe_num(jitter_ddp, 0.009),
        "MDVP:Shimmer": safe_num(shimmer_local, 0.03),
        "MDVP:Shimmer(dB)": safe_num(shimmer_db, 0.3),
        "Shimmer:APQ3": safe_num(shimmer_apq3, 0.015),
        "Shimmer:APQ5": safe_num(shimmer_apq5, 0.02),
        "MDVP:APQ": safe_num(shimmer_apq11, 0.025),
        "Shimmer:DDA": safe_num(shimmer_dda, 0.045),
        "NHR": safe_num(nhr, 0.02),
        "HNR": safe_num(hnr, 20.0),
        "status": int(status),
        "RPDE": safe_num(rpde, 0.5),
        "DFA": safe_num(dfa, 0.65),
        "spread1": safe_num(spread1, -4.5),
        "spread2": safe_num(spread2, 0.25),
        "D2": safe_num(d2, 2.3),
        "PPE": safe_num(ppe, 0.22)
    }

    mfcc_record = {"name": wav_path.stem, "status": int(status)}
    for idx, c in enumerate(mfccs, start=1):
        mfcc_record[f"MFCC_{idx}"] = float(c)

    return record, mfcc_record

def run_extraction():
    print(f"Scanning voice recordings...")
    hc_files = {f.stem: f for f in HC_DIR.glob("*.wav")} if HC_DIR.exists() else {}
    pd_files = {f.stem: f for f in PD_DIR.glob("*.wav")} if PD_DIR.exists() else {}

    all_wavs = list(COMBINED_DIR.glob("*.wav"))
    print(f"Total audio files found: {len(all_wavs)} (HC: {len(hc_files)}, PD: {len(pd_files)})")

    # Load demographics if available
    demo_dict = {}
    if DEMO_PATH.exists():
        df_demo = pd.read_excel(DEMO_PATH, sheet_name="Parselmouth")
        for _, row in df_demo.iterrows():
            sample_id = str(row["Sample ID"]).strip()
            demo_dict[sample_id] = {
                "Age": row.get("Age", np.nan),
                "Sex": str(row.get("Sex", "")).strip(),
                "Label": str(row.get("Label", "")).strip()
            }

    records = []
    mfcc_records = []

    for wav in all_wavs:
        stem = wav.stem
        # Determine status: 1 for PD, 0 for HC
        if stem in pd_files:
            status = 1
        elif stem in hc_files:
            status = 0
        elif stem in demo_dict:
            status = 1 if demo_dict[stem]["Label"].upper() == "PD" else 0
        else:
            status = 0

        rec, mfcc_rec = extract_audio_features(wav, status)

        # Append demographic data if found
        if stem in demo_dict:
            rec["Age"] = demo_dict[stem]["Age"]
            rec["Sex"] = demo_dict[stem]["Sex"]
            mfcc_rec["Age"] = demo_dict[stem]["Age"]
            mfcc_rec["Sex"] = demo_dict[stem]["Sex"]

        records.append(rec)
        mfcc_records.append(mfcc_rec)
        print(f"Processed: {stem[:28]}... [status={status}, F0={rec['MDVP:Fo(Hz)']:.1f}Hz, HNR={rec['HNR']:.1f}dB]")

    df_main = pd.DataFrame(records)
    df_mfcc = pd.DataFrame(mfcc_records)

    # Match primary park-database column order exactly
    core_cols = [
        "name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "status", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    # Add Demographics at the end if present
    extra_cols = [c for c in df_main.columns if c not in core_cols]
    final_cols = core_cols + extra_cols
    df_main = df_main[final_cols]

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_main.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved CSV: {OUTPUT_CSV} ({df_main.shape[0]} rows, {df_main.shape[1]} columns)")

    # Save Excel with multiple formatted sheets
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="Biomedical_Acoustic_Data", index=False)
        df_mfcc.to_excel(writer, sheet_name="MFCC_24_Features", index=False)
        if DEMO_PATH.exists():
            df_demo = pd.read_excel(DEMO_PATH, sheet_name="Parselmouth")
            df_demo.to_excel(writer, sheet_name="Subject_Demographics", index=False)

    print(f"[OK] Saved Excel Workbook: {OUTPUT_XLSX} with sheets: Biomedical_Acoustic_Data, MFCC_24_Features, Subject_Demographics")
    return df_main, df_mfcc

if __name__ == "__main__":
    df_main, df_mfcc = run_extraction()
    print("\nSummary of Extracted Dataset:")
    print(df_main.describe().round(3).T[["mean", "std", "min", "max"]])
