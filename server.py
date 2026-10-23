"""
NeuroVoice AI - Web Server and Clinical Biomarker Diagnostic API
Serves the web application and handles real-time voice acoustic prediction.
"""

import os
import sys
import json
import mimetypes
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import tempfile
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

import parselmouth
from parselmouth.praat import call
import librosa

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
FUSED_CSV = BASE_DIR / "Databases" / "fused-multimodal-database.csv"
COMBINED_AUDIO_DIR = BASE_DIR / "Voice data" / "Combined"
OUTPUT_DIR = BASE_DIR / "Outputs"

# Global Model & Data Cache
MODEL_PIPELINE = None
FUSED_DF = None
FEATURE_COLS = None
SAMPLE_LIST = []

def init_multimodal_model():
    """Initializes and pre-trains the Multimodal Soft-Voting Ensemble in memory."""
    global MODEL_PIPELINE, FUSED_DF, FEATURE_COLS, SAMPLE_LIST

    if not FUSED_CSV.exists():
        print(f"Warning: {FUSED_CSV} not found.")
        return

    FUSED_DF = pd.read_csv(FUSED_CSV)
    drop_cols = ["name", "status", "Sex"]
    FEATURE_COLS = [c for c in FUSED_DF.columns if c not in drop_cols]

    X = FUSED_DF[FEATURE_COLS].fillna(0)
    y = FUSED_DF["status"]

    rf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=14)
    et = ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=14)
    hgb = HistGradientBoostingClassifier(max_iter=100, random_state=14)
    svc = SVC(C=1.5, kernel="rbf", probability=True, random_state=14)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb), ("svc", svc)],
        voting="soft"
    )

    MODEL_PIPELINE = Pipeline([
        ("scaler", RobustScaler()),
        ("selector", SelectKBest(f_classif, k=min(20, len(FEATURE_COLS)))),
        ("clf", ensemble)
    ])

    MODEL_PIPELINE.fit(X, y)
    print("Multimodal diagnostic model trained and ready in memory.")

    # Prepare Sample List
    SAMPLE_LIST = []
    for _, row in FUSED_DF.iterrows():
        sample_name = str(row["name"])
        wav_name = f"{sample_name}.wav" if not sample_name.endswith(".wav") else sample_name
        wav_path = COMBINED_AUDIO_DIR / wav_name
        if wav_path.exists():
            SAMPLE_LIST.append({
                "name": sample_name,
                "filename": wav_name,
                "label": "Parkinson's Disease" if row["status"] == 1 else "Healthy Control",
                "status": int(row["status"]),
                "age": float(row["Age"]) if not np.isnan(row.get("Age", np.nan)) else None,
                "sex": str(row.get("Sex", "U")),
                "f0": round(float(row.get("MDVP:Fo(Hz)", 0)), 1),
                "hnr": round(float(row.get("HNR", 0)), 1)
            })
    print(f"Loaded {len(SAMPLE_LIST)} verified sample audio files for web testing.")

def extract_features_from_wav(wav_path, age=60.0, sex="M"):
    """Extracts clinical vocal measures + dynamic MFCC variances from an uploaded WAV."""
    sound = parselmouth.Sound(str(wav_path))
    pitch_floor, pitch_ceiling = 75.0, 500.0

    # 1. Pitch
    pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
    f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    pitch_array = pitch.selected_array["frequency"]

    # 2. PointProcess
    point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    # 3. Shimmer
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # 4. Harmonicity
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = float(10 ** (-hnr / 10)) if (not np.isnan(hnr) and hnr > 0) else 0.05

    # 5. Dynamics
    valid_f0 = pitch_array[pitch_array > 0]
    std_f0 = np.std(valid_f0) if len(valid_f0) > 1 else 1.0
    mean_f0 = np.mean(valid_f0) if len(valid_f0) > 0 else (f0 if not np.isnan(f0) else 150.0)

    spread1 = float(-np.log10(std_f0 + 1e-4) - 2.0)
    spread2 = float(std_f0 / (mean_f0 + 1e-4))

    # RPDE
    counts_rpde, _ = np.histogram(pitch_array[pitch_array > 0], bins=16)
    p_rpde = counts_rpde / max(1, np.sum(counts_rpde))
    p_rpde = p_rpde[p_rpde > 0]
    rpde = float(-np.sum(p_rpde * np.log2(p_rpde)) / np.log2(16)) if len(p_rpde) > 0 else 0.5

    # DFA
    sig = sound.values[0]
    dfa = 0.65
    if len(sig) > 64:
        y = np.cumsum(sig[::4] - np.mean(sig[::4]))
        scales = [8, 16, 32, 64]
        flucts = [np.std(y[:s * (len(y) // s)].reshape(-1, s), axis=1).mean() for s in scales if s < len(y) // 2]
        if len(flucts) >= 2:
            dfa = float(np.polyfit(np.log(scales[:len(flucts)]), np.log(flucts), 1)[0])

    # PPE
    semitones = 12.0 * np.log2((valid_f0 + 1e-3) / (np.median(valid_f0) + 1e-3)) if len(valid_f0) > 10 else np.array([0.0])
    counts_ppe, _ = np.histogram(np.diff(semitones), bins=16)
    p_ppe = counts_ppe / max(1, np.sum(counts_ppe))
    p_ppe = p_ppe[p_ppe > 0]
    ppe = float(-np.sum(p_ppe * np.log2(p_ppe)) / np.log2(16)) if len(p_ppe) > 0 else 0.3

    # D2
    d2 = 2.3

    # MFCC frame-by-frame variance
    y_lib, sr_lib = librosa.load(str(wav_path), sr=None, res_type="kaiser_fast")
    mfcc_matrix = librosa.feature.mfcc(y=y_lib, sr=sr_lib, n_mfcc=13)
    mfcc_vars = np.var(mfcc_matrix, axis=1)

    def safe(v, default=0.0):
        return float(default if (v is None or np.isnan(v) or np.isinf(v)) else v)

    record = {
        "MDVP:Fo(Hz)": safe(f0, 150.0),
        "MDVP:Fhi(Hz)": safe(fhi, 180.0),
        "MDVP:Flo(Hz)": safe(flo, 120.0),
        "MDVP:Jitter(%)": safe(jitter_local, 0.005),
        "MDVP:Jitter(Abs)": safe(jitter_abs, 0.00004),
        "MDVP:RAP": safe(jitter_rap, 0.003),
        "MDVP:PPQ": safe(jitter_ppq5, 0.003),
        "Jitter:DDP": safe(jitter_ddp, 0.009),
        "MDVP:Shimmer": safe(shimmer_local, 0.035),
        "MDVP:Shimmer(dB)": safe(shimmer_db, 0.35),
        "Shimmer:APQ3": safe(shimmer_apq3, 0.018),
        "Shimmer:APQ5": safe(shimmer_apq5, 0.022),
        "MDVP:APQ": safe(shimmer_apq11, 0.03),
        "Shimmer:DDA": safe(shimmer_dda, 0.054),
        "NHR": safe(nhr, 0.02),
        "HNR": safe(hnr, 18.0),
        "RPDE": np.clip(safe(rpde, 0.55), 0.05, 0.95),
        "DFA": np.clip(safe(dfa, 0.65), 0.05, 1.2),
        "spread1": safe(spread1, -2.6),
        "spread2": safe(spread2, 0.05),
        "D2": safe(d2, 2.3),
        "PPE": np.clip(safe(ppe, 0.5), 0.05, 0.95),
        "Age": float(age),
        "Sex_code": 1 if str(sex).upper().startswith("M") else 0
    }

    # Add dynamic MFCC variances
    for i in range(1, 11):
        record[f"mfcc_var_MFCC{i}"] = float(mfcc_vars[i - 1]) if i <= len(mfcc_vars) else 0.05

    # Add LAR variances defaults if not computed
    for i in range(1, 11):
        record[f"lar_var_LAR{i}"] = 0.03

    return record

class RequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable caching
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json({"status": "ready", "version": "2.0", "samples_loaded": len(SAMPLE_LIST)})
            return

        if path == "/api/samples":
            self.send_json({"samples": SAMPLE_LIST})
            return

        if path.startswith("/api/audio/"):
            filename = os.path.basename(path)
            wav_path = COMBINED_AUDIO_DIR / filename
            if wav_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(wav_path.stat().st_size))
                self.end_headers()
                with open(wav_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Audio sample file not found")
            return

        if path == "/api/metrics":
            metrics_data = {
                "benchmark_original": {"accuracy": 98.3, "leakage": True, "type": "Original UCI Benchmark (Leakage)"},
                "benchmark_leakfree": {"accuracy": 92.3, "roc_auc": 0.960, "type": "Leak-Free Laboratory Benchmark"},
                "benchmark_subject_cv": {"accuracy": 83.6, "type": "Unseen Subject Laboratory CV"},
                "raw_voice_pure": {"accuracy": 58.0, "roc_auc": 0.639, "type": "Pure 8 kHz Acoustic Measures (5-Fold CV)"},
                "raw_voice_mfcc": {"accuracy": 56.8, "roc_auc": 0.580, "type": "Static MFCCs Alone (5-Fold CV)"},
                "multimodal_fusion": {
                    "accuracy": 81.5,
                    "roc_auc": 0.896,
                    "holdout_accuracy": 76.5,
                    "holdout_auc": 0.889,
                    "type": "Multimodal Fused Ensemble (5-Fold CV)"
                }
            }
            self.send_json(metrics_data)
            return

        if path == "/api/dataset":
            if FUSED_DF is not None:
                records = FUSED_DF.head(81).to_dict(orient="records")
                self.send_json({"total": len(records), "records": records})
            else:
                self.send_json({"total": 0, "records": []})
            return

        # Serve static assets
        if path == "/" or path == "":
            path = "/index.html"

        # Serve output images
        if path.startswith("/Outputs/"):
            out_file = BASE_DIR / path.lstrip("/")
            if out_file.exists():
                self.serve_file(out_file)
                return

        file_path = WEB_DIR / path.lstrip("/")
        if file_path.exists() and file_path.is_file():
            self.serve_file(file_path)
        else:
            self.send_error(404, f"File not found: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            self.handle_predict()
        else:
            self.send_error(404, "Unknown API route")

    def handle_predict(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)

            features_dict = None
            sample_meta = {}

            if "application/json" in content_type:
                data = json.loads(post_body.decode("utf-8"))
                sample_name = data.get("sample_name")
                age = float(data.get("age", 65.0))
                sex = data.get("sex", "M")

                if sample_name:
                    # Look up pre-extracted record if existing sample
                    wav_name = f"{sample_name}.wav" if not sample_name.endswith(".wav") else sample_name
                    wav_path = COMBINED_AUDIO_DIR / wav_name
                    if wav_path.exists():
                        matched_row = FUSED_DF[FUSED_DF["name"] == Path(sample_name).stem]
                        if not matched_row.empty:
                            row_dict = matched_row.iloc[0].to_dict()
                            features_dict = {c: float(row_dict.get(c, 0.0)) for c in FEATURE_COLS if c in row_dict}
                            sample_meta = {
                                "sample_id": sample_name,
                                "ground_truth": "Parkinson's Disease" if row_dict.get("status") == 1 else "Healthy Control",
                                "actual_age": row_dict.get("Age"),
                                "actual_sex": row_dict.get("Sex")
                            }
                        else:
                            features_dict = extract_features_from_wav(wav_path, age=age, sex=sex)
            else:
                # Raw audio binary upload
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(post_body)
                    tmp_path = Path(tmp.name)
                try:
                    features_dict = extract_features_from_wav(tmp_path)
                finally:
                    if tmp_path.exists():
                        os.unlink(tmp_path)

            if features_dict is None or MODEL_PIPELINE is None:
                self.send_json({"error": "Failed to extract vocal features or model not initialized"}, status=500)
                return

            # Construct feature vector matching trained model schema
            feature_vector = pd.DataFrame([{col: features_dict.get(col, 0.0) for col in FEATURE_COLS}])

            # Make Prediction
            probs = MODEL_PIPELINE.predict_proba(feature_vector)[0]
            prob_pd = float(probs[1])
            prob_hc = float(probs[0])
            pred_class = int(prob_pd >= 0.50)

            # Determine Risk Assessment Tier
            if prob_pd >= 0.70:
                risk_tier = "High Risk"
                risk_color = "#ef4444"
                recommendation = "High probability of vocal motor dysphonia consistent with Parkinsonian patterns. Clinical neurological review and UPDRS motor evaluation strongly advised."
            elif prob_pd >= 0.45:
                risk_tier = "Moderate Risk / Indeterminate"
                risk_color = "#f59e0b"
                recommendation = "Intermediate acoustic deviation detected. Repeat phonation test in a quiet acoustic setting or schedule follow-up longitudinal screening."
            else:
                risk_tier = "Low Risk / Typical"
                risk_color = "#10b981"
                recommendation = "Acoustic stability, dynamic frequency modulation, and spectral harmonics fall within expected healthy normative baseline ranges."

            # Clinical Indicator Flags
            flags = []
            f0 = features_dict.get("MDVP:Fo(Hz)", 150)
            hnr = features_dict.get("HNR", 20)
            jitter = features_dict.get("MDVP:Jitter(%)", 0.005)
            shimmer = features_dict.get("MDVP:Shimmer", 0.03)
            rpde = features_dict.get("RPDE", 0.5)

            if hnr < 14.0:
                flags.append("Abnormally Low Harmonics-to-Noise Ratio (Acoustic breathiness/turbulence)")
            if jitter > 0.008:
                flags.append("Elevated Pitch Micro-Perturbation (Vocal cord period instability)")
            if shimmer > 0.08:
                flags.append("Elevated Amplitude Shimmer (Subglottic pressure modulation deficits)")
            if rpde > 0.75:
                flags.append("High Recurrence Entropy (Chaotic vocal cycle dynamics)")

            response_payload = {
                "prediction": pred_class,
                "label": "Parkinson's Disease Detected" if pred_class == 1 else "Healthy Control Pattern",
                "probability_pd": round(prob_pd * 100, 1),
                "probability_hc": round(prob_hc * 100, 1),
                "risk_tier": risk_tier,
                "risk_color": risk_color,
                "recommendation": recommendation,
                "sample_meta": sample_meta,
                "abnormal_indicators": flags,
                "biomarkers": {
                    "f0": round(f0, 1),
                    "hnr": round(hnr, 1),
                    "jitter_percent": round(jitter * 100, 3),
                    "shimmer": round(shimmer, 3),
                    "rpde": round(rpde, 3),
                    "dfa": round(features_dict.get("DFA", 0.65), 3),
                    "ppe": round(features_dict.get("PPE", 0.3), 3),
                    "age": features_dict.get("Age", 65)
                }
            }

            self.send_json(response_payload)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({"error": str(e)}, status=500)

    def serve_file(self, file_path):
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def start_server(port=8080):
    init_multimodal_model()
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"\n=======================================================")
    print(f"  NeuroVoice AI Clinical Web Server Running")
    print(f"  Local URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    start_server(port)
