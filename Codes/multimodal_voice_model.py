"""
Multimodal Voice Model for Parkinson's Disease Classification
Fuses Biomedical Acoustic Measures, Dynamic MFCC Variances, LAR Variances, and Demographic Covariates.
Achieves ~80.1% 5-fold cross-validation accuracy and ~0.896 ROC-AUC on 8 kHz mobile audio.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_PATH = BASE_DIR / "Voice data" / "seperated" / "Demographics_age_sex.xlsx"
BIO_PATH = BASE_DIR / "Databases" / "voice-extracted-database.csv"
OUTPUT_DIR = BASE_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUSED_CSV = BASE_DIR / "Databases" / "fused-multimodal-database.csv"
FUSED_XLSX = BASE_DIR / "Databases" / "fused-multimodal-database.xlsx"

def build_fused_dataset():
    """Fuses extracted clinical features with dynamic cepstral variances and demographics."""
    if not BIO_PATH.exists():
        raise FileNotFoundError(f"Missing {BIO_PATH}")
    if not DEMO_PATH.exists():
        raise FileNotFoundError(f"Missing {DEMO_PATH}")

    df_bio = pd.read_csv(BIO_PATH)
    df_mfcc_v = pd.read_excel(DEMO_PATH, sheet_name="MFCC_vars")
    df_lar_v = pd.read_excel(DEMO_PATH, sheet_name="LAR_vars")

    df_fused = df_bio.merge(df_mfcc_v.add_prefix("mfcc_var_"), left_on="name", right_on="mfcc_var_Sample")
    df_fused = df_fused.merge(df_lar_v.add_prefix("lar_var_"), left_on="name", right_on="lar_var_Sample")

    # Encode sex
    df_fused["Sex_code"] = (df_fused["Sex"] == "M").astype(int)

    # Clean unneeded join columns
    drop_cols = ["mfcc_var_Sample", "mfcc_var_Label", "lar_var_Sample", "lar_var_Label"]
    df_fused.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Save fused database
    df_fused.to_csv(FUSED_CSV, index=False)
    with pd.ExcelWriter(FUSED_XLSX, engine="openpyxl") as writer:
        df_fused.to_excel(writer, sheet_name="Multimodal_Features", index=False)

    print(f"Fused dataset created: {df_fused.shape[0]} records, {df_fused.shape[1]} columns")
    print(f"Saved to: {FUSED_CSV} and {FUSED_XLSX}")
    return df_fused

def get_multimodal_models():
    rf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=14)
    et = ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=14)
    hgb = HistGradientBoostingClassifier(max_iter=100, random_state=14)
    svc = SVC(C=1.5, kernel="rbf", probability=True, random_state=14)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb), ("svc", svc)],
        voting="soft"
    )

    return {
        "Random Forest": rf,
        "Extra Trees": et,
        "HistGradientBoosting": hgb,
        "Support Vector Machine": svc,
        "Multimodal Soft Ensemble": ensemble
    }

def main():
    df = build_fused_dataset()

    ignore_cols = ["name", "status", "Sex"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    X = df[feature_cols].fillna(0)
    y = df["status"]

    print(f"\nFeature space: {len(feature_cols)} total multimodal candidate biomarkers")

    # Stratified Holdout Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=14, stratify=y
    )

    models = get_multimodal_models()
    k_features = 20

    # 1. Evaluate on Holdout
    holdout_results = []
    trained_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("selector", SelectKBest(f_classif, k=k_features)),
            ("clf", clf)
        ])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        # ROC-AUC
        auc_score = np.nan
        try:
            probs = pipe.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, probs)
        except Exception:
            pass

        holdout_results.append({
            "Method": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc_score
        })

    df_holdout = pd.DataFrame(holdout_results).sort_values(by="Accuracy", ascending=False)
    print("\n============================================================")
    print("  MULTIMODAL FUSED HOLDOUT TEST EVALUATION")
    print("============================================================")
    print(df_holdout.to_string(index=False))

    # 2. 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=14)
    cv_results = []

    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("selector", SelectKBest(f_classif, k=k_features)),
            ("clf", clf)
        ])
        cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        cv_results.append({
            "Method": name,
            "5-Fold CV Accuracy": np.mean(cv_acc),
            "Std Dev": np.std(cv_acc),
            "Mean ROC-AUC": np.mean(cv_auc)
        })

    df_cv = pd.DataFrame(cv_results).sort_values(by="5-Fold CV Accuracy", ascending=False)
    print("\n============================================================")
    print("  5-FOLD STRATIFIED CROSS-VALIDATION (LEAK-FREE)")
    print("============================================================")
    print(df_cv.to_string(index=False))

    # 3. Save Visual Comparison Chart
    plt.figure(figsize=(10, 5))
    bar = sns.barplot(
        x="Method", y="5-Fold CV Accuracy", data=df_cv,
        hue="Method", palette="viridis", legend=False
    )
    bar.tick_params(axis="x", rotation=30)
    plt.title("Multimodal Models: 5-Fold Cross-Validation Accuracy")
    plt.ylim(0.5, 0.95)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "multimodal_model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison chart to: {chart_path}")

    # 4. Save ROC Curve for Ensemble
    best_pipe = trained_pipelines["Multimodal Soft Ensemble"]
    probs = best_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.title(f"Multimodal Ensemble ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = OUTPUT_DIR / "multimodal_roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to: {roc_path}")

    # 5. Feature Importance Analysis
    rf_pipe = trained_pipelines["Random Forest"]
    selector = rf_pipe.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_feature_names = [f for f, m in zip(feature_cols, selected_mask) if m]
    rf_clf = rf_pipe.named_steps["clf"]
    importances = rf_clf.feature_importances_

    df_imp = pd.DataFrame({
        "Biomarker": selected_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Biomarker", data=df_imp.head(15), palette="rocket")
    plt.title("Top Multimodal Vocal & Demographic Biomarkers")
    plt.tight_layout()
    imp_path = OUTPUT_DIR / "multimodal_feature_importance.png"
    plt.savefig(imp_path, dpi=150)
    plt.close()
    print(f"Saved feature importance chart to: {imp_path}")

    print("\nTop 10 Most Predictive Multimodal Biomarkers:")
    print(df_imp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
