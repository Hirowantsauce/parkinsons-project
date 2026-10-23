"""
Machine Learning Evaluation Pipeline on Newly Extracted Voice Biomarkers
Performs leak-free training, holdout evaluation, 5-fold cross-validation, and saves visual reports.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
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
CSV_PATH = BASE_DIR / "Databases" / "voice-extracted-database.csv"
XLSX_PATH = BASE_DIR / "Databases" / "voice-extracted-database.xlsx"
OUTPUT_DIR = BASE_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing extracted database at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded voice-extracted database: {df.shape[0]} records, {df.shape[1]} columns")
    print(f"Cohort balance: Healthy={sum(df['status'] == 0)}, Parkinson's={sum(df['status'] == 1)}")
    return df

def get_models():
    return {
        "Logistic Regression": LogisticRegression(C=0.4, max_iter=1000, solver="liblinear", random_state=14),
        "Decision Tree": DecisionTreeClassifier(random_state=14),
        "Random Forest (Gini)": RandomForestClassifier(criterion="gini", random_state=14),
        "Random Forest (Entropy)": RandomForestClassifier(criterion="entropy", random_state=14),
        "Support Vector Machine": SVC(probability=True, random_state=14),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Bernoulli Naive Bayes": BernoulliNB()
    }

def preprocess_train_test(X_train_raw, y_train_raw, X_test_raw, use_oversampling=True, pca_variance=0.95):
    """Leak-free preprocessing fitted exclusively on training fold."""
    if use_oversampling:
        ros = RandomOverSampler(random_state=14)
        X_train_res, y_train_res = ros.fit_resample(X_train_raw, y_train_raw)
    else:
        X_train_res, y_train_res = X_train_raw.copy(), y_train_raw.copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=pca_variance, random_state=14)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, y_train_res, X_test_pca, scaler, pca

def evaluate_models(trained_dict, X_eval, y_eval):
    results = []
    for name, clf in trained_dict.items():
        y_pred = clf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)

        auc_score = np.nan
        try:
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_eval)[:, 1]
                auc_score = roc_auc_score(y_eval, y_prob)
            elif hasattr(clf, "decision_function"):
                y_prob = clf.decision_function(X_eval)
                auc_score = roc_auc_score(y_eval, y_prob)
        except Exception:
            pass

        results.append({
            "Method": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc_score
        })
    return pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

def plot_roc_curve(model, X_eval, y_eval, model_name, save_path):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_eval)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_eval)
    else:
        probabilities = model.predict(X_eval)

    fpr, tpr, _ = roc_curve(y_eval, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.title(f"ROC Curve - {model_name} (AUC = {roc_auc:.3f})")
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve plot to: {save_path}")

def run_cross_validation(X_raw, y_raw, n_splits=5):
    """5-Fold Stratified Cross Validation with leak-free fold preprocessing."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=14)
    cv_scores = {name: [] for name in get_models().keys()}

    for train_idx, test_idx in skf.split(X_raw, y_raw):
        X_tr, y_tr = X_raw.iloc[train_idx], y_raw.iloc[train_idx]
        X_te, y_te = X_raw.iloc[test_idx], y_raw.iloc[test_idx]

        X_tr_proc, y_tr_proc, X_te_proc, _, _ = preprocess_train_test(
            X_tr, y_tr, X_te, use_oversampling=True
        )

        models = get_models()
        for name, clf in models.items():
            clf.fit(X_tr_proc, y_tr_proc)
            pred = clf.predict(X_te_proc)
            cv_scores[name].append(accuracy_score(y_te, pred))

    summary = []
    for name, scores in cv_scores.items():
        summary.append({
            "Method": name,
            "5-Fold CV Mean Accuracy": np.mean(scores),
            "Std Dev": np.std(scores)
        })
    return pd.DataFrame(summary).sort_values(by="5-Fold CV Mean Accuracy", ascending=False)

def main():
    df = load_data()

    # Acoustic feature columns matching park-database schema
    feature_cols = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    X = df[feature_cols].copy()
    y = df["status"].copy()

    # 1. Holdout Train/Test Split (80/20 Stratified)
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.2, random_state=14, stratify=y
    )

    X_train, y_train, X_test, scaler, pca = preprocess_train_test(
        X_train_raw, y_train_raw, X_test_raw, use_oversampling=True, pca_variance=0.95
    )

    print(f"\n--- Feature Space Preprocessing ---")
    print(f"Original features: {len(feature_cols)}")
    print(f"PCA reduced components (95% variance): {pca.n_components_}")
    print(f"Training shape after oversampling: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")

    # Train individual models
    base_models = get_models()
    trained_models = {}
    for name, clf in base_models.items():
        clf.fit(X_train, y_train)
        trained_models[name] = clf

    # Add Voting Classifiers
    estimators = [(name.replace(" ", "_"), clf) for name, clf in trained_models.items()]
    hard_voting = VotingClassifier(estimators=estimators, voting="hard")
    hard_voting.fit(X_train, y_train)
    trained_models["Voting Ensemble (Hard)"] = hard_voting

    soft_voting = VotingClassifier(estimators=estimators, voting="soft")
    soft_voting.fit(X_train, y_train)
    trained_models["Voting Ensemble (Soft)"] = soft_voting

    # Evaluate on Holdout
    df_eval = evaluate_models(trained_models, X_test, y_test)
    print("\n=======================================================")
    print("  EVALUATION ON VOICE-EXTRACTED HOLDOUT TEST SET")
    print("=======================================================")
    print(df_eval.to_string(index=False))

    # Save Comparison Plot
    plt.figure(figsize=(10, 5))
    bar = sns.barplot(x="Method", y="Accuracy", data=df_eval, hue="Method", palette="viridis", legend=False)
    bar.tick_params(axis="x", rotation=45)
    plt.title("Voice-Extracted Clinical Features: Model Accuracy (Holdout Test)")
    plt.ylim(0.4, 1.05)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "voice_extracted_model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {chart_path}")

    # Best Model Report & ROC
    best_name = df_eval.iloc[0]["Method"]
    best_clf = trained_models[best_name]
    best_preds = best_clf.predict(X_test)

    print(f"\n--- Best Model Details: {best_name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds))

    roc_path = OUTPUT_DIR / "voice_extracted_roc_curve.png"
    plot_roc_curve(best_clf, X_test, y_test, best_name, roc_path)

    # 2. 5-Fold Cross Validation
    df_cv = run_cross_validation(X, y, n_splits=5)
    print("\n=======================================================")
    print("  5-FOLD STRATIFIED CROSS-VALIDATION (LEAK-FREE)")
    print("=======================================================")
    print(df_cv.to_string(index=False))

    # 3. Compare with MFCCs if available in Excel
    if XLSX_PATH.exists():
        try:
            df_mfcc = pd.read_excel(XLSX_PATH, sheet_name="MFCC_24_Features")
            mfcc_cols = [c for c in df_mfcc.columns if c.startswith("MFCC_")]
            X_mfcc = df_mfcc[mfcc_cols].copy()
            y_mfcc = df_mfcc["status"].copy()

            df_mfcc_cv = run_cross_validation(X_mfcc, y_mfcc, n_splits=5)
            print("\n=======================================================")
            print("  5-FOLD CV COMPARISON: 24 MFCCs ALONE")
            print("=======================================================")
            print(df_mfcc_cv.to_string(index=False))
        except Exception as e:
            print(f"Could not run MFCC sheet comparison: {e}")

if __name__ == "__main__":
    main()
