# %% [markdown]
# # Parkinson's Disease Detection from Voice Biomarkers
# Refactored to eliminate data leakage, ensure robust evaluation, and support subject-level validation.

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
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

# %%
# Resolve dynamic project path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Databases" / "park-database.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Database not found at expected path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")

# %%
# Extract subject ID to enable subject-level cross-validation
# e.g., 'phon_R01_S01_1' -> 'phon_R01_S01'
df["subject_id"] = df["name"].apply(lambda x: x.rsplit("_", 1)[0])
print(f"Unique subjects: {df['subject_id'].nunique()}")
print(f"Target distribution ('status'):\n{df['status'].value_counts()}")

# %%
# Features and Target definition
feature_cols = [c for c in df.columns if c not in ["status", "name", "subject_id"]]
X = df[feature_cols].copy()
y = df["status"].copy()
groups = df["subject_id"].copy()

# %% [markdown]
# ### Exploratory Data Analysis Helper Functions

# %%
def plot_distributions(data, columns, max_cols=6):
    """Plots distribution with KDE for selected numerical features."""
    cols_to_plot = columns[:max_cols]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, col in enumerate(cols_to_plot):
        sns.histplot(data[col], kde=True, ax=axes[idx], color="teal")
        axes[idx].set_title(col)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data, features):
    """Plots correlation heatmap among vocal features."""
    plt.figure(figsize=(16, 14))
    corr = data[features].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Vocal Biomarker Feature Correlations", fontsize=14)
    plt.tight_layout()
    plt.show()

# plot_distributions(df, feature_cols)
# plot_correlation_heatmap(df, feature_cols)

# %% [markdown]
# ### Leak-Free Preprocessing Pipeline Function
# Preprocessing (Resampling, Scaling, PCA) is fitted ONLY on training data, then applied to test data.

# %%
def preprocess_train_test(X_train_raw, y_train_raw, X_test_raw, use_oversampling=True, pca_variance=0.95):
    """
    Applies resampling (train only), MinMax scaling, and PCA without data leakage.
    """
    # 1. Resample training fold ONLY
    if use_oversampling:
        ros = RandomOverSampler(random_state=14)
        X_train_res, y_train_res = ros.fit_resample(X_train_raw, y_train_raw)
    else:
        X_train_res, y_train_res = X_train_raw.copy(), y_train_raw.copy()

    # 2. Fit Scaler on training fold ONLY, transform test
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test_raw)

    # 3. Fit PCA on training fold ONLY, transform test
    pca = PCA(n_components=pca_variance, random_state=14)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, y_train_res, X_test_pca, scaler, pca

# %% [markdown]
# ### 1. Leak-Free Holdout Train/Test Split (80/20 Stratified)

# %%
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=14, stratify=y
)

X_train, y_train, X_test, scaler, pca = preprocess_train_test(
    X_train_raw, y_train_raw, X_test_raw, use_oversampling=True, pca_variance=0.95
)

print(f"Training features shape after PCA: {X_train.shape}")
print(f"Testing features shape after PCA:  {X_test.shape}")
print(f"PCA retained components: {pca.n_components_} (explaining ~95% variance)")

# %% [markdown]
# ### Build Classifiers

# %%
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

models = get_models()

# Fit individual models
trained_models = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    trained_models[name] = clf

# Construct Hard & Soft Voting Classifiers
voting_estimators = [(name.replace(" ", "_"), clf) for name, clf in trained_models.items()]

hard_voting = VotingClassifier(estimators=voting_estimators, voting="hard")
hard_voting.fit(X_train, y_train)
trained_models["Voting Ensemble (Hard)"] = hard_voting

soft_voting = VotingClassifier(estimators=voting_estimators, voting="soft")
soft_voting.fit(X_train, y_train)
trained_models["Voting Ensemble (Soft)"] = soft_voting

# %% [markdown]
# ### Model Evaluation on Leak-Free Test Set

# %%
def evaluate_models(trained_dict, X_eval, y_eval):
    results = []
    for name, clf in trained_dict.items():
        y_pred = clf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)

        # ROC-AUC if probabilities or decision function available
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

df_results = evaluate_models(trained_models, X_test, y_test)
print("\n--- Leak-Free Holdout Test Set Performance ---")
print(df_results.to_string(index=False))

# Plot accuracy comparison
OUTPUT_DIR = BASE_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
bar = sns.barplot(x="Method", y="Accuracy", data=df_results, hue="Method", palette="Blues_r", legend=False)
bar.tick_params(axis="x", rotation=45)
plt.title("Model Comparison on Test Split (Leak-Free)")
plt.ylim(0.5, 1.05)
plt.tight_layout()
chart_path = OUTPUT_DIR / "model_comparison_leak_free.png"
plt.savefig(chart_path, dpi=150)
print(f"Saved comparison plot to: {chart_path}")
plt.close()

# %% [markdown]
# ### Detailed Reports for Top Performers

# %%
best_model_name = df_results.iloc[0]["Method"]
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test)

print(f"\n--- Best Model: {best_model_name} ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# %% [markdown]
# ### Corrected ROC-AUC Plotter Function

# %%
def plot_roc(model, X_eval, y_eval, model_name="Classifier", save_name=None):
    """Correctly plots ROC curve using the passed model's predictions."""
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
    if save_name:
        out_roc = OUTPUT_DIR / save_name
        plt.savefig(out_roc, dpi=150)
        print(f"Saved ROC curve to: {out_roc}")
    plt.close()

plot_roc(trained_models["Random Forest (Entropy)"], X_test, y_test, "Random Forest (Entropy)", "roc_curve_rf_entropy.png")

# %% [markdown]
# ### 2. Subject-Wise Cross-Validation (Prevents Speaker Identity Memorization)
# In medical voice diagnosis, recordings from the same individual must NEVER be split between train and test.

# %%
def run_subject_wise_evaluation(X_raw, y_raw, groups, n_splits=5):
    """Evaluates models where each test fold contains completely unseen subjects."""
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=14)
    cv_scores = {name: [] for name in get_models().keys()}

    for train_idx, test_idx in gss.split(X_raw, y_raw, groups):
        X_tr, y_tr = X_raw.iloc[train_idx], y_raw.iloc[train_idx]
        X_te, y_te = X_raw.iloc[test_idx], y_raw.iloc[test_idx]

        # Leak-free fold preprocessing
        X_tr_proc, y_tr_proc, X_te_proc, _, _ = preprocess_train_test(
            X_tr, y_tr, X_te, use_oversampling=True
        )

        fold_models = get_models()
        for name, clf in fold_models.items():
            clf.fit(X_tr_proc, y_tr_proc)
            pred = clf.predict(X_te_proc)
            cv_scores[name].append(accuracy_score(y_te, pred))

    cv_summary = []
    for name, scores in cv_scores.items():
        cv_summary.append({
            "Method": name,
            "Subject-CV Mean Accuracy": np.mean(scores),
            "Std Dev": np.std(scores)
        })

    df_cv = pd.DataFrame(cv_summary).sort_values(by="Subject-CV Mean Accuracy", ascending=False)
    print("\n--- Subject-Wise Group Validation (Unseen Patients) ---")
    print(df_cv.to_string(index=False))
    return df_cv

if __name__ == "__main__":
    df_subject_cv = run_subject_wise_evaluation(X, y, groups)
