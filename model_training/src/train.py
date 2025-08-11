import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# silence noisy MI warnings from internal clustering utils
warnings.filterwarnings("ignore", module="sklearn.metrics.cluster._supervised")


# ========================
# Preprocessing builder
# ========================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for categorical and numeric columns (no imputation)."""
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    steps = []
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    if num:
        steps.append(("num", StandardScaler(), num))

    if not steps:
        raise ValueError("No usable columns detected.")
    return ColumnTransformer(steps, remainder="drop")


# ========================
# Target Mapping
# ========================
def make_binary_target(y: pd.Series, unknown_policy: str = "negative"):
    """
    Map the three known CAFV strings to 0/1.
    unknown_policy: "negative" (default) maps UNKNOWN to 0,
                    or "drop" to return a mask for filtering.
    """
    ELIGIBLE = "clean alternative fuel vehicle eligible"
    NOT_ELIG = "not eligible due to low battery range"
    UNKNOWN = "eligibility unknown as battery range has not been researched"

    y_norm = y.astype(str).str.strip().str.lower()

    valid_values = {ELIGIBLE, NOT_ELIG, UNKNOWN}
    unexpected = sorted(set(y_norm.unique()) - valid_values)
    if unexpected:
        raise ValueError(f"Unexpected target values: {unexpected}")

    if unknown_policy == "drop":
        keep_mask = y_norm != UNKNOWN
        mapped = np.where(y_norm[keep_mask] == ELIGIBLE, 1, 0).astype(int)
        return mapped, keep_mask.values

    # Treat UNKNOWN as negative
    mapped = np.where(y_norm == ELIGIBLE, 1, 0).astype(int)
    return mapped


# ========================
# Evaluation
# ========================
def evaluate(y_true: np.ndarray, proba: np.ndarray, thresh: float) -> dict:
    """Binary classification evaluation with fixed label order."""
    y_true = np.asarray(y_true, dtype=int)
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1_at_threshold": float(f1_score(y_true, pred)),
        "accuracy_at_threshold": float(accuracy_score(y_true, pred)),
        "threshold": float(thresh),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "positives_in_test": int(y_true.sum()),
        "negatives_in_test": int((1 - y_true).sum()),
    }


def get_feature_names(pre: ColumnTransformer) -> np.ndarray:
    """
    Get output feature names from a fitted ColumnTransformer.
    """
    try:
        return pre.get_feature_names_out()
    except Exception:
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                fn = trans.get_feature_names_out(cols)
                fn = [f"{name}__{f}" for f in fn]
                names.extend(fn)
            else:
                if isinstance(cols, (list, tuple, np.ndarray)):
                    names.extend([f"{name}__{c}" for c in cols])
                else:
                    names.append(f"{name}__{cols}")
        return np.array(names)


def print_topk(label: str, pairs, k=20):
    print(f"\n=== Top {k} {label} ===")
    for i, (feat, score) in enumerate(pairs[:k], 1):
        try:
            score_f = float(score)
        except Exception:
            score_f = score
        print(f"{i:>2}. {feat}: {score_f:.6f}")


# ========================
# Main
# ========================
def main():
    # ---- Paths & constants (ENV first, fall back to local defaults) ----
    train_csv = os.getenv("DATA_PATH", "../data/02_preprocessed/clean_dataset.csv")
    out_dir = os.getenv("MODEL_DIR", "../data/04_model_output")
    target_col = "Clean Alternative Fuel Vehicle (CAFV) Eligibility"

    threshold = 0.50
    test_size = 0.2
    seed = 42

    os.makedirs(out_dir, exist_ok=True)

    # ---- Load full dataset ----
    df_full = pd.read_csv(train_csv)

    # ---- Drop leakage cols from features ----
    leakage_cols = [
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
        "cafv_eligible",
        "VIN (1-10)",
        "DOL Vehicle ID",
        "Vehicle Location",
        "2020 Census Tract",
        "Electric Range",
        "Base_MSRP_missing",
        "Model Year",
    ]
    X = df_full.drop(columns=[c for c in leakage_cols if c in df_full.columns])

    # ---- Build binary target ----
    y_text = df_full[target_col]
    y = make_binary_target(y_text, unknown_policy="negative")  # or "drop"

    # ---- Model pipeline ----
    pre = build_preprocessor(X)
    clf = RandomForestClassifier(
        n_estimators=400,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # ---- Split, train ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    pipe.fit(X_tr, y_tr)

    # ---- Evaluate ----
    classes_ = pipe.named_steps["clf"].classes_
    if 1 not in classes_:
        raise RuntimeError(f"Expected class '1' in classes_, got: {classes_}")
    pos_idx = int(np.where(classes_ == 1)[0][0])
    proba = pipe.predict_proba(X_te)[:, pos_idx]
    metrics = evaluate(y_te, proba, threshold)
    print(json.dumps(metrics, indent=2))

    # ---- Feature names (after encoding) ----
    pre_fitted: ColumnTransformer = pipe.named_steps["pre"]
    feat_names = get_feature_names(pre_fitted)

    # ---- RandomForest feature importances ----
    rf = pipe.named_steps["clf"]
    importances = rf.feature_importances_
    fi_pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    print_topk("RandomForest feature importances", fi_pairs, k=20)

    pd.DataFrame(fi_pairs, columns=["feature", "rf_importance"]).to_csv(
        os.path.join(out_dir, "feature_importances_rf.csv"), index=False
    )

    # ---- Mutual information on encoded TRAIN matrix ----
    Xtr_enc = pre_fitted.transform(X_tr)
    feat_names = get_feature_names(pre_fitted)
    is_discrete = np.array(
        [name.startswith("cat__") for name in feat_names], dtype=bool
    )

    def _colslice(M, mask):
        if sparse.issparse(M):
            return M[:, np.where(mask)[0]]
        return M[:, mask]

    mi_scores = np.zeros(len(feat_names), dtype=float)

    # Categorical block (discrete) — keep sparse
    if is_discrete.any():
        X_cat = _colslice(Xtr_enc, is_discrete)
        if sparse.issparse(X_cat) and not sparse.isspmatrix_csr(X_cat):
            X_cat = sparse.csr_matrix(X_cat)
        mi_cat = mutual_info_classif(
            X_cat, y_tr, discrete_features=True, random_state=seed
        )
        mi_scores[is_discrete] = mi_cat

    # Numeric block (continuous)
    num_mask = ~is_discrete
    if num_mask.any():
        X_num = _colslice(Xtr_enc, num_mask)
        if sparse.issparse(X_num):
            X_num = X_num.toarray()
        mi_num = mutual_info_classif(
            X_num, y_tr, discrete_features=False, random_state=seed
        )
        mi_scores[num_mask] = mi_num

    mi_pairs = sorted(zip(feat_names, mi_scores), key=lambda x: x[1], reverse=True)
    print_topk("Mutual Information (train, encoded)", mi_pairs, k=20)

    pd.DataFrame(mi_pairs, columns=["feature", "mi_score"]).to_csv(
        os.path.join(out_dir, "mutual_information_train.csv"), index=False
    )

    # ---- Save artifacts (to MODEL_DIR) ----
    joblib.dump(pipe, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(threshold))
    with open(os.path.join(out_dir, "columns.json"), "w") as f:
        json.dump(
            {
                "raw_feature_columns": X.columns.tolist(),
                "encoded_feature_columns": feat_names.tolist(),
                "target": target_col,
                "class_mapping": {"negative": 0, "positive": 1},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    np.random.seed(42)
    main()
