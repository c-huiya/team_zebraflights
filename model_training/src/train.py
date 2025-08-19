import os
import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify
from scipy import sparse
from scipy.sparse import load_npz

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

app = Flask(__name__)

# ENV defaults (override via POST /train body)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/data/04_model_output"))
ENCODED_DIR = Path(os.getenv("ENCODED_DIR", "/data/02_preprocessed/encoded_split"))
PORT = int(os.getenv("PORT", "8000"))


# Evaluate model at a chosen probability threshold
def evaluate(y_true: np.ndarray, proba: np.ndarray, thresh: float) -> dict:
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


# load preprocessed train/test + feature names
def load_artifacts(encoded_dir: Path):
    X_tr = load_npz(encoded_dir / "X_train.npz")
    X_te = load_npz(encoded_dir / "X_test.npz")
    y_tr = np.load(encoded_dir / "y_train.npy")
    y_te = np.load(encoded_dir / "y_test.npy")
    with open(encoded_dir / "feature_names.json", "r") as f:
        feat_names = np.array(json.load(f), dtype=object)
    return X_tr, X_te, y_tr, y_te, feat_names


# Liveness check endpoint
@app.get("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})


# Train Endpoint
@app.post("/train")
def train():
    """
    Body JSON:
    {
      "encoded_dir": "/data/02_preprocessed/encoded_split",
      "model_dir": "/data/04_model_output",
      "threshold": 0.50,
      "seed": 42,
      "rf_params": {
        "n_estimators": 400,
        "class_weight": "balanced",
        "n_jobs": -1,
        "max_depth": null
      },
      "compute_mi": true
    }
    """
    p = request.get_json(silent=True) or {}
    encoded_dir = Path(p.get("encoded_dir", ENCODED_DIR))
    model_dir = Path(p.get("model_dir", MODEL_DIR))
    threshold = float(p.get("threshold", 0.50))
    seed = int(p.get("seed", 42))
    rf_params = p.get("rf_params", {}) or {}
    compute_mi = bool(p.get("compute_mi", True))

    if not encoded_dir.exists():
        return jsonify({"error": f"encoded_dir not found: {encoded_dir}"}), 400
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-encoded, pre-split data
    try:
        X_tr, X_te, y_tr, y_te, feat_names = load_artifacts(encoded_dir)
    except Exception as e:
        return jsonify({"error": f"Loading encoded artifacts failed: {str(e)}"}), 400

    # Train RF
    defaults = dict(
        n_estimators=400, n_jobs=-1, random_state=seed, class_weight="balanced"
    )
    clf = RandomForestClassifier(**{**defaults, **rf_params})

    try:
        clf.fit(X_tr, y_tr)
        if 1 not in clf.classes_:
            return jsonify(
                {"error": f"Expected class '1' in classes_, got: {clf.classes_}"}
            ), 500
        pos_idx = int(np.where(clf.classes_ == 1)[0][0])
        proba = clf.predict_proba(X_te)[:, pos_idx]
        metrics = evaluate(y_te, proba, threshold)
    except Exception as e:
        return jsonify({"error": f"Training/Evaluation failed: {str(e)}"}), 500

    # Feature importances
    try:
        importances = clf.feature_importances_
        if importances.shape[0] != len(feat_names):
            # fallback names if mismatch (shouldn't happen if preprocess is consistent)
            feat_names = np.array(
                [f"f{i}" for i in range(importances.shape[0])], dtype=object
            )
        fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        import pandas as pd

        pd.DataFrame(fi, columns=["feature", "rf_importance"]).to_csv(
            model_dir / "feature_importances_rf.csv", index=False
        )
    except Exception as e:
        metrics["_warning_fi"] = f"Feature importance failed: {str(e)}"

    # Mutual Information
    if compute_mi:
        try:
            is_discrete = np.array(
                [str(n).startswith("cat__") for n in feat_names], dtype=bool
            )
            mi_scores = np.zeros(len(feat_names), dtype=float)

            if is_discrete.any():
                X_cat = X_tr[:, np.where(is_discrete)[0]]
                if sparse.issparse(X_cat) and not sparse.isspmatrix_csr(X_cat):
                    X_cat = sparse.csr_matrix(X_cat)
                mi_scores[is_discrete] = mutual_info_classif(
                    X_cat, y_tr, discrete_features=True, random_state=seed
                )

            num_mask = ~is_discrete
            if num_mask.any():
                X_num = X_tr[:, np.where(num_mask)[0]]
                if sparse.issparse(X_num):
                    X_num = X_num.toarray()
                mi_scores[num_mask] = mutual_info_classif(
                    X_num, y_tr, discrete_features=False, random_state=seed
                )

            import pandas as pd

            mi_pairs = sorted(
                zip(feat_names, mi_scores), key=lambda x: x[1], reverse=True
            )
            pd.DataFrame(mi_pairs, columns=["feature", "mi_score"]).to_csv(
                model_dir / "mutual_information_train.csv", index=False
            )
        except Exception as e:
            metrics["_warning_mi"] = f"MI computation failed: {str(e)}"

    # Persist artifacts
    try:
        joblib.dump(clf, model_dir / "model.joblib")
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (model_dir / "threshold.txt").write_text(str(threshold))
        (model_dir / "columns.json").write_text(
            json.dumps(
                {
                    "encoded_feature_columns": list(map(str, feat_names)),
                    "target": "binary_target",
                    "class_mapping": {"negative": 0, "positive": 1},
                    "input_mode": "matrix",
                },
                indent=2,
            )
        )
    except Exception as e:
        return jsonify({"error": f"Saving artifacts failed: {str(e)}"}), 500

    # Return a training report as JSON response
    return jsonify(
        {
            "status": "trained",
            "encoded_dir": str(encoded_dir),
            "model_dir": str(model_dir),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "class_balance_train": {
                "neg": int((y_tr == 0).sum()),
                "pos": int((y_tr == 1).sum()),
            },
            "metrics": metrics,
            "artifacts": [
                "model.joblib",
                "metrics.json",
                "threshold.txt",
                "columns.json",
                "feature_importances_rf.csv",
                "mutual_information_train.csv",
            ],
        }
    )


# Metrics retrieval endpoint
@app.get("/metrics")
def get_metrics():
    model_dir = Path(request.args.get("model_dir", MODEL_DIR))
    path = model_dir / "metrics.json"
    if not path.exists():
        return jsonify({"error": f"metrics.json not found in {model_dir}"}), 404
    return jsonify(json.loads(path.read_text()))


# List produced files endpoint
@app.get("/artifacts")
def list_artifacts():
    model_dir = Path(request.args.get("model_dir", MODEL_DIR))
    if not model_dir.exists():
        return jsonify({"error": f"MODEL_DIR not found: {model_dir}"}), 404
    files = sorted([f.name for f in model_dir.glob("*") if f.is_file()])
    return jsonify({"model_dir": str(model_dir), "files": files})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
