from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os, json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy import sparse
from scipy.sparse import save_npz

app = Flask(__name__)
DATA_ROOT = Path(os.getenv("DATA_DIR", "/data"))

# Helpers
def _iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return s.quantile(0.01), s.quantile(0.99)
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def map_cafv(y: pd.Series, unknown_policy="negative"):
    ELIGIBLE = "clean alternative fuel vehicle eligible"
    NOT_ELIG = "not eligible due to low battery range"
    UNKNOWN = "eligibility unknown as battery range has not been researched"

    y_norm = y.astype(str).str.strip().str.lower()
    valid = {ELIGIBLE, NOT_ELIG, UNKNOWN}
    bad = sorted(set(y_norm.unique()) - valid)
    if bad:
        raise ValueError(f"Unexpected target values: {bad}")

    if unknown_policy == "drop":
        keep = y_norm != UNKNOWN
        y_bin = np.where(y_norm[keep] == ELIGIBLE, 1, 0).astype(int)
        return y_bin, keep
    return np.where(y_norm == ELIGIBLE, 1, 0).astype(int), np.ones(len(y), bool)

def get_feature_names(pre: ColumnTransformer):
    try:
        return pre.get_feature_names_out()
    except Exception:
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                fn = trans.get_feature_names_out(cols)
                names.extend([f"{name}{f}" for f in fn])
            else:
                names.extend([f"{name}{c}" for c in cols])
        return np.array(names)

# Core pipeline
def run_preprocessing(
    input_path,
    output_path,
    outlier_strategy="clip",
    do_encode_split=True,
    target_col="Clean Alternative Fuel Vehicle (CAFV) Eligibility",
    unknown_policy="negative",
    test_size=0.2,
    seed=42,
    encoded_dir=None
):
    df = pd.read_csv(input_path)

    # Cleaning
    df = df.drop_duplicates()
    if "DOL Vehicle ID" in df.columns:
        df = df.drop_duplicates(subset=["DOL Vehicle ID"])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace({"nan": np.nan}).fillna("unknown")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[f"{col}_was_na"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median(skipna=True))

    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype(str)
        sample = s.sample(min(len(s), 2000), random_state=42)
        clean = sample.str.replace(r"[,\s]", "", regex=True).str.replace(r"^\$", "", regex=True)
        if pd.to_numeric(clean, errors="coerce").notna().mean() >= 0.85:
            df[col] = pd.to_numeric(
                s.str.replace(r"[,\s]", "", regex=True).str.replace(r"^\$", "", regex=True),
                errors="coerce"
            )

    if "Base MSRP" in df.columns:
        df["Base MSRP"] = pd.to_numeric(df["Base MSRP"], errors="coerce")
        df.loc[df["Base MSRP"] <= 0, "Base MSRP"] = np.nan
        df["Base MSRP"] = df["Base MSRP"].fillna(df["Base MSRP"].median(skipna=True))

    hard_caps = {"Base MSRP": (0, 200000)}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty: continue
        lo, hi = _iqr_bounds(s)
        if col in hard_caps:
            lo = max(lo, hard_caps[col][0]); hi = min(hi, hard_caps[col][1])
        if outlier_strategy == "remove":
            df = df[(df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))]
        else:
            df[col] = df[col].clip(lower=lo, upper=hi)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    result = {"rows_cleaned": len(df), "cleaned_file": str(output_path)}

    # Encoding + splitting
    if do_encode_split:
        y_bin, keep = map_cafv(df[target_col], unknown_policy)
        df = df.loc[keep].reset_index(drop=True)

        leakage_cols = [
            target_col,
            "cafv_eligible",
            "VIN (1-10)",
            "DOL Vehicle ID",
            "Vehicle Location",
            "2020 Census Tract",
            "Electric Range",
            "Base_MSRP_missing",
            "Model Year",
        ]
        X = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")
        y = y_bin.astype(int)

        cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else [] +
            [("num", StandardScaler(), num)] if num else [],
            remainder="drop"
        )
        X_enc = pre.fit_transform(X)
        feat_names = get_feature_names(pre)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_enc, y, test_size=test_size, stratify=y, random_state=seed
        )

        out_dir = Path(encoded_dir or Path(output_path).parent / "encoded_split")
        out_dir.mkdir(parents=True, exist_ok=True)

        save_npz(out_dir / "X_train.npz", sparse.csr_matrix(X_tr))
        save_npz(out_dir / "X_test.npz", sparse.csr_matrix(X_te))
        np.save(out_dir / "y_train.npy", y_tr)
        np.save(out_dir / "y_test.npy", y_te)
        (out_dir / "feature_names.json").write_text(json.dumps(list(map(str, feat_names)), indent=2))

        result.update({
            "encoded_artifacts": str(out_dir),
            "n_features": int(X_enc.shape[1]),
            "class_balance": {"neg": int((y == 0).sum()), "pos": int((y == 1).sum())}
        })

    return result

# Flask endpoints
@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        body = request.get_json(force=True)
        input_rel = body.get("input", "01_raw/working_dataset.csv")
        output_rel = body.get("output", "02_preprocessed/clean_dataset.csv")
        outlier_strategy = body.get("outlier_strategy", "clip")
        do_encode_split = bool(body.get("do_encode_split", True))
        target_col = body.get("target_col", "Clean Alternative Fuel Vehicle (CAFV) Eligibility")
        unknown_policy = body.get("unknown_policy", "negative")
        test_size = float(body.get("test_size", 0.2))
        seed = int(body.get("seed", 42))
        encoded_rel = body.get("encoded_dir", "02_preprocessed/encoded_split")

        input_path = DATA_ROOT / input_rel
        output_path = DATA_ROOT / output_rel
        encoded_dir = DATA_ROOT / encoded_rel

        result = run_preprocessing(
            input_path, output_path,
            outlier_strategy=outlier_strategy,
            do_encode_split=do_encode_split,
            target_col=target_col,
            unknown_policy=unknown_policy,
            test_size=test_size,
            seed=seed,
            encoded_dir=encoded_dir
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
