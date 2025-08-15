from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)
DATA_ROOT = Path(os.getenv("DATA_DIR", "/data"))

def _iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return s.quantile(0.01), s.quantile(0.99)
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def run_preprocessing(input_path, output_path, outlier_strategy="clip"):
    df = pd.read_csv(input_path)

    # 1. Drop duplicates
    df = df.drop_duplicates()
    if 'DOL Vehicle ID' in df.columns:
        df = df.drop_duplicates(subset=['DOL Vehicle ID'])

    # 2. Handle missing data
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace({"nan": np.nan}).fillna("unknown")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[f"{col}_was_na"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median(skipna=True))

    # 3. Fix data types
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype(str)
        sample = s.sample(min(len(s), 2000), random_state=42)
        clean = sample.str.replace(r"[,\s]", "", regex=True).str.replace(r"^\$", "", regex=True)
        if pd.to_numeric(clean, errors="coerce").notna().mean() >= 0.85:
            df[col] = pd.to_numeric(
                s.str.replace(r"[,\s]", "", regex=True).str.replace(r"^\$", "", regex=True),
                errors="coerce"
            )

    # Fix Base MSRP if exists
    if "Base MSRP" in df.columns:
        df["Base MSRP"] = pd.to_numeric(df["Base MSRP"], errors="coerce")
        df.loc[df["Base MSRP"] <= 0, "Base MSRP"] = np.nan
        df["Base MSRP"] = df["Base MSRP"].fillna(df["Base MSRP"].median(skipna=True))

    # 4. Outlier treatment
    hard_caps = {"Base MSRP": (0, 200000)}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty:
            continue
        lo, hi = _iqr_bounds(s)
        if col in hard_caps:
            lo = max(lo, hard_caps[col][0])
            hi = min(hi, hard_caps[col][1])
        if outlier_strategy == "remove":
            df = df[(df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))]
        else:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "rows_cleaned": len(df),
        "cleaned_file": str(output_path)
    }

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        body = request.get_json(force=True)
        input_rel = body.get("input", "01_raw/working_dataset.csv")
        output_rel = body.get("output", "02_preprocessed/clean_dataset.csv")
        outlier_strategy = body.get("outlier_strategy", "clip")  # "clip" or "remove"

        input_path = DATA_ROOT / input_rel
        output_path = DATA_ROOT / output_rel

        result = run_preprocessing(input_path, output_path, outlier_strategy=outlier_strategy)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
