from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)

DATA_ROOT = Path(os.getenv("DATA_DIR", "/data"))

def find_outliers(series, upper_bound=None, ignore_zeros=True, lower_tail=False,
                  method="iqr", p_hi=0.99):
    s = pd.to_numeric(series, errors="coerce")
    if ignore_zeros:
        s = s.mask(s == 0)
    s = s.dropna()
    if s.empty:
        return {"count": 0, "percent_col": 0.0, "percent_ds": 0.0,
                "bound_used": None, "max_value": None, "outlier_values": []}

    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_upper = s.quantile(0.99) if iqr == 0 else q3 + 1.5 * iqr
        bound = iqr_upper if upper_bound is None else min(iqr_upper, upper_bound)
    elif method == "percentile":
        bound = s.quantile(p_hi) if upper_bound is None else min(s.quantile(p_hi), upper_bound)
    else:
        raise ValueError("method must be 'iqr' or 'percentile'")

    outliers = s[s < bound] if lower_tail else s[s > bound]

    return {
        "count": int(len(outliers)),
        "percent_col": float(len(outliers) / len(s) * 100),
        "percent_ds": float(len(outliers) / len(series) * 100),
        "iqr_upper": float(bound) if method == "iqr" else None,
        "bound_used": float(bound) if np.isfinite(bound) else None,
        "max_value": float(s.max()),
        "outlier_values": outliers.sort_values(ascending=False).head(10).astype(float).tolist()
    }

def run_preprocessing(input_path, inference_path, working_path, clean_path):
    original_df = pd.read_csv(input_path)

    # Split dataset
    inference_df = original_df.sample(frac=0.25, random_state=42)
    df = original_df.drop(inference_df.index)

    # Save inference and working files
    inference_df.to_csv(inference_path, index=False)
    df.to_csv(working_path, index=False)

    # Remove duplicates
    before_count = len(df)
    df.drop_duplicates(inplace=True)
    if 'DOL Vehicle ID' in df.columns:
        df.drop_duplicates(subset=['DOL Vehicle ID'], inplace=True)
    after_count = len(df)

    # Drop rows with missing values
    df = df.dropna()

    # Normalize string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    if "Model" in df.columns:
        df["Model"] = df["Model"].str.replace(r"\s+", "", regex=True)

    # Fix Base MSRP
    if "Base MSRP" in df.columns:
        df["Base MSRP"] = pd.to_numeric(df["Base MSRP"], errors="coerce")
        df.loc[df["Base MSRP"] <= 0, "Base MSRP"] = np.nan
        df = df[~df["Base MSRP"].isin([845000.0, 184400.0])]
        df["Base_MSRP_missing"] = df["Base MSRP"].isna().astype(int)

        msrp = df["Base MSRP"].copy()
        if all(c in df.columns for c in ["Make","Model","Model Year"]):
            msrp = msrp.fillna(df.groupby(["Make","Model","Model Year"])["Base MSRP"].transform("median"))
        if all(c in df.columns for c in ["Make","Model"]):
            msrp = msrp.fillna(df.groupby(["Make","Model"])["Base MSRP"].transform("median"))
        if "Make" in df.columns:
            msrp = msrp.fillna(df.groupby("Make")["Base MSRP"].transform("median"))
        msrp = msrp.fillna(msrp.median())
        df["Base MSRP"] = msrp

    # Save final cleaned output
    df.to_csv(clean_path, index=False)

    msrp_outliers = {}
    if "Base MSRP" in df.columns:
        msrp_outliers = find_outliers(df["Base MSRP"], upper_bound=200_000)

    return {
        "rows_before": before_count,
        "rows_after": after_count,
        "rows_cleaned": len(df),
        "inference_file": str(inference_path),
        "working_file": str(working_path),
        "cleaned_file": str(clean_path),
        "msrp_outliers": msrp_outliers
    }

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        body = request.get_json(force=True)
        input_rel = body.get("input", "01_raw/Electric_Vehicle_Population_Data.csv")
        inference_rel = body.get("inference", "03_inference/inference_dataset.csv")
        working_rel = body.get("working", "01_raw/working_dataset.csv")
        clean_rel = body.get("clean", "02_preprocessed/clean_dataset.csv")

        input_path = DATA_ROOT / input_rel
        inference_path = DATA_ROOT / inference_rel
        working_path = DATA_ROOT / working_rel
        clean_path = DATA_ROOT / clean_rel

        input_path.parent.mkdir(parents=True, exist_ok=True)
        inference_path.parent.mkdir(parents=True, exist_ok=True)
        working_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.parent.mkdir(parents=True, exist_ok=True)

        result = run_preprocessing(input_path, inference_path, working_path, clean_path)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
