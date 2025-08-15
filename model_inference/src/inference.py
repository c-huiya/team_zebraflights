from flask import Flask, request, jsonify
import os, sys, tempfile, shutil
import pandas as pd
import joblib
from pathlib import Path

# --- Resolve project root both locally and in Docker ---
BASE_DIR = Path(__file__).resolve().parent
candidates = [BASE_DIR.parent, BASE_DIR.parent.parent, Path.cwd()]

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else None
if not PROJECT_ROOT:
    for c in candidates:
        if (c / "data" / "04_model_output").exists():
            PROJECT_ROOT = c
            break

if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not locate project root. "
        "Set PROJECT_ROOT env var to override."
    )

MODEL_PATH = PROJECT_ROOT / "data" / "04_model_output" / "model.joblib"

# --- Ensure we can import preprocess.py ---
PREPROC_DIR = PROJECT_ROOT / "data_preprocessing" / "src"
if PREPROC_DIR.exists():
    sys.path.insert(0, str(PREPROC_DIR))
else:
    raise RuntimeError(f"Preprocessing module not found at {PREPROC_DIR}")

from preprocess import run_preprocessing

# --- Load model ---
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

PREDICTION_COLUMN = os.getenv("PREDICTION_COLUMN", "prediction")
OUTLIER_STRATEGY = os.getenv("OUTLIER_STRATEGY", "clip")  # "clip" or "remove"

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Model Inference API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    temp_dir = tempfile.mkdtemp()
    try:
        # use existing inference CSV, but don’t overwrite it
        src = PROJECT_ROOT / "data" / "03_inference" / "inference_dataset.csv"
        if not src.exists():
            return jsonify({"error": f"Source file not found: {src}"}), 404

        inp = os.path.join(temp_dir, "input.csv")
        clean = os.path.join(temp_dir, "clean.csv")

        shutil.copy(src, inp)

        # NEW API: only input + output (no working/inference paths)
        run_preprocessing(inp, clean, outlier_strategy=OUTLIER_STRATEGY)

        df = pd.read_csv(clean)
        preds = model.predict(df)
        df[PREDICTION_COLUMN] = preds
        return df.to_json(orient="records"), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
