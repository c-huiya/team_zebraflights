from flask import Flask, request, jsonify
import os, sys, tempfile, shutil
import pandas as pd
import joblib
from pathlib import Path
import os, sys

# --- Resolve project root both locally and in Docker ---
BASE_DIR = Path(__file__).resolve().parent

# Try a few sensible candidates:
candidates = [
    BASE_DIR.parent,          # /app  (Docker) or .../project/model_inference (local)
    BASE_DIR.parent.parent,   # .../project (local)
    Path.cwd(),               # current working directory, just in case
]

# Prefer the first folder that contains 'data/04_model_output'
PROJECT_ROOT = None
for c in candidates:
    if (c / "data" / "04_model_output").exists():
        PROJECT_ROOT = c
        break

# Allow override via env var if needed
if os.getenv("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not locate project root. "
        "Checked: "
        + ", ".join(str(p) for p in candidates)
        + ". Set PROJECT_ROOT env var to override."
    )

MODEL_PATH = PROJECT_ROOT / "data" / "04_model_output" / "model.joblib"

# --- Ensure we can import preprocess.py ---
PREPROC_DIR = PROJECT_ROOT / "data_preprocessing" / "src"
if PREPROC_DIR.exists():
    sys.path.insert(0, str(PREPROC_DIR))
else:
    raise RuntimeError(f"Preprocessing module not found at {PREPROC_DIR}")

from preprocess import run_preprocessing  # now safe to import


# --- Diagnostics (helpful if it still fails) ---
print("=== Startup path diagnostics ===", flush=True)
print(f"__file__          : {__file__}", flush=True)
print(f"BASE_DIR          : {BASE_DIR}", flush=True)
print(f"PROJECT_ROOT      : {PROJECT_ROOT}", flush=True)
print(f"Resolved MODEL_PATH: {MODEL_PATH}", flush=True)
print("Contents of model dir:", flush=True)
try:
    for name in os.listdir(os.path.dirname(MODEL_PATH)):
        print("  ", name, flush=True)
except FileNotFoundError:
    print("  (Model directory not found)", flush=True)
print("=== End diagnostics ===", flush=True)

# --- Load the model or fail fast ---
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Ensure it exists in 'data/04_model_output' relative to project root."
    )

model = joblib.load(MODEL_PATH)

# --- Flask app init ---
app = Flask(__name__)


def run_inference(pipeline, data: pd.DataFrame):
    return pipeline.predict(data)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Model Inference API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    temp_dir = tempfile.mkdtemp()
    try:
        input_data = request.get_json()
        df_raw = pd.DataFrame(input_data)

        temp_csv_path = os.path.join(temp_dir, "preprocessed_data.csv")
        run_preprocessing(df_raw, temp_csv_path)
        df_clean = pd.read_csv(temp_csv_path)

        if "Base MSRP" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Base MSRP"])

        preds = run_inference(model, df_clean)
        df_raw["Clean Alternative Fuel Vehicle (CAFV) Eligibility"] = preds

        return df_raw.to_json(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
