from flask import Flask, request, jsonify
import os, sys, tempfile, shutil
import pandas as pd
import joblib
from preprocess import run_preprocessing

# --- Resolve model path dynamically ---
# Start from the current file's directory (/app/src in Docker, something like .../model_inference/src locally)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to reach /app (in Docker) or your project root locally
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Build the path to the model inside the data folder
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "04_model_output", "model.joblib")

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
