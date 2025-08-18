from flask import Flask, request, jsonify
import os, sys, tempfile, shutil, json
import pandas as pd
import joblib
from pathlib import Path
import traceback
import logging
from flask_cors import CORS

# ------------------------------
# Project + paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [BASE_DIR.parent, BASE_DIR.parent.parent, Path.cwd()]

# Allow override via env var
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else None
if not PROJECT_ROOT:
    for c in CANDIDATES:
        if (c / "data" / "04_model_output").exists():
            PROJECT_ROOT = c
            break

if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root. Set PROJECT_ROOT env var to override.")

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "data" / "04_model_output" / "model.joblib"))

# ------------------------------
# Import preprocessing module (delegated)
# ------------------------------
PREPROC_DIR = PROJECT_ROOT / "data_preprocessing" / "src"
if PREPROC_DIR.exists():
    sys.path.insert(0, str(PREPROC_DIR))  # ensure our preprocess.py is first
else:
    raise RuntimeError(f"Preprocessing module not found at {PREPROC_DIR}")

from preprocess import run_preprocessing  # noqa: E402

# ------------------------------
# Load model (no local preprocessing here)
# ------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# Defaults (overridable via env)
DEFAULT_INFERENCE_REL = os.getenv("INFERENCE_INPUT", "03_inference/inference_dataset.csv")
PREDICTION_COLUMN = os.getenv("PREDICTION_COLUMN", "prediction")

# Parameters passed through to run_preprocessing
# NOTE: we run in "inference mode" (do_encode_split=False) → no target is required
UNKNOWN_POLICY = os.getenv("UNKNOWN_POLICY", "negative")   # 'negative' or 'drop'
OUTLIER_STRATEGY = os.getenv("OUTLIER_STRATEGY", "clip")   # 'clip' or 'remove'


def _respond_error(msg, code=400, extra=None):
    payload = {"status": "error", "message": msg}
    if extra is not None:
        payload["details"] = extra
    return jsonify(payload), code


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "Model Inference API",
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "model_path": str(MODEL_PATH)
    }), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle CORS preflight quickly
    if request.method == "OPTIONS":
        return ("", 204)

    temp_dir = tempfile.mkdtemp(prefix="inference_")
    try:
        # 1) Parse JSON (accept a single object or {"rows":[...]})
        try:
            raw = request.get_json(force=True)
        except Exception:
            return _respond_error("Request body must be valid JSON with Content-Type: application/json.", 400)

        body, rows = {}, None
        if raw is None:
            body = {}
        elif isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            body = raw
            # allow single object as shorthand
            if "rows" in body and isinstance(body["rows"], list):
                rows = body["rows"]
            elif body and not body.get("input"):
                # treat the dict itself as one record if it looks like data
                rows = [body]
                body = {}
        else:
            return _respond_error("JSON must be an object or a list of row-objects.", 400)

        # 2) Determine input source
        input_rel = body.get("input") if isinstance(body, dict) else None

        if rows is not None:
            if not isinstance(rows, list) or (rows and not isinstance(rows[0], dict)):
                return _respond_error("'rows' must be a list of JSON records (objects).", 400)
            src_csv = Path(temp_dir) / "input_inline.csv"
            pd.DataFrame(rows).to_csv(src_csv, index=False)
        else:
            src_rel = input_rel or DEFAULT_INFERENCE_REL
            src_csv = DATA_DIR / src_rel
            if not src_csv.exists():
                return _respond_error(f"Source file not found: {src_csv}", 404)

        # 3) Delegate ALL preprocessing (inference mode: no target, no split)
        temp_clean_csv = Path(temp_dir) / "cleaned.csv"
        prep_result = run_preprocessing(
            input_path=str(src_csv),
            output_path=str(temp_clean_csv),
            outlier_strategy=OUTLIER_STRATEGY,
            do_encode_split=False,   # inference mode
            target_col=None,         # not needed in inference
            unknown_policy=UNKNOWN_POLICY,
            encoded_dir=None
        )

        # 4) Load cleaned features
        cleaned_path = Path(prep_result.get("cleaned_file", temp_clean_csv))
        if not cleaned_path.exists():
            return _respond_error(
                "Preprocessing completed but cleaned file was not found.",
                500,
                extra={"expected_cleaned_file": str(cleaned_path)}
            )
        X = pd.read_csv(cleaned_path)

        # 5) Predict (the model should encapsulate any needed encoders)
        try:
            preds = model.predict(X)
        except Exception as ex:
            logging.error("Model predict() failed:\n%s", traceback.format_exc())
            return _respond_error(
                "Model prediction failed. Ensure your saved model expects exactly the "
                "columns produced by preprocessing (e.g., a Pipeline with encoders).",
                500,
                extra={
                    "exception": repr(ex),
                    "cleaned_shape": list(X.shape),
                    "cleaned_columns": list(X.columns)
                }
            )

        result = pd.DataFrame({PREDICTION_COLUMN: preds})

        # 6) Optional: save predictions
        saved_path = None
        save_to = body.get("save_to") if isinstance(body, dict) else None
        if save_to:
            out_path = DATA_DIR / save_to
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.suffix.lower() == ".json":
                out_path.write_text(result.to_json(orient="records", indent=2))
            else:
                result.to_csv(out_path, index=False)
            saved_path = str(out_path)

        # 7) Respond
        return jsonify({
            "status": "ok",
            "n_rows": int(len(result)),
            "prediction_column": PREDICTION_COLUMN,
            "saved_to": saved_path,
            "predictions": json.loads(result.to_json(orient="records")),
            "preprocessing_summary": {
                "cleaned_file": str(cleaned_path),
                "rows_cleaned": prep_result.get("rows_cleaned"),
            }
        })

    except Exception as e:
        logging.error("Prediction error:\n%s", traceback.format_exc())
        return _respond_error(f"{type(e).__name__}: {str(e)}", 500)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # In containers, prefer gunicorn in production; this is for local dev.
    app.run(host="0.0.0.0", port=5002)
