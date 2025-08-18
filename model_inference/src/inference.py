from flask import Flask, request, jsonify
import os, sys, tempfile, shutil, json
import pandas as pd
import joblib
from pathlib import Path
import traceback
import logging

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
    sys.path.insert(0, str(PREPROC_DIR))
else:
    raise RuntimeError(f"Preprocessing module not found at {PREPROC_DIR}")

from preprocess import run_preprocessing  # noqa: E402

# ------------------------------
# Load model
# ------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Defaults (overridable via env)
DEFAULT_INFERENCE_REL = os.getenv("INFERENCE_INPUT", "03_inference/inference_dataset.csv")
PREDICTION_COLUMN = os.getenv("PREDICTION_COLUMN", "prediction")

# Parameters passed through to run_preprocessing
# NOTE: target_col is deliberately ignored during inference
UNKNOWN_POLICY = os.getenv("UNKNOWN_POLICY", "negative")   # 'negative' or 'drop'
OUTLIER_STRATEGY = os.getenv("OUTLIER_STRATEGY", "clip")   # 'clip' or 'remove'


def _respond_error(msg, code=400, extra=None):
    """Consistent error responses; msg can be str or a dict with details."""
    payload = {"status": "error", "message": msg}
    if extra:
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


@app.route("/predict", methods=["POST"])
def predict():
    temp_dir = tempfile.mkdtemp(prefix="inference_")
    try:
        # 1) Parse JSON
        try:
            raw = request.get_json(force=True)  # may be dict or list
        except Exception:
            return _respond_error("Request body must be valid JSON with Content-Type: application/json.", 400)

        body, rows = {}, None
        if raw is None:
            body = {}
        elif isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            body = raw
            rows = body.get("rows")
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

        # 3) Preprocess in "inference mode" (NO target, NO split)
        temp_clean_csv = Path(temp_dir) / "cleaned.csv"
        prep_result = run_preprocessing(
            input_path=str(src_csv),
            output_path=str(temp_clean_csv),
            outlier_strategy=OUTLIER_STRATEGY,
            do_encode_split=False,     # IMPORTANT: avoid target-dependent code paths
            target_col=None,           # IMPORTANT: do not require a target at inference
            unknown_policy=UNKNOWN_POLICY,
            encoded_dir=None
        )

        # Sanity: ensure cleaned file exists
        cleaned_path = Path(prep_result.get("cleaned_file", temp_clean_csv))
        if not cleaned_path.exists():
            return _respond_error(
                "Preprocessing completed but cleaned file was not found.",
                500,
                extra={"expected_cleaned_file": str(cleaned_path)}
            )

        # 4) Load cleaned features
        X = pd.read_csv(cleaned_path)

        # 5) Predict
        try:
            preds = model.predict(X)
        except Exception as ex:
            logging.error("Model predict() failed:\n%s", traceback.format_exc())
            return _respond_error(
                "Model prediction failed. Check that your saved model expects the same columns "
                "produced by preprocessing (and that categorical encoders use handle_unknown).",
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
                # The following keys exist only in training mode; kept here for consistency:
                "encoded_artifacts": prep_result.get("encoded_artifacts"),
                "n_features": prep_result.get("n_features"),
                "class_balance": prep_result.get("class_balance"),
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
