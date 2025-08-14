from flask import Flask, request, jsonify
import os
import sys
import pandas as pd
import joblib
import tempfile
import shutil

# Dynamically add path to preprocessing module
# This is a good practice to ensure the import works regardless of where the script is executed.
# It navigates up two directories to find the 'data_preprocessing' folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_preprocessing", "src")))

from data_preprocessing.preprocess import run_preprocessing 

# Define model path
# os.path.abspath resolves the full path, making it more robust.
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "04_model_output", "model.joblib"))

# Load the trained model at startup to avoid reloading it on every request.
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1) # Exit if the model cannot be loaded

# Flask app initialization
app = Flask(__name__)

def run_inference(pipeline, data):
    """
    A helper function to run predictions.
    
    Args:
        pipeline: The trained scikit-learn pipeline or model.
        data (pd.DataFrame): The preprocessed data to predict on.
    
    Returns:
        np.array: An array of predictions.
    """
    return pipeline.predict(data)

@app.route("/", methods=["GET"])
def home():
    """
    Returns a status message to indicate the API is running.
    """
    return jsonify({"status": "Model Inference API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON data, preprocesses it, and returns predictions.
    
    Returns:
        JSON: The original data with an added prediction column.
    """
    # Use a temporary directory for processing files
    temp_dir = tempfile.mkdtemp()
    
    try:
        input_data = request.get_json()
        df_raw = pd.DataFrame(input_data)

        # Use a temporary file path for the preprocessing function's output
        temp_csv_path = os.path.join(temp_dir, 'preprocessed_data.csv')

        # Call the existing preprocessing function, writing to the temporary file
        run_preprocessing(df_raw, temp_csv_path)

        # Read the preprocessed data back into a DataFrame
        df_clean = pd.read_csv(temp_csv_path)

        # Drop the target if it exists, as it's not needed for inference
        if "Base MSRP" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Base MSRP"])

        # Predict using the loaded model
        predictions = run_inference(model, df_clean)

        # Attach predictions to the original raw DataFrame
        df_raw["Clean Alternative Fuel Vehicle (CAFV) Eligibility"] = predictions

        # Return the original data with predictions as JSON
        return df_raw.to_json(orient="records")
    
    except Exception as e:
        # Provide a meaningful error message if something goes wrong
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Ensure the temporary directory is removed, regardless of success or failure
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Start the Flask development server on host "0.0.0.0" to make it accessible outside the container
    app.run(host="0.0.0.0", port=5002)
