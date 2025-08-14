from flask import Flask, request, render_template
import os
import pandas as pd
import requests

app = Flask(__name__)

# Get the model endpoint from environment variable (used in Kubernetes too)
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://localhost:5002/predict")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get user inputs
            input_data = {
                "Vehicle Make": [request.form["vehicle_make"]],
                "Model Year": [int(request.form["model_year"])],
                "Electric Vehicle Type": [request.form["vehicle_type"]],
                "Electric Range": [int(request.form["electric_range"])],
                "Base MSRP": [int(request.form["base_msrp"])],
                "City": [request.form["city"]],
                "County": [request.form["county"]],
            }

            df = pd.DataFrame(input_data)

            # Send to inference API
            response = requests.post(MODEL_ENDPOINT, json=df.to_dict(orient="records"))

            if response.status_code == 200:
                prediction = response.json()[0]["Clean Alternative Fuel Vehicle (CAFV) Eligibility"]
            else:
                error = response.json().get("error", "Unknown error")

        except Exception as e:
            error = str(e)

    return render_template("form.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
