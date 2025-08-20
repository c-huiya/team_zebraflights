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
            # ----- allow MSRP to be missing / 0 -----
            msrp_raw = request.form["base_msrp"].strip()
            if msrp_raw == "":
                msrp_val = None
            else:
                try:
                    msrp_val = int(msrp_raw)
                    if msrp_val <= 0:
                        msrp_val = None        # treat zero / negative as missing
                except ValueError:
                    msrp_val = None            # non-numeric → missing

            # Get user inputs
            input_data = {
                # ----- geography -----
                "County":              request.form["county"].strip() ,
                "City":                request.form["city"].strip() ,
                "State":               request.form["state"].strip() ,
                "Postal Code":         request.form["postal_code"].strip(),   # keep as string

                # ----- vehicle info -----
                "Make":                request.form["make"].strip() ,
                "Model":               request.form["model"].strip() ,
                "Electric Vehicle Type": request.form["vehicle_type"].strip() ,

                # ----- numeric -----
                "Base MSRP":           msrp_val,
                "Legislative District":request.form["legislative_district"].strip(),

                # ----- utility -----
                "Electric Utility":    request.form["electric_utility"].strip() ,
            }

            df = pd.DataFrame([input_data])

            # Send to inference API
            response = requests.post(MODEL_ENDPOINT, json=df.to_dict(orient="records"))

            if response.status_code == 200:
                resp = response.json()

                # For current inference service (dict with "predictions")
                if isinstance(resp, dict) and "predictions" in resp:
                    raw_pred = resp["predictions"][0].get("prediction", None)
                # For older style (list of dicts)
                elif isinstance(resp, list) and len(resp) > 0:
                    raw_pred = resp[0].get("Clean Alternative Fuel Vehicle (CAFV) Eligibility", None)
                else:
                    raw_pred = None

                if raw_pred is None:
                    error = "No prediction found in response."
                else:
                    prediction = raw_pred
            else:
                try:
                    # show whatever keys the backend sent
                    backend = response.json()
                    error = backend.get("message") or backend.get("error") \
                            or f"Model service error (HTTP {response.status_code})"
                except ValueError:
                    # non-JSON body
                    error = f"Model service returned HTTP {response.status_code} and non-JSON payload"

        except Exception as e:
            import traceback
            error = traceback.format_exc()


    if prediction:
        return render_template("result.html", eligibility=prediction, record=input_data)
    else:
        return render_template("form.html", prediction=None, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
