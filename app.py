from flask import Flask, render_template, request
import pandas as pd
import joblib

from model.utils import validate_input, prepare_features

app = Flask(__name__)

# Load trained model pipeline
model = joblib.load("model/framingham_logreg.joblib")
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # Collect form data
    data = {
        "male": int(request.form["male"]),
        "age": float(request.form["age"]),
        "education": float(request.form["education"]),
        "currentSmoker": int(request.form["currentSmoker"]),
        "cigsPerDay": float(request.form["cigsPerDay"]),
        "BPMeds": int(request.form["BPMeds"]),
        "prevalentStroke": int(request.form["prevalentStroke"]),
        "prevalentHyp": int(request.form["prevalentHyp"]),
        "diabetes": int(request.form["diabetes"]),
        "totChol": float(request.form["totChol"]),
        "sysBP": float(request.form["sysBP"]),
        "diaBP": float(request.form["diaBP"]),
        "BMI": float(request.form["BMI"]),
        "heartRate": float(request.form["heartRate"]),
        "glucose": float(request.form["glucose"])
    }

    input_df = pd.DataFrame([data])

    # Validate & prepare
    input_df = validate_input(input_df)
    input_df = prepare_features(input_df)

    # Predict probability
    risk_prob = model.predict_proba(input_df)[0][1]
    risk_percent = round(risk_prob * 100, 2)

    # Risk thresholds (medical-style)
    if risk_percent < 20:
        risk_level = "Low Risk"
    elif risk_percent < 40:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return render_template(
        "result.html",
        probability=risk_percent,
        risk=risk_level
    )
if __name__ == "__main__":
    app.run()
