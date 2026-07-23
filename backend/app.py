
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# ---------------------------------------------------
# Initialize Flask App
# ---------------------------------------------------

app = Flask("Superkart Sales Prediction Model")

# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------

model = joblib.load("random_forest_sales_forecast.joblib")

# ---------------------------------------------------
# Expected Features
# ---------------------------------------------------

FEATURES = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Age"
]

# ---------------------------------------------------
# Home Route
# ---------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "application": "SuperKart Sales Forecast API",
        "status": "Running",
        "model": "Random Forest Regressor",
        "version": "1.0"
    })

# ---------------------------------------------------
# Single Prediction
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON payload received."
            }), 400

        missing = [col for col in FEATURES if col not in data]

        if len(missing) > 0:

            return jsonify({
                "error": "Missing input fields.",
                "missing_fields": missing
            }), 400

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "Predicted_Product_Store_Sales_Total": round(float(prediction), 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ---------------------------------------------------
# Batch Prediction
# ---------------------------------------------------

@app.route("/predict_batch", methods=["POST"])
def predict_batch():

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "CSV file not uploaded."
            }), 400

        file = request.files["file"]

        df = pd.read_csv(file)

        missing = [col for col in FEATURES if col not in df.columns]

        if len(missing) > 0:

            return jsonify({
                "error": "Missing columns in uploaded CSV.",
                "missing_columns": missing
            }), 400

        predictions = model.predict(df)

        output = df.copy()

        output["Predicted_Product_Store_Sales_Total"] = predictions.round(2)

        return jsonify(
            output.to_dict(orient="records")
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ---------------------------------------------------
# Run Flask
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
