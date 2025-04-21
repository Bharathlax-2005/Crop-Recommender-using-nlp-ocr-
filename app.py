from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import re
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import logging
logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the model and scaler
df = pd.read_csv("Crop_Recommendation.csv")
df = df.rename(columns={
    "pH_Value": "pH",
    "Nitrogen": "Nitrogen (kg/ha)",
    "Phosphorus": "Phosphorus (kg/ha)",
    "Potassium": "Potassium (kg/ha)"
})
X = df[[ "pH","Nitrogen (kg/ha)", "Phosphorus (kg/ha)", "Potassium (kg/ha)"]]
y = df["Crop"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier()
model.fit(X_scaled, y)

# OCR and prediction functions
def extract_values(text):
    data = {
        "pH": None,
        "Nitrogen (kg/ha)": None,
        "Phosphorus (kg/ha)": None,
        "Potassium (kg/ha)": None
    }

    # pH
    ph_match = re.search(r"pH.*?([\d.]+)", text, re.IGNORECASE)
    if ph_match:
        data["pH"] = float(ph_match.group(1))

    # Nitrogen
    n_match = re.search(r"N\s*\(kg/?ha\).*?([\d.]+)", text, re.IGNORECASE)
    if n_match:
        data["Nitrogen (kg/ha)"] = float(n_match.group(1))

    # Phosphorus
    p_match = re.search(r"P[\W_]*[O0][\W_,\d]*.*?([\d.]+)", text, re.IGNORECASE)
    if p_match:
        data["Phosphorus (kg/ha)"] = float(p_match.group(1))

    # Potassium
    k_match = re.search(r"K[\W_,0O]*\(?\s*Kg\s*/?\s*ha\)?[\W_]*([\d.]+)", text, re.IGNORECASE)
    if k_match:
        data["Potassium (kg/ha)"] = float(k_match.group(1))

    return data

def predict_crop(soil_data):
    input_df = pd.DataFrame([soil_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Perform OCR
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)

        # Extract values
        soil_data = extract_values(text)

        # Validate data
        if None in soil_data.values():
            return render_template("error.html", message="Could not extract all required values from the image.")

        # Predict crop
        recommended_crop = predict_crop(soil_data)
        return render_template("result.html", soil_data=soil_data, crop=recommended_crop)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            file = request.files["image"]
            if file:
                file_path = save_uploaded_file(file)
                soil_data = process_image(file_path)
                if not soil_data:
                    raise ValueError("Could not extract all required values from the image.")
                recommended_crop = predict_crop(soil_data)
                return render_template("results.html", soil_data=soil_data, recommended_crop=recommended_crop)
        return render_template("index.html")
    except Exception as e:
        logging.error(f"Error: {e}")
        return render_template("error.html", message="An error occurred while processing your request.")