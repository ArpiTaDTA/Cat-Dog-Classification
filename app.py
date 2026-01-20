from flask import Flask, render_template, request
import joblib
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

models = {
    "svm": joblib.load("models/svm.pkl"),
    "rf": joblib.load("models/rf.pkl"),
    "lr": joblib.load("models/lr.pkl")
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        model_name = request.form["model"]

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath, 0)
        img = cv2.resize(img, (64,64)).flatten().reshape(1, -1)

        prediction = models[model_name].predict(img)
        result = "Dog 🐶" if prediction[0] == 1 else "Cat 🐱"

        image_path = filepath

    return render_template("index.html", result=result, image_path=image_path)

app.run(debug=True)

