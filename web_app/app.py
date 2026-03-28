import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime

app = Flask(__name__)

# Determine absolute path to the model dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "blood_group_fingerprint_model.h5")

try:
    # load trained model
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

classes = ['A+','A-','AB+','AB-','B+','B-','O+','O-']

@app.route("/", methods=["GET","POST"])
def index():

    result = None
    confidence = None
    mobile_no = None
    address = None
    timestamp = None
    image_quality = None

    if request.method == "POST":

        file = request.files["file"]

        path = "temp.jpg"
        file.save(path)

        img = image.load_img(path, color_mode="rgb", target_size=(224,224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model is not None:
            prediction = model.predict(img_array)
            max_idx = np.argmax(prediction)
            result = classes[max_idx]
            confidence = round(float(np.max(prediction) * 100), 1)
        else:
            result = "Error: Model not loaded"
            confidence = 0.0

        mobile_no = request.form.get("mobile_no", "Not Provided")
        address = request.form.get("address", "Not Provided")
        timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        image_quality = "EXCELLENT" # Placeholder for now

    return render_template("index.html", result=result, confidence=confidence, mobile_no=mobile_no, address=address, timestamp=timestamp, image_quality=image_quality)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)