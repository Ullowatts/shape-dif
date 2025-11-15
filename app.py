from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from google.cloud import storage
import cv2
import tempfile

app = Flask(__name__)

BUCKET_NAME = "shape-classifier-bucket"
MODEL_PATH = "shapes_model.keras"

def load_model_from_gcs():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)

    temp_file = tempfile.NamedTemporaryFile()
    blob.download_to_filename(temp_file.name)

    model = tf.keras.models.load_model(temp_file.name)
    return model

model = load_model_from_gcs()
class_names = ["circle", "square", "triangle"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img = request.files["file"].read()
    npimg = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, 128, 128, 1) / 255.0

    prediction = model.predict(img)
    idx = np.argmax(prediction)

    return jsonify({
        "shape": class_names[idx],
        "confidence": float(np.max(prediction))
    })
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


