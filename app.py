import os
import logging
import tempfile

from flask import Flask, request, jsonify
from google.cloud import storage
import tensorflow as tf
import numpy as np
import cv2

# -------------------------
# LOGGING
# -------------------------
logging.getLogger().setLevel(logging.INFO)
logging.info("Iniciando aplicación Flask...")

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
BUCKET_NAME = "shape-classifier-bucket"   # <-- tu bucket
MODEL_PATH = "shapes_model.h5"            # <-- tu archivo en el bucket (IMPORTANTE)

model = None
class_names = ["circle", "square", "triangle"]


def load_model_from_gcs():
    """Descarga el modelo desde GCS a /tmp y lo carga con Keras."""
    global model
    if model is not None:
        logging.info("Modelo ya estaba cargado en memoria, reutilizando.")
        return model

    logging.info(f"Cargando modelo desde GCS: bucket={BUCKET_NAME}, path={MODEL_PATH}")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)

    if not blob.exists():
        msg = f"El archivo {MODEL_PATH} no existe en el bucket {BUCKET_NAME}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    # Archivo temporal con extensión .h5 (MUY IMPORTANTE)
    fd, temp_path = tempfile.mkstemp(suffix=".h5", dir="/tmp")
    os.close(fd)

    logging.info(f"Descargando modelo a {temp_path} ...")
    blob.download_to_filename(temp_path)
    logging.info("Descarga completada, cargando modelo con tf.keras.models.load_model...")

    model = tf.keras.models.load_model(temp_path)
    logging.info("Modelo cargado correctamente.")

    return model


@app.route("/", methods=["GET"])
def root():
    logging.info("Health check en '/'")
    return "OK - Shape API funcionando", 200


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Petición recibida en /predict")

    # Cargar modelo lazy
    try:
        clf = load_model_from_gcs()
    except Exception as e:
        logging.error(f"Error cargando el modelo: {e}")
        return jsonify({"error": "No se pudo cargar el modelo", "details": str(e)}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    # Procesar imagen: 128x128x3 RGB (lo que espera el modelo)
    try:
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # BGR
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # a RGB

        img = img.astype("float32") / 255.0
        img = img.reshape(1, 128, 128, 3)
    except Exception as e:
        logging.error(f"Error procesando imagen: {e}")
        return jsonify({"error": "Invalid image", "details": str(e)}), 400

    # Predicción
    try:
        preds = clf.predict(img)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        shape = class_names[idx]
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    logging.info(f"Predicción: {shape} (confianza={confidence:.4f})")

    return jsonify({
        "shape": shape,
        "confidence": confidence
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Iniciando servidor local en puerto {port}...")
    app.run(host="0.0.0.0", port=port)





