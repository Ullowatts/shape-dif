import logging
import os
import tempfile
from flask import Flask, request, jsonify
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

# -------------------------------
# CONFIG LOGGING
# -------------------------------
logging.getLogger().setLevel(logging.INFO)
logging.info("Arrancando la app Flask...")

app = Flask(__name__)

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
BUCKET_NAME = "shape-dif"
MODEL_PATH_GCS = "models/red_shape_classifier_private.keras"

model = None  # global


# -------------------------------
# FUNCIÓN PARA CARGAR MODELO
# -------------------------------
def load_model_from_gcs():
    global model
    if model is not None:
        logging.info("Modelo ya estaba cargado, reusando.")
        return model

    logging.info("Iniciando descarga del modelo desde GCS...")

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH_GCS)

        if not blob.exists():
            logging.error(f"ERROR: El archivo {MODEL_PATH_GCS} NO existe en bucket {BUCKET_NAME}")
            raise FileNotFoundError(f"{MODEL_PATH_GCS} not found in GCS")

        # Crear archivo temporal
        temp_model_path = tempfile.mktemp(suffix=".keras")
        blob.download_to_filename(temp_model_path)
        logging.info(f"Modelo descargado exitosamente a {temp_model_path}")

        # Cargar modelo
        logging.info("Cargando modelo en TensorFlow...")
        model = tf.keras.models.load_model(temp_model_path)
        logging.info("Modelo cargado correctamente.")

    except Exception as e:
        logging.error(f"ERROR al cargar el modelo: {str(e)}")
        raise e

    return model


# -------------------------------
# ENDPOINT PARA PROBAR ARRANQUE
# -------------------------------
@app.route("/", methods=["GET"])
def health_check():
    logging.info("Petición recibida en '/'")
    return "OK - Servicio activo", 200


# -------------------------------
# ENDPOINT DE PREDICCIÓN
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Petición recibida en /predict")

    try:
        model = load_model_from_gcs()
    except Exception as e:
        logging.error("No se pudo cargar el modelo.")
        return jsonify({"error": "Modelo no cargado", "details": str(e)}), 500

    if "file" not in request.files:
        logging.warning("No se recibió archivo en la petición.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))  # Ajusta al tamaño de tu modelo
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        logging.error(f"Error procesando imagen: {str(e)}")
        return jsonify({"error": "Invalid image", "details": str(e)}), 400

    try:
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    logging.info(f"Predicción exitosa: class={predicted_class} confidence={confidence}")

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence
    })


# -------------------------------
# MAIN PARA DEBUG LOCAL
# -------------------------------
if __name__ == "__main__":
    logging.info("Iniciando servidor local en puerto 8080...")
    app.run(host="0.0.0.0", port=8080)



