import os
import logging
import tempfile

from flask import Flask, request, jsonify
from google.cloud import storage
import tensorflow as tf
import numpy as np
import cv2

# -----------------------------------
# CONFIGURAR LOGS
# -----------------------------------
logging.getLogger().setLevel(logging.INFO)
logging.info("Iniciando aplicación Flask...")

app = Flask(__name__)

# -----------------------------------
# CONFIGURACIÓN DE GCS Y MODELO
# -----------------------------------
BUCKET_NAME = "shape-classifier-bucket"      # <-- tu bucket
MODEL_PATH = "shapes_model.keras"            # <-- tu archivo en el bucket

model = None  # se cargará bajo demanda
class_names = ["circle", "square", "triangle"]  # ajusta si tu modelo usa otras


def load_model_from_gcs():
    """
    Descarga el modelo desde Cloud Storage a un archivo temporal
    y lo carga con tf.keras.
    """
    logging.info("Intentando cargar modelo desde GCS...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)

    if not blob.exists():
        msg = f"El archivo {MODEL_PATH} no existe en el bucket {BUCKET_NAME}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    # Archivo temporal en /tmp (recomendado en Cloud Run)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras", dir="/tmp") as tmp:
        temp_path = tmp.name

    logging.info(f"Descargando modelo a {temp_path} ...")
    blob.download_to_filename(temp_path)
    logging.info("Descarga completada, cargando con tf.keras.models.load_model...")

    model_loaded = tf.keras.models.load_model(temp_path)
    logging.info("Modelo cargado correctamente desde GCS.")

    return model_loaded


def get_model():
    """
    Carga el modelo solo la primera vez que se llama.
    Si ya está cargado, reutiliza la misma instancia.
    """
    global model
    if model is None:
        model = load_model_from_gcs()
    return model


# -----------------------------------
# ENDPOINT DE SALUD
# -----------------------------------
@app.route("/", methods=["GET"])
def root():
    logging.info("Health check en '/'")
    return "OK - Shape API funcionando", 200


# -----------------------------------
# ENDPOINT DE PREDICCIÓN
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Petición recibida en /predict")

    # Cargar modelo (lazy)
    try:
        clf = get_model()
    except Exception as e:
        logging.error(f"Error cargando el modelo: {e}")
        return jsonify({
            "error": "No se pudo cargar el modelo",
            "details": str(e)
        }), 500

    if "file" not in request.files:
        logging.warning("No se recibió archivo en el campo 'file'")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    try:
        # Leer y procesar con OpenCV en escala de grises
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img.reshape(1, 128, 128, 1) / 255.0
    except Exception as e:
        logging.error(f"Error procesando la imagen: {e}")
        return jsonify({
            "error": "Invalid image",
            "details": str(e)
        }), 400

    try:
        preds = clf.predict(img)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        shape = class_names[idx]
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

    logging.info(f"Predicción: {shape} (confianza={confidence:.4f})")

    return jsonify({
        "shape": shape,
        "confidence": confidence
    })


# -----------------------------------
# MAIN PARA PRUEBA LOCAL
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Iniciando servidor local en puerto {port}...")
    app.run(host="0.0.0.0", port=port)



