from flask import Flask
import os
import logging

logging.getLogger().setLevel(logging.INFO)
logging.info("Arrancando app mínima...")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    logging.info("Petición a '/' recibida")
    return "Hola desde Cloud Run (API mínima)!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Iniciando servidor local en puerto {port}...")
    app.run(host="0.0.0.0", port=port)




