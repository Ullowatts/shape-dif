FROM python:3.10

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app.py .

# Comando de arranque
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
