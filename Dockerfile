FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.h5 .
COPY app.py .

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
