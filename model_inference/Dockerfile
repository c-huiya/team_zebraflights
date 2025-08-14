FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY model_inference/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY model_inference/src/ ./src/
COPY data/ ./data/
COPY data_preprocessing/src/ ./data_preprocessing/src/

EXPOSE 5002
CMD ["python", "./src/inference.py"]
