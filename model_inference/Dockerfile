FROM python:3.11-slim

WORKDIR /app

COPY model_inference/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY model_inference/src/ ./src/
COPY data_preprocessing/src/ ./data_preprocessing/src/
COPY data/ ./data/

ENV PYTHONPATH=/app/src:/app/data_preprocessing/src

EXPOSE 5002
CMD ["python", "/app/src/inference.py"]