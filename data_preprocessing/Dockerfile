# data_preprocessing/Dockerfile
FROM python:3.11-slim

# Keep it minimal — no tini
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Paths are relative to build context (repo root)
COPY data_preprocessing/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./requirements.txt

COPY data_preprocessing/src/ ./src/

ENV DATA_DIR=/data \
    PORT=8000

EXPOSE 8000

# No init/entrypoint needed
CMD ["python", "src/preprocess.py"]
