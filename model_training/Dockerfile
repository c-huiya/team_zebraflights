FROM python:3.11-slim

# System deps (no tini)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps (Dockerfile at project root; build context = repo root)
COPY model_training/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY model_training/src/ ./src/

# Create expected data dirs
ENV MODEL_DIR=/data/04_model_output \
    ENCODED_DIR=/data/02_preprocessed/encoded_split \
    PORT=8000
RUN mkdir -p "$MODEL_DIR" "$ENCODED_DIR"

EXPOSE 8000

# No init/entrypoint
CMD ["python","src/train.py"]
