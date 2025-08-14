FROM python:3.11-slim

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training source code
COPY src/ ./src/

# Set default env vars (overridable in Kubernetes)
ENV DATA_PATH=/data/02_preprocessed/clean_dataset.csv \
    MODEL_DIR=/model_output

# Run the training script
CMD ["python", "src/train.py"]