FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training source code
COPY src/ ./src/

# Set environment variables
ENV MODEL_DIR=/data/04_model_output \
    ENCODED_DIR=/data/02_preprocessed/encoded_split \
    PORT=8000

# Use tini to handle PID 1 correctly (good practice)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run Flask server (serve train.py as API)
CMD ["python", "src/train.py"]
