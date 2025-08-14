FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY data_preprocessing/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ✅ Copy your source code from correct path
COPY data_preprocessing/src /app/src

# Entry
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/app/src/preprocess.py"]
