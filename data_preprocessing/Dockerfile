FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src/ /app/src/

# Set envs
ENV DATA_DIR=/data
ENV PORT=8000

# Entry
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "src/preprocess.py"]
