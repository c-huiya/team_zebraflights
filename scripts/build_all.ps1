# build_all.ps1

Write-Host "Building Docker images for all modules..."

# Preprocessing
docker build -t preprocess:latest -f data_preprocessing/Dockerfile .

# Model Training
docker build -t training:latest -f model_training/Dockerfile .

# Model Inference
docker build -t model-inference:latest -f model_inference/Dockerfile .

# UI
docker build -t ui:latest -f ui/Dockerfile .

Write-Host "All images built successfully."