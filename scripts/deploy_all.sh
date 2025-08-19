#!/bin/bash

echo "Deploying all Kubernetes resources..."

# Namespace must be applied first
kubectl apply -f k8s/namespace.yaml

# Config & storage
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml

# Preprocessing
kubectl apply -f k8s/data_preprocessing_deployment.yaml
kubectl apply -f k8s/data_preprocessing_service.yaml

# Model training
kubectl apply -f k8s/model_training_deployment.yaml
kubectl apply -f k8s/model_training_service.yaml

# Model inference
kubectl apply -f k8s/model_inference_deployment.yaml
kubectl apply -f k8s/model_inference_service.yaml

# UI
kubectl apply -f k8s/ui_deployment.yaml
kubectl apply -f k8s/ui_service.yaml

# Ingress (should come last)
kubectl apply -f k8s/ingress.yaml

echo "All Kubernetes resources deployed successfully."
