# deploy_all.ps1
Write-Host "Deploying all Kubernetes resources..."

$minikubeExe = Get-Command minikube.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path
if (-not $minikubeExe) {
    Write-Host "ERROR: minikube.exe not found on PATH. Install it or add its folder to PATH."
    exit 1
}

Write-Host "Starting/ensuring cluster is running..."
& $minikubeExe start --driver=docker

# Enable ingress if needed
& kubectl get ns ingress-nginx *> $null
$nsExists = ($LASTEXITCODE -eq 0)
if (-not $nsExists) {
    Write-Host "Enabling ingress addon..."
    & $minikubeExe addons enable ingress
}

# Wait for ingress controller deployment to appear
Write-Host "Waiting for ingress controller deployment to appear..."
$timeout = New-TimeSpan -Seconds 60
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed -lt $timeout) {
    & kubectl get deploy ingress-nginx-controller -n ingress-nginx *> $null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 1
}
if ($sw.Elapsed -ge $timeout) {
    Write-Host "controller deploy missing; check 'minikube addons enable ingress' output"
    exit 1
}

Write-Host "Waiting for ingress controller rollout..."
& kubectl rollout status deploy/ingress-nginx-controller -n ingress-nginx --timeout=300s

Write-Host "Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/data_preprocessing_deployment.yaml
kubectl apply -f k8s/data_preprocessing_service.yaml
kubectl apply -f k8s/model_training_deployment.yaml
kubectl apply -f k8s/model_training_service.yaml
kubectl apply -f k8s/model_inference_deployment.yaml
kubectl apply -f k8s/model_inference_service.yaml
kubectl apply -f k8s/ui_deployment.yaml
kubectl apply -f k8s/ui_service.yaml

# Ingress last
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/ingress-controller-svc.yaml

$miniIp  = & $minikubeExe ip
$nodePort = & kubectl -n ingress-nginx get svc ingress-nginx-controller -o jsonpath='{.spec.ports[?(@.name=="http")].nodePort}'
Write-Host ""
Write-Host "All Kubernetes resources deployed successfully."
Write-Host "Ingress controller NodePort: http://${miniIp}:${nodePort}/"
Write-Host "Tip: For a permanent LAN URL on port 80, map your PC:80 -> ${miniIp}:${nodePort} with Windows 'portproxy'."
