# run_all.ps1
$ErrorActionPreference = "Stop"
Write-Host "Starting full pipeline:"

# 1) Build images
.\scripts\build_all.ps1

Write-Host "Enabling Minikube Ingress addon..."
minikube addons enable ingress | Out-Null
Start-Sleep -Seconds 10

Write-Host "Waiting for ingress-nginx namespace to be created..."
$timeout = New-TimeSpan -Seconds 120
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed -lt $timeout) {
    & kubectl get namespace ingress-nginx *> $null
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "Namespace 'ingress-nginx' not found, waiting..."
    Start-Sleep -Seconds 5
}
if ($sw.Elapsed -ge $timeout) {
    Write-Host "Error: Namespace was not found within the timeout period."
    exit 1
}
Write-Host "Namespace 'ingress-nginx' is ready."

# 2) Deploy to Kubernetes
.\scripts\deploy_all.ps1

Write-Host "Waiting for the ingress-nginx-controller deployment to be ready..."
& kubectl rollout status deployment/ingress-nginx-controller -n ingress-nginx --timeout=120s
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Deployment did not become ready within the timeout period."
    exit 1
}
Write-Host "All base Kubernetes resources deployed successfully."

# 3) Port-forward UI (optional)
Write-Host "Starting port-forward to access UI..."
.\scripts\start_ui.ps1
