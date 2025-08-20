kubectl port-forward -n ingress-nginx service/ingress-nginx-controller 8080:80
Write-Host "Port-forwarding started. Access UI at http://localhost:8080"