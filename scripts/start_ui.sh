#!/bin/bash

echo "Forwarding UI port..."

kubectl port-forward svc/ui-service 8080:80 -n zebraflights