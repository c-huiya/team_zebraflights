#!/bin/bash

echo "Forwarding UI port..."

kubectl port-forward -n zebraflights service/ui-service 5002:5002
