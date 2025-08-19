#!/bin/bash

set -e  # Stop immediately on any error

echo "Starting full pipeline: "

# 1. Build images
./scripts/build_all.sh

# 2. Deploy to Kubernetes
./scripts/deploy_all.sh

# 3. Port-forward UI (optional step — blocks terminal)
echo "Starting port-forward to access UI..."
./scripts/start_ui.sh
