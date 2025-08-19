#!/bin/bash

set -e  # Stop immediately on any error

echo "Starting full pipeline: build → deploy → port-forward"

# 1. Build images
./build_all.sh

# 2. Deploy to Kubernetes
./deploy_all.sh

# 3. Port-forward UI (optional step — blocks terminal)
echo "tarting port-forward to access UI..."
./start_ui.sh
