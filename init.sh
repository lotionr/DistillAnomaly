#!/bin/bash
# init.sh — Environment setup for anomaly detection pipeline
# Run this once at the start of every new session or fresh instance

set -e

echo "=== Setting up anomaly detection environment ==="

# 1. System dependencies
echo "[1/6] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y python3-pip python3-venv git unzip wget

# 2. Python virtual environment
echo "[2/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Python packages
echo "[3/6] Installing Python packages..."
pip install --upgrade pip
pip install \
    numpy pandas scikit-learn scipy \
    pyod \
    requests tqdm \
    matplotlib seaborn \
    jupyter notebook

# 4. ADBench datasets
echo "[4/6] Downloading ADBench datasets..."
mkdir -p data/adbench
if [ ! "$(ls -A data/adbench)" ]; then
    wget -q https://github.com/Minqi824/ADBench/archive/refs/heads/main.zip -O adbench.zip
    unzip -q adbench.zip "ADBench-main/adbench/datasets/*" -d /tmp/adbench_extract
    cp /tmp/adbench_extract/ADBench-main/adbench/datasets/*.npz data/adbench/ 2>/dev/null || true
    rm -f adbench.zip
    echo "ADBench datasets downloaded: $(ls data/adbench/*.npz 2>/dev/null | wc -l) files"
else
    echo "ADBench datasets already present: $(ls data/adbench/*.npz | wc -l) files"
fi

# 5. Verify codebase
echo "[5/6] Verifying codebase structure..."
for f in feature_list.json claude-progress.txt; do
    if [ -f "$f" ]; then
        echo "  ✓ $f exists"
    else
        echo "  ✗ $f missing — may need to be created"
    fi
done

# 6. Quick smoke test
echo "[6/6] Running smoke test..."
python3 -c "
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
print('  ✓ Core imports OK')
data = np.load('data/adbench/' + sorted(__import__('os').listdir('data/adbench'))[0])
print(f'  ✓ Sample dataset loaded: X={data[\"X\"].shape}, y={data[\"y\"].shape}')
" && echo "Smoke test passed." || echo "Smoke test failed — check data/adbench/ directory."

echo ""
echo "=== Setup complete. Activate with: source venv/bin/activate ==="
