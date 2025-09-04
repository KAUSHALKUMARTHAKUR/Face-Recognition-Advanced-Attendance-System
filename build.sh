#!/bin/bash
set -e

echo "Installing system dependencies..."
apt-get update >/dev/null 2>&1
apt-get install -y build-essential cmake libopenblas-dev libatlas-base-dev >/dev/null 2>&1

echo "Installing Python build tools..."
pip install --upgrade pip cmake wheel

echo "Installing dlib specifically..."
pip install --no-cache-dir --timeout=600 dlib==19.24.2

echo "Installing remaining packages..."
pip install --no-cache-dir -r requirements.txt

echo "Build completed!"
