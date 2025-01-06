#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p logs
chmod 777 uploads
chmod 777 logs
