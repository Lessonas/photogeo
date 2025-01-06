#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create uploads directory
mkdir -p uploads
chmod 777 uploads
