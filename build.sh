#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create credentials directory
mkdir -p /etc/google/auth/

# Write Google Cloud credentials from environment variable
echo "$GOOGLE_CREDENTIALS_JSON" > /etc/google/auth/credentials.json

# Set environment variable to point to the credentials file
export GOOGLE_APPLICATION_CREDENTIALS=/etc/google/auth/credentials.json
