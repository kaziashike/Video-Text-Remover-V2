#!/bin/bash

# Update script for Video Subtitle Remover on RunPod

echo "Updating Video Subtitle Remover code..."

# Pull the latest code from repository (if using git)
if command -v git &> /dev/null; then
    echo "Git detected. Pulling latest changes..."
    git pull
else
    echo "Git not available. Please update code manually or install git."
    echo "You can also rebuild and push the Docker image with new code."
fi

# Install/upgrade any new dependencies if requirements.txt was updated
if [ -f requirements.txt ]; then
    echo "Installing/upgrading dependencies..."
    pip install --no-cache-dir -r requirements.txt
fi

echo "Code update completed. Please restart the application for changes to take effect."