#!/usr/bin/env bash

# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove all .pyc files
find . -type f -name "*.pyc" -delete

echo "Python caches cleaned."
