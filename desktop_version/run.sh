#!/bin/bash
# Created: 2026-02-17
# Run the MNIST Digit Recognizer desktop app

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies (no flask needed for desktop version)
pip install -q scikit-learn pillow numpy scipy

echo "Starting Digit Recognizer (Desktop)..."
python app.py
