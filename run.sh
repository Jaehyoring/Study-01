#!/bin/bash
# Created: 2026-02-17
# Run the MNIST Digit Recognizer web app

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
pip install -q flask scikit-learn pillow numpy scipy

echo "Starting Digit Recognizer..."
echo "Open http://localhost:5001 in your browser"
python app.py
