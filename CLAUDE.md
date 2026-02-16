# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Handwritten digit recognition web app. A Flask server serves a single-page drawing canvas where users draw digits (0-9), which are classified by a scikit-learn SVM model trained on MNIST data.

## Architecture

Everything lives in a single file `digit_recognizer.py`:
- **Model layer**: SVM classifier (`SVC` with RBF kernel) trained on 20k MNIST samples. The trained model is serialized to `digit_model.pkl` via joblib. On first run (no pkl file), MNIST is downloaded and the model is trained (~1-2 min).
- **Web layer**: Flask app with two routes — `GET /` serves an inline HTML/JS/CSS page, `POST /predict` accepts a base64-encoded canvas image and returns the predicted digit with probability distribution.
- **Image preprocessing** (`/predict`): Canvas image → grayscale → crop to bounding box → resize to fit 20×20 area → center in 28×28 → flatten to 784-dim vector normalized to [0,1].

## Running the App

```bash
./run.sh          # or: python3 digit_recognizer.py
```

Server starts at http://localhost:5000.

## Dependencies

Python packages: `numpy`, `Pillow`, `flask`, `scikit-learn`, `joblib`
