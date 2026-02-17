# Created: 2026-02-17
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Overview

Flask-based web app for handwritten digit recognition. Users draw on an HTML canvas and get predictions from an SVM model trained on MNIST.

## Running

```bash
bash run.sh   # Creates venv, installs deps, starts server on http://localhost:5001
```

## Architecture

- **`app.py`** — Flask server with two routes:
  - `GET /` — Serves the drawing UI
  - `POST /predict` — Accepts base64 PNG, returns `{"digit": N}`
- **`templates/index.html`** — Single-page UI with embedded CSS/JS, 280x280 canvas (white-on-black), mouse + touch support

## Model

`SVC(kernel='rbf', C=5, gamma=0.05)` trained on 20k MNIST samples via `fetch_openml('mnist_784', parser='liac-arff')`.

## Image Preprocessing Pipeline

Canvas base64 PNG → grayscale (no inversion) → crop bounding box → square padding → resize 28x28 → center-of-mass alignment (`scipy.ndimage.shift`) → normalize [0,1] → 784-dim vector

## Dependencies

flask, scikit-learn, pillow, numpy, scipy
