# Created: 2026-02-17
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Overview

Tkinter-based desktop app for handwritten digit recognition. Users draw on a canvas and get predictions from an SVM model trained on MNIST.

## Running

```bash
bash run.sh   # Creates venv, installs deps, launches Tkinter window
```

## Architecture

- **`app.py`** — Single file containing:
  - `train_model()` — SVM training on MNIST subset
  - `preprocess_image()` — PIL image → 28x28 numpy array
  - `DigitRecognizerApp` — Tkinter GUI class with canvas, buttons, result display

## Model

`SVC(kernel='rbf', C=5, gamma=0.05)` trained on 20k MNIST samples via `fetch_openml('mnist_784', parser='liac-arff')`.

## Image Preprocessing Pipeline

PIL canvas image → grayscale → crop bounding box → square padding → resize 28x28 → center-of-mass alignment (`scipy.ndimage.shift`) → normalize [0,1] → 784-dim vector

## GUI

- 280x280 black canvas with white brush (size 18)
- Mouse drawing via `<B1-Motion>` and `<Button-1>` events
- Dual drawing: Tkinter canvas (display) + PIL image (prediction input)

## Dependencies

scikit-learn, pillow, numpy, scipy (no flask needed)
