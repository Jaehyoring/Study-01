# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
bash run.sh        # Creates venv, installs deps, starts server
# or manually:
source venv/bin/activate
python app.py      # Serves on http://localhost:5001
```

First startup is slow (~2 min) due to MNIST dataset download and SVM training on 20k samples.

## Dependencies

flask, scikit-learn, pillow, numpy, scipy (installed via `pip install` in run.sh)

## Architecture

Single-file Flask app (`app.py`) with an inline HTML/JS/CSS frontend (`templates/index.html`).

**Data flow:** Canvas (280x280, white-on-black) → base64 PNG → `POST /predict` → `preprocess_image()` → SVM prediction → JSON response

**Image preprocessing pipeline** (`preprocess_image` in app.py):
1. Decode base64 → grayscale (no color inversion — canvas matches MNIST format)
2. Crop to bounding box → square padding → resize to 28x28
3. Center-of-mass alignment via `scipy.ndimage.shift` (critical for accuracy)
4. Normalize to [0, 1], flatten to 784-dim vector

**Model:** `SVC(kernel='rbf', C=5, gamma=0.05)` trained on 20k MNIST samples via `fetch_openml('mnist_784', parser='liac-arff')`. The `parser='liac-arff'` flag is required to avoid a pandas dependency.

## Key Conventions

- All code and comments in English
- Port 5001 (not 5000, which conflicts with macOS AirPlay Receiver)
