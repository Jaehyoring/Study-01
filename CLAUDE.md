# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

Two versions of the MNIST handwritten digit recognizer:

- **`web_version/`** — Flask web app with HTML canvas UI (http://localhost:5001)
- **`desktop_version/`** — Tkinter desktop GUI app

Each version has its own `CLAUDE.md`, `app.py`, and `run.sh`.

## Running

```bash
bash web_version/run.sh       # Web version → http://localhost:5001
bash desktop_version/run.sh   # Desktop version → Tkinter window
```

First startup is slow (~2 min) due to MNIST dataset download and SVM training on 20k samples.

## Shared Model & Preprocessing

Both versions use the same SVM model and preprocessing pipeline:

- **Model:** `SVC(kernel='rbf', C=5, gamma=0.05)` on 20k MNIST samples
- **Data source:** `fetch_openml('mnist_784', parser='liac-arff')` — `parser='liac-arff'` avoids pandas dependency
- **Preprocessing:** crop → square padding → resize 28x28 → center-of-mass alignment (`scipy.ndimage.shift`) → normalize [0,1]

## Key Conventions

- All code and comments in English
- Web version uses port 5001 (not 5000, which conflicts with macOS AirPlay Receiver)
