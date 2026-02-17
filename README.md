# Created: 2026-02-17

# MNIST Handwritten Digit Recognizer

A web app where you draw digits (0-9) on a canvas and get real-time predictions from an SVM model trained on the MNIST dataset.

## Demo

1. Draw a digit on the black canvas
2. Click **Predict**
3. See the recognized digit

## Tech Stack

- **Backend**: Flask, scikit-learn (SVM with RBF kernel), scipy
- **Frontend**: HTML5 Canvas, vanilla JavaScript
- **Preprocessing**: PIL/Pillow, NumPy

## How It Works

1. SVM model trains on 20,000 MNIST samples at startup
2. User draws a digit on the 280x280 canvas (white on black)
3. The image is sent to the server as base64 PNG
4. Preprocessing: crop → square padding → resize to 28x28 → center-of-mass alignment
5. Model predicts the digit and returns the result

## Getting Started

```bash
# Run the app (creates venv and installs dependencies automatically)
bash run.sh
```

Open http://localhost:5001 in your browser.

## Project Structure

```
├── app.py                 # Flask server + SVM model + image preprocessing
├── templates/
│   └── index.html         # Drawing canvas UI
├── run.sh                 # Startup script
└── README.md
```
