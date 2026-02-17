# Created: 2026-02-17

# MNIST Handwritten Digit Recognizer

Draw digits (0-9) and get real-time predictions from an SVM model trained on the MNIST dataset. Available as both a **web app** and a **desktop app**.

## Demo

1. Draw a digit on the black canvas
2. Click **Predict**
3. See the recognized digit

## Versions

### Web Version
Flask web app with HTML5 Canvas UI.

```bash
bash web_version/run.sh
```
Open http://localhost:5001 in your browser.

### Desktop Version
Tkinter desktop GUI app.

```bash
bash desktop_version/run.sh
```

## Tech Stack

- **Model**: scikit-learn SVM (RBF kernel), trained on 20k MNIST samples
- **Web**: Flask, HTML5 Canvas, vanilla JavaScript
- **Desktop**: Tkinter, PIL/Pillow
- **Preprocessing**: PIL/Pillow, NumPy, SciPy

## Project Structure

```
├── web_version/
│   ├── app.py             # Flask server + SVM model + preprocessing
│   ├── templates/
│   │   └── index.html     # Drawing canvas UI
│   ├── run.sh             # Startup script
│   └── CLAUDE.md
├── desktop_version/
│   ├── app.py             # Tkinter GUI + SVM model + preprocessing
│   ├── run.sh             # Startup script
│   └── CLAUDE.md
├── CLAUDE.md
└── README.md
```
