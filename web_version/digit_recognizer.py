# Created: 2026-02-16 22:30
import os
import base64
import io
import gzip
import struct
import urllib.request

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Model file is stored in the project root (one level up)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "digit_model.pkl")


def _download_mnist():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    data = {}
    for key, fname in files.items():
        url = base_url + fname
        print(f"  Fetching {fname}...")
        resp = urllib.request.urlopen(url)
        raw = gzip.decompress(resp.read())
        if "images" in key:
            _, n, rows, cols = struct.unpack(">IIII", raw[:16])
            data[key] = np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, rows * cols)
        else:
            _ = struct.unpack(">II", raw[:8])
            data[key] = np.frombuffer(raw[8:], dtype=np.uint8)
    X = np.vstack([data["train_images"], data["test_images"]])
    y = np.concatenate([data["train_labels"], data["test_labels"]])
    return X, y


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        return joblib.load(MODEL_PATH)

    print("Downloading MNIST dataset...")
    X, y = _download_mnist()

    # Use a subset for faster training (still gives ~97% accuracy)
    train_size = 20000
    X_train = X[:train_size] / 255.0
    y_train = y[:train_size]

    print(f"Training SVM on {train_size} samples (this takes ~1-2 min)...")
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model.fit(X_train, y_train)

    # Quick accuracy check on a small test set
    X_test = X[60000:61000] / 255.0
    y_test = y[60000:61000]
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.2%}")

    joblib.dump(model, MODEL_PATH)
    print("Model saved.")
    return model


model = load_or_train_model()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Handwritten Digit Recognizer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #eee;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  h1 {
    font-size: 1.8rem;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 0.9rem; }
  .container {
    display: flex;
    gap: 32px;
    align-items: flex-start;
    flex-wrap: wrap;
    justify-content: center;
  }
  .canvas-section { text-align: center; }
  canvas {
    border: 2px solid #444;
    border-radius: 12px;
    cursor: crosshair;
    background: #000;
    display: block;
    margin-bottom: 16px;
    touch-action: none;
  }
  .buttons { display: flex; gap: 12px; justify-content: center; }
  button {
    padding: 10px 28px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.1s, opacity 0.2s;
  }
  button:active { transform: scale(0.96); }
  .btn-predict {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
  }
  .btn-clear { background: #333; color: #ccc; }
  .btn-clear:hover { background: #444; }
  .result-section {
    text-align: center;
    min-width: 200px;
  }
  .predicted-digit {
    font-size: 7rem;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0;
    color: #667eea;
  }
  .confidence { font-size: 1.1rem; color: #aaa; }
  .bar-chart { margin-top: 16px; text-align: left; }
  .bar-row {
    display: flex;
    align-items: center;
    margin: 3px 0;
    font-size: 0.85rem;
  }
  .bar-label { width: 20px; text-align: right; margin-right: 8px; color: #aaa; }
  .bar-bg {
    flex: 1;
    height: 16px;
    background: #222;
    border-radius: 4px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.3s;
  }
  .bar-val { width: 44px; text-align: right; margin-left: 6px; color: #888; font-size: 0.8rem; }
  .placeholder { color: #555; margin-top: 30px; }
</style>
</head>
<body>
  <h1>Digit Recognizer</h1>
  <p class="subtitle">Draw a digit (0-9) and click Predict</p>
  <div class="container">
    <div class="canvas-section">
      <canvas id="canvas" width="280" height="280"></canvas>
      <div class="buttons">
        <button class="btn-predict" onclick="predict()">Predict</button>
        <button class="btn-clear" onclick="clearCanvas()">Clear</button>
      </div>
    </div>
    <div class="result-section" id="resultSection">
      <p class="placeholder">Draw a digit and<br>click Predict</p>
    </div>
  </div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

ctx.fillStyle = '#000';
ctx.fillRect(0, 0, 280, 280);
ctx.strokeStyle = '#fff';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
  const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
  return [x, y];
}

canvas.addEventListener('mousedown', e => { drawing = true; ctx.beginPath(); ctx.moveTo(...getPos(e)); });
canvas.addEventListener('mousemove', e => { if (!drawing) return; ctx.lineTo(...getPos(e)); ctx.stroke(); });
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseleave', () => drawing = false);

canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing = true; ctx.beginPath(); ctx.moveTo(...getPos(e)); });
canvas.addEventListener('touchmove', e => { e.preventDefault(); if (!drawing) return; ctx.lineTo(...getPos(e)); ctx.stroke(); });
canvas.addEventListener('touchend', () => drawing = false);

function clearCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 280, 280);
  document.getElementById('resultSection').innerHTML = '<p class="placeholder">Draw a digit and<br>click Predict</p>';
}

async function predict() {
  const dataURL = canvas.toDataURL('image/png');
  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image: dataURL})
  });
  const data = await res.json();
  const digit = data.digit;
  const probs = data.probabilities;
  const conf = (probs[digit] * 100).toFixed(1);

  let html = `<div class="predicted-digit">${digit}</div>`;
  html += `<div class="confidence">${conf}% confidence</div>`;
  html += '<div class="bar-chart">';
  for (let i = 0; i < 10; i++) {
    const pct = (probs[i] * 100).toFixed(1);
    html += `<div class="bar-row">
      <span class="bar-label">${i}</span>
      <div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div>
      <span class="bar-val">${pct}%</span>
    </div>`;
  }
  html += '</div>';
  document.getElementById('resultSection').innerHTML = html;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Crop to bounding box of the drawing, then center in 20x20 area within 28x28
    bbox = img.getbbox()
    if bbox is None:
        return jsonify(digit=0, probabilities=[0.1] * 10)

    img_cropped = img.crop(bbox)

    # Resize longest side to 20px, preserving aspect ratio
    w, h = img_cropped.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img_cropped.resize((new_w, new_h), Image.LANCZOS)

    # Center in 28x28 canvas
    img_28 = Image.new("L", (28, 28), 0)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    img_28.paste(img_resized, (offset_x, offset_y))

    pixels = np.array(img_28, dtype=np.float64).reshape(1, 784) / 255.0

    digit = int(model.predict(pixels)[0])
    probs = model.predict_proba(pixels)[0]
    prob_dict = {int(i): float(p) for i, p in enumerate(probs)}

    return jsonify(digit=digit, probabilities=prob_dict)


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(debug=False, port=5000)
