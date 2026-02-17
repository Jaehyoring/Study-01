# Created: 2026-02-17
# MNIST Handwritten Digit Recognition Web App
# Flask server that trains an SVM model on MNIST data and serves predictions

import base64
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift
from PIL import Image, ImageOps

app = Flask(__name__)
model = None


def train_model():
    """Train an SVM classifier on a subset of MNIST data."""
    global model
    print("Fetching MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data, mnist.target.astype(int)

    # Subsample for faster training (20k samples for good accuracy)
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=20000, stratify=y, random_state=42
    )

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0

    print(f"Training SVM on {len(X_train)} samples...")
    model = SVC(kernel='rbf', C=5, gamma=0.05)
    model.fit(X_train, y_train)

    # Quick accuracy check on training data
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Training accuracy: {train_acc:.4f}")
    print("Model ready!")


def preprocess_image(image_data_url):
    """Convert base64 canvas image to 28x28 grayscale numpy array matching MNIST format."""
    # Remove data URL prefix
    header, encoded = image_data_url.split(',', 1)
    image_bytes = base64.b64decode(encoded)

    # Open image and convert to grayscale
    # Canvas: white (255) drawing on black (0) background — same as MNIST, no inversion needed
    img = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Find bounding box of the drawn content and crop
    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    # Make it square by padding the shorter side
    width, height = img.size
    max_dim = max(width, height)
    square = Image.new('L', (max_dim, max_dim), 0)
    offset_x = (max_dim - width) // 2
    offset_y = (max_dim - height) // 2
    square.paste(img, (offset_x, offset_y))

    # Add padding around the digit (~20% on each side, similar to MNIST)
    pad = int(max_dim * 0.3)
    padded = Image.new('L', (max_dim + 2 * pad, max_dim + 2 * pad), 0)
    padded.paste(square, (pad, pad))

    # Resize to 28x28 using antialiasing
    img_resized = padded.resize((28, 28), Image.LANCZOS)

    # Center the digit by center of mass (matching MNIST preprocessing)
    pixel_array = np.array(img_resized, dtype=np.float64)
    total = pixel_array.sum()
    if total > 0:
        cy, cx = np.indices(pixel_array.shape)
        center_x = int(np.round(np.sum(cx * pixel_array) / total))
        center_y = int(np.round(np.sum(cy * pixel_array) / total))
        shift_x = 14 - center_x
        shift_y = 14 - center_y
        pixel_array = shift(pixel_array, [shift_y, shift_x], mode='constant', cval=0)

    # Normalize to [0, 1]
    pixel_array = pixel_array.reshape(1, 784) / 255.0
    return pixel_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    pixel_array = preprocess_image(data['image'])
    if pixel_array is None:
        return jsonify({'error': 'Empty canvas - please draw a digit'}), 400

    prediction = model.predict(pixel_array)[0]
    return jsonify({'digit': int(prediction)})


if __name__ == '__main__':
    train_model()
    app.run(debug=False, host='0.0.0.0', port=5001)
