# Created: 2026-02-17
# MNIST Handwritten Digit Recognition Desktop App
# Tkinter GUI that trains an SVM model on MNIST data and predicts drawn digits

import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift


# --- Model ---

def train_model():
    """Train an SVM classifier on a subset of MNIST data."""
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

    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Training accuracy: {train_acc:.4f}")
    print("Model ready!")
    return model


# --- Image Preprocessing ---

def preprocess_image(pil_image):
    """Convert PIL image to 28x28 grayscale numpy array matching MNIST format."""
    # Convert to grayscale
    img = pil_image.convert('L')

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


# --- GUI ---

class DigitRecognizerApp:
    CANVAS_SIZE = 280
    BRUSH_SIZE = 18

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Digit Recognizer")
        self.root.resizable(False, False)
        self.root.configure(bg='#1a1a2e')

        # PIL image to track drawing (canvas doesn't export pixels directly)
        self.pil_image = Image.new('L', (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._build_ui()

    def _build_ui(self):
        # Title
        title = tk.Label(
            self.root, text="Digit Recognizer",
            font=("Helvetica", 24, "bold"), fg="#8b7ecf", bg='#1a1a2e'
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            self.root, text="Draw a digit (0-9) and click Predict",
            font=("Helvetica", 12), fg="#888", bg='#1a1a2e'
        )
        subtitle.pack(pady=(0, 15))

        # Drawing canvas
        self.canvas = tk.Canvas(
            self.root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg='black', cursor='crosshair', highlightthickness=2,
            highlightbackground='#333'
        )
        self.canvas.pack()

        # Mouse events
        self.canvas.bind('<B1-Motion>', self._draw)
        self.canvas.bind('<Button-1>', self._draw)

        # Buttons
        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(pady=15)

        predict_btn = tk.Button(
            btn_frame, text="Predict", command=self._predict,
            font=("Helvetica", 14, "bold"), fg="white", bg="#667eea",
            activebackground="#764ba2", activeforeground="white",
            padx=20, pady=8, relief=tk.FLAT
        )
        predict_btn.pack(side=tk.LEFT, padx=10)

        clear_btn = tk.Button(
            btn_frame, text="Clear", command=self._clear,
            font=("Helvetica", 14), fg="#ccc", bg="#333",
            activebackground="#444", activeforeground="#ccc",
            padx=20, pady=8, relief=tk.FLAT
        )
        clear_btn.pack(side=tk.LEFT, padx=10)

        # Result label
        self.result_label = tk.Label(
            self.root, text="Draw a digit above",
            font=("Helvetica", 16), fg="#555", bg='#1a1a2e'
        )
        self.result_label.pack(pady=(10, 5))

        self.digit_label = tk.Label(
            self.root, text="",
            font=("Helvetica", 72, "bold"), fg="#8b7ecf", bg='#1a1a2e'
        )
        self.digit_label.pack(pady=(0, 20))

    def _draw(self, event):
        r = self.BRUSH_SIZE // 2
        x, y = event.x, event.y
        # Draw on tkinter canvas
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill='white', outline='white'
        )
        # Draw on PIL image (for prediction)
        self.pil_draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=255
        )

    def _clear(self):
        self.canvas.delete('all')
        self.pil_image = Image.new('L', (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text="Draw a digit above", fg="#555")
        self.digit_label.config(text="")

    def _predict(self):
        pixel_array = preprocess_image(self.pil_image)
        if pixel_array is None:
            self.result_label.config(text="Empty canvas!", fg="#e74c3c")
            self.digit_label.config(text="")
            return

        prediction = self.model.predict(pixel_array)[0]
        self.result_label.config(text="Predicted Digit", fg="#888")
        self.digit_label.config(text=str(prediction))


if __name__ == '__main__':
    model = train_model()
    root = tk.Tk()
    app = DigitRecognizerApp(root, model)
    root.mainloop()
