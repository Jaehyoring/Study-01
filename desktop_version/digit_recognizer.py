# Created: 2026-02-16 22:30
import os
import gzip
import struct
import urllib.request
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
from sklearn.svm import SVC
import joblib

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

    train_size = 20000
    X_train = X[:train_size] / 255.0
    y_train = y[:train_size]

    print(f"Training SVM on {train_size} samples (this takes ~1-2 min)...")
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model.fit(X_train, y_train)

    X_test = X[60000:61000] / 255.0
    y_test = y[60000:61000]
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.2%}")

    joblib.dump(model, MODEL_PATH)
    print("Model saved.")
    return model


class DigitRecognizerApp:
    CANVAS_SIZE = 280
    LINE_WIDTH = 18

    def __init__(self, root, model):
        self.model = model
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # Title
        title = tk.Label(
            root, text="Digit Recognizer",
            font=("Helvetica", 22, "bold"), fg="#667eea", bg="#1a1a2e",
        )
        title.pack(pady=(16, 2))

        subtitle = tk.Label(
            root, text="Draw a digit (0-9) and click Predict",
            font=("Helvetica", 11), fg="#888888", bg="#1a1a2e",
        )
        subtitle.pack(pady=(0, 12))

        # Main frame
        main_frame = tk.Frame(root, bg="#1a1a2e")
        main_frame.pack(padx=20, pady=(0, 16))

        # Canvas
        self.canvas = tk.Canvas(
            main_frame, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="black", cursor="crosshair", highlightthickness=2,
            highlightbackground="#444444",
        )
        self.canvas.grid(row=0, column=0, padx=(0, 20))

        # PIL image for accurate pixel capture
        self.pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        # Mouse bindings
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.last_x = None
        self.last_y = None

        # Result panel
        result_frame = tk.Frame(main_frame, bg="#1a1a2e", width=220)
        result_frame.grid(row=0, column=1, sticky="n")
        result_frame.grid_propagate(False)
        result_frame.configure(height=self.CANVAS_SIZE)

        self.digit_label = tk.Label(
            result_frame, text="?", font=("Helvetica", 96, "bold"),
            fg="#667eea", bg="#1a1a2e",
        )
        self.digit_label.pack(pady=(10, 0))

        self.confidence_label = tk.Label(
            result_frame, text="", font=("Helvetica", 12), fg="#aaaaaa", bg="#1a1a2e",
        )
        self.confidence_label.pack()

        # Probability bars
        self.bar_frame = tk.Frame(result_frame, bg="#1a1a2e")
        self.bar_frame.pack(fill="x", padx=8, pady=(8, 0))

        self.bar_canvases = []
        self.bar_labels = []
        for i in range(10):
            row = tk.Frame(self.bar_frame, bg="#1a1a2e")
            row.pack(fill="x", pady=1)
            lbl = tk.Label(row, text=str(i), width=2, anchor="e",
                           font=("Helvetica", 9), fg="#aaaaaa", bg="#1a1a2e")
            lbl.pack(side="left")
            bar_cv = tk.Canvas(row, height=12, bg="#222222", highlightthickness=0)
            bar_cv.pack(side="left", fill="x", expand=True, padx=(4, 4))
            val_lbl = tk.Label(row, text="0.0%", width=6, anchor="e",
                               font=("Helvetica", 8), fg="#888888", bg="#1a1a2e")
            val_lbl.pack(side="left")
            self.bar_canvases.append(bar_cv)
            self.bar_labels.append(val_lbl)

        # Buttons
        btn_frame = tk.Frame(root, bg="#1a1a2e")
        btn_frame.pack(pady=(0, 16))

        predict_btn = tk.Button(
            btn_frame, text="Predict", font=("Helvetica", 13, "bold"),
            fg="white", bg="#667eea", activebackground="#764ba2",
            activeforeground="white", relief="flat", padx=24, pady=6,
            command=self.predict,
        )
        predict_btn.pack(side="left", padx=8)

        clear_btn = tk.Button(
            btn_frame, text="Clear", font=("Helvetica", 13),
            fg="#cccccc", bg="#333333", activebackground="#444444",
            activeforeground="#cccccc", relief="flat", padx=24, pady=6,
            command=self.clear_canvas,
        )
        clear_btn.pack(side="left", padx=8)

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.last_x is not None:
            r = self.LINE_WIDTH // 2
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="white", width=self.LINE_WIDTH, capstyle="round", joinstyle="round",
            )
            self.pil_draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=self.LINE_WIDTH,
            )
            # Draw round caps on PIL image
            self.pil_draw.ellipse(
                [event.x - r, event.y - r, event.x + r, event.y + r], fill=255,
            )
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, _event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.digit_label.config(text="?")
        self.confidence_label.config(text="")
        for bc in self.bar_canvases:
            bc.delete("all")
        for bl in self.bar_labels:
            bl.config(text="0.0%")

    def predict(self):
        img = self.pil_image

        bbox = img.getbbox()
        if bbox is None:
            return

        img_cropped = img.crop(bbox)

        w, h = img_cropped.size
        scale = 20.0 / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_resized = img_cropped.resize((new_w, new_h), Image.LANCZOS)

        img_28 = Image.new("L", (28, 28), 0)
        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2
        img_28.paste(img_resized, (offset_x, offset_y))

        pixels = np.array(img_28, dtype=np.float64).reshape(1, 784) / 255.0

        digit = int(self.model.predict(pixels)[0])
        probs = self.model.predict_proba(pixels)[0]

        self.digit_label.config(text=str(digit))
        self.confidence_label.config(text=f"{probs[digit] * 100:.1f}% confidence")

        # Update bars
        for i in range(10):
            pct = probs[i] * 100
            bc = self.bar_canvases[i]
            bc.delete("all")
            bc.update_idletasks()
            bar_width = bc.winfo_width()
            fill_width = int(bar_width * probs[i])
            if fill_width > 0:
                bc.create_rectangle(0, 0, fill_width, 12, fill="#667eea", outline="")
            self.bar_labels[i].config(text=f"{pct:.1f}%")


def main():
    model = load_or_train_model()
    root = tk.Tk()
    DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
