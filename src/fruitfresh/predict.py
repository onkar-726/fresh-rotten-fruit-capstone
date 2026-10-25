"""Inference pipeline: PIL image -> predicted class, fresh/rotten status, confidence.

Used by the Gradio app, scripts/predict.py and the notebooks. Works with the trained Keras model
(.keras) or an exported TensorFlow Lite model (.tflite). Pre-processing is identical to training:
EXIF rotation fixed, RGB, bilinear resize, rounded to uint8, then float 0..255 (the model rescales itself).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from .config import CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE

CONFIDENCE_THRESHOLD = 0.60   # below this the app shows an "uncertain" warning
_FRUIT = {"apples": "apple", "banana": "banana", "oranges": "orange"}


def split_label(class_name: str) -> tuple[str, str]:
    """'rottenapples' -> ('apple', 'rotten')"""
    status = "fresh" if class_name.startswith("fresh") else "rotten"
    return _FRUIT[class_name[len(status):]], status


def preprocess_pil(image: Image.Image, size=IMAGE_SIZE) -> np.ndarray:
    """-> float32 array (1, H, W, 3) with values 0..255, exactly like the training pipeline."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    arr = tf.image.resize(np.array(image), size)
    return tf.cast(tf.round(arr), tf.uint8).numpy()[None].astype("float32")


class FruitPredictor:
    def __init__(self, model_path, threshold: float = CONFIDENCE_THRESHOLD):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.threshold = threshold
        self.is_tflite = self.model_path.suffix == ".tflite"
        if self.is_tflite:
            self.interp = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interp.allocate_tensors()
            det = self.interp.get_input_details()[0]
            self._in, self._out = det["index"], self.interp.get_output_details()[0]["index"]
            self.input_size = tuple(int(v) for v in det["shape"][1:3])
            self.model = None
        else:
            self.model = tf.keras.models.load_model(self.model_path)
            self.input_size = tuple(int(v) for v in self.model.input_shape[1:3])

    def probabilities(self, image: Image.Image) -> np.ndarray:
        x = preprocess_pil(image, self.input_size)
        if self.is_tflite:
            self.interp.set_tensor(self._in, x)
            self.interp.invoke()
            return self.interp.get_tensor(self._out)[0]
        return self.model.predict(x, verbose=0)[0]

    def predict(self, image: Image.Image) -> dict:
        probs = self.probabilities(image)
        top = int(probs.argmax())
        cls = CLASS_NAMES[top]
        fruit, status = split_label(cls)
        p_fresh = float(sum(p for c, p in zip(CLASS_NAMES, probs) if c.startswith("fresh")))
        order = np.argsort(probs)[::-1][:3]
        return {
            "class_name": cls,
            "label": DISPLAY_NAMES[cls],
            "fruit": fruit,
            "status": status,                                   # "fresh" | "rotten"
            "confidence": float(probs[top]),                    # confidence of the predicted class
            "status_confidence": p_fresh if status == "fresh" else 1.0 - p_fresh,
            "uncertain": bool(probs[top] < self.threshold),
            "top3": [(DISPLAY_NAMES[CLASS_NAMES[i]], float(probs[i])) for i in order],
            "probabilities": {DISPLAY_NAMES[c]: float(p) for c, p in zip(CLASS_NAMES, probs)},
        }
