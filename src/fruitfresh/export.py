"""Export a trained model to TensorFlow Lite (small + fast, for phones and edge devices)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def export_tflite(keras_path, tflite_path, quantization: str = "dynamic") -> Path:
    """quantization: 'none' | 'dynamic' (int8 weights) | 'float16'."""
    model = keras.models.load_model(keras_path)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif quantization != "none":
        raise ValueError("quantization must be 'none', 'dynamic' or 'float16'")
    path = Path(tflite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(conv.convert())
    return path


def tflite_predict(tflite_path, images) -> np.ndarray:
    """images: float32 (N, H, W, 3) in 0..255 -> probabilities (N, classes)."""
    it = tf.lite.Interpreter(model_path=str(tflite_path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    res = []
    for img in images:
        it.set_tensor(inp["index"], np.asarray(img, dtype="float32")[None])
        it.invoke()
        res.append(it.get_tensor(out["index"])[0])
    return np.stack(res)


def validate_tflite(keras_path, tflite_path, images) -> dict:
    """Check the TFLite model agrees with the Keras model on sample images."""
    model = keras.models.load_model(keras_path)
    images = np.asarray(images, dtype="float32")
    p_keras = model.predict(images, verbose=0)
    p_lite = tflite_predict(tflite_path, images)
    return {
        "n_images": int(len(images)),
        "max_abs_prob_diff": float(np.abs(p_keras - p_lite).max()),
        "top1_agreement": float((p_keras.argmax(1) == p_lite.argmax(1)).mean()),
        "keras_mb": round(Path(keras_path).stat().st_size / 1e6, 2),
        "tflite_mb": round(Path(tflite_path).stat().st_size / 1e6, 2),
    }
