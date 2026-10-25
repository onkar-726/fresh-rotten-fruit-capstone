"""Grad-CAM: which image regions drove the prediction?

Works for both models because they share ``input -> [rescale] -> backbone -> gap -> dropout -> probs``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CLASS_NAMES


def gradcam_heatmap(model, image, class_index: int | None = None):
    """image: float array (H, W, 3) in 0..255. Returns (heatmap 0..1, class_index, probabilities)."""
    names = [l.name for l in model.layers]
    bb = names.index("backbone")
    pre, post, backbone = model.layers[1:bb], model.layers[bb + 1 :], model.layers[bb]
    x = tf.convert_to_tensor(np.asarray(image, dtype="float32")[None])
    for layer in pre:
        x = layer(x)
    with tf.GradientTape() as tape:
        fmap = backbone(x, training=False)
        tape.watch(fmap)
        y = fmap
        for layer in post:
            y = layer(y, training=False)
        if class_index is None:
            class_index = int(tf.argmax(y[0]))
        score = y[:, class_index]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8), class_index, y[0].numpy()


def overlay_heatmap(image_uint8, heatmap, alpha: float = 0.45):
    """Blend the colour-mapped heat-map over the image. Returns uint8 RGB."""
    h, w = image_uint8.shape[:2]
    cam = np.array(Image.fromarray((heatmap * 255).astype("uint8")).resize((w, h), Image.BILINEAR)) / 255.0
    colour = (plt.get_cmap("jet")(cam)[..., :3] * 255).astype("uint8")
    return ((1 - alpha) * image_uint8 + alpha * colour).astype("uint8")


def plot_gradcam_grid(model, images, true_labels, cols: int = 4):
    """images: list of uint8 RGB arrays at the model input size."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(3 * cols, 3 * rows * 2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (img, true) in enumerate(zip(images, true_labels)):
        r, c = divmod(k, cols)
        cam, pred, probs = gradcam_heatmap(model, img)
        axes[2 * r, c].imshow(img)
        axes[2 * r, c].set_title(f"true: {CLASS_NAMES[true]}", fontsize=8)
        axes[2 * r + 1, c].imshow(overlay_heatmap(img, cam))
        axes[2 * r + 1, c].set_title(f"pred: {CLASS_NAMES[pred]} ({probs[pred]:.2f})", fontsize=8,
                                     color="green" if pred == true else "red")
    fig.tight_layout()
    return fig
