from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "densenet121_tb_best.keras"

IMG_SIZE = (224, 224)

# ----------------------------------------------------
# Load model once
# ----------------------------------------------------

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

# Force build
_ = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)

backbone = model.get_layer("densenet121")
gap = model.layers[2]
dropout = model.layers[3]
classifier = model.layers[4]

# ----------------------------------------------------
# Rebuild graph correctly (Keras 3 fix)
# ----------------------------------------------------

inputs = model.input

features = backbone(inputs, training=False)
x = gap(features)
x = dropout(x, training=False)
outputs = classifier(x)

grad_model = tf.keras.Model(
    inputs=inputs,
    outputs=[features, outputs],
)

# ----------------------------------------------------
# Lung mask helper
# ----------------------------------------------------

def _prepare_mask(mask, w, h):

    if mask is None:
        return None

    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
    else:
        mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return (mask > 127).astype(np.float32)

# ----------------------------------------------------
# Grad-CAM
# ----------------------------------------------------

def generate_gradcam(image, lung_mask=None):

    rgb = image.convert("RGB")
    original = np.array(rgb)

    h, w = original.shape[:2]

    x = rgb.resize(IMG_SIZE)
    x = np.array(x, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    x = tf.convert_to_tensor(x)

    with tf.GradientTape() as tape:

        conv_output, preds = grad_model(x, training=False)

        score = preds[:, 0]

    grads = tape.gradient(score, conv_output)

    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None.")

    weights = tf.reduce_mean(grads, axis=(1, 2))

    cam = tf.reduce_sum(
        conv_output * weights[:, None, None, :],
        axis=-1,
    )[0]

    cam = tf.maximum(cam, 0)

    cam = cam.numpy().astype(np.float32)

    if cam.max() > 0:
        cam /= cam.max()

    cam = np.squeeze(cam)

    cam = cv2.resize(
        cam,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    mask = _prepare_mask(lung_mask, w, h)

    if mask is not None:

        cam *= mask

        if cam.max() > 0:
            cam /= cam.max()

    heat = np.uint8(np.clip(cam * 255, 0, 255))

    heat = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET,
    )

    heat = cv2.cvtColor(
        heat,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        heat,
        0.40,
        0,
    )

    return Image.fromarray(overlay)