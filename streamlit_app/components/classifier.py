from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "weights" / "densenet121_tb_best.keras"
THRESHOLD_PATH = ROOT / "evaluation" / "threshold.json"

IMG_SIZE = 224

# Load model once
model = tf.keras.models.load_model(str(MODEL_PATH))

# Load validated threshold
THRESHOLD = 0.5
if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH) as f:
        THRESHOLD = json.load(f)["threshold"]


def predict_tb(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    probability = float(model.predict(img, verbose=0)[0][0])

    label = "Tuberculosis" if probability >= THRESHOLD else "Normal"
    confidence = probability if label == "Tuberculosis" else 1 - probability

    return {
        "label": label,
        "confidence": confidence,
        "tb_probability": probability,
        "threshold": THRESHOLD,
    }