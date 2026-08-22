from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "densenet121_tb_best.keras"

IMG_SIZE = 224

# Load model only once
@tf.keras.utils.register_keras_serializable()
def load_classifier():
    return tf.keras.models.load_model(str(MODEL_PATH))

model = load_classifier()


def predict_tb(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image).astype("float32")
    img = np.expand_dims(img, axis=0)

    probability = float(model.predict(img, verbose=0)[0][0])

    if probability >= 0.5:
        label = "Tuberculosis"
    else:
        label = "Normal"

    confidence = probability if label == "Tuberculosis" else 1 - probability

    return {
        "label": label,
        "confidence": confidence,
        "tb_probability": probability,
    }