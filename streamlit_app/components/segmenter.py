from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "unet_best.keras"

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

IMG_SIZE = 256

def segment_lungs(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)/255.0

    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    pred = model.predict(arr[np.newaxis,...], verbose=0)[0,...,0]

    mask = (pred>0.5).astype(np.uint8)*255

    return Image.fromarray(mask)