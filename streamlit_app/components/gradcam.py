from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "weights" / "densenet121_tb_best.keras"
IMG_SIZE = 224

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

# DenseNet backbone
backbone = model.get_layer("densenet121")

LAST_CONV = "conv5_block16_concat"

# Backbone model (input -> last conv + backbone output)
backbone_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        backbone.get_layer(LAST_CONV).output,
        backbone.output,
    ],
)

# Classifier head (everything after backbone)
head_layers = model.layers[model.layers.index(backbone) + 1 :]


def generate_gradcam(image):

    original = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    img = np.asarray(original, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:

        last_conv, x = backbone_model(img)

        for layer in head_layers:
            x = layer(x)

        preds = x

        class_idx = tf.argmax(preds[0])

        loss = preds[:, class_idx]

    grads = tape.gradient(loss, last_conv)

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv = last_conv[0]

    heatmap = tf.reduce_sum(last_conv * pooled, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = heatmap.numpy()

    # Resize heatmap
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (IMG_SIZE, IMG_SIZE)
    ).numpy()

    heatmap = np.squeeze(heatmap)

    # Apply JET colormap
    colored = cm.jet(heatmap)[..., :3]
    colored = (colored * 255).astype(np.uint8)

    colored = Image.fromarray(colored)

    overlay = Image.blend(original, colored, alpha=0.45)

    return overlay