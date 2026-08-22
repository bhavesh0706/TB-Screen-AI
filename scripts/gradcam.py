from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "weights" / "densenet121_tb_best.keras"
TEST_DIR = ROOT / "data" / "test"
IMG_SIZE = 224

# Load trained model
model = tf.keras.models.load_model(str(MODEL_PATH))

# Extract backbone and classifier head
backbone = model.get_layer("densenet121")
gap = model.get_layer("global_average_pooling2d")
dropout = model.get_layer("dropout")
classifier = model.get_layer("dense")

print("Backbone:", backbone.name)

# Pick one TB image
img_path = sorted((TEST_DIR / "Tuberculosis").glob("*.png"))[0]

img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Same preprocessing used during training
img_tensor = tf.keras.applications.densenet.preprocess_input(img_array)

# Forward pass + Gradients
with tf.GradientTape() as tape:
    conv_output = backbone(img_tensor, training=False)
    tape.watch(conv_output)

    x = gap(conv_output)
    x = dropout(x, training=False)
    prediction = classifier(x)

    loss = prediction[:, 0]

grads = tape.gradient(loss, conv_output)

# Grad-CAM
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_output = conv_output[0]
heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0)
heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

heatmap = heatmap.numpy()

# Resize heatmap
heatmap = tf.image.resize(
    heatmap[..., np.newaxis],
    (IMG_SIZE, IMG_SIZE)
).numpy().squeeze()

# Plot
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img)
plt.imshow(heatmap, cmap="jet", alpha=0.45)
plt.axis("off")
plt.title("Grad-CAM")

plt.tight_layout()

out = ROOT / "results" / "gradcam_tb.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=300)
plt.close()

print("Saved:", out)