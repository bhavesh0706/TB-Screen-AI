from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs" / "unet.yaml") as f:
    cfg = yaml.safe_load(f)

IMAGE_DIR = ROOT / "data" / "raw" / "LungSegmentation704" / "image"
MASK_DIR = ROOT / "data" / "raw" / "LungSegmentation704" / "mask"

WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = cfg["image_size"]
BATCH_SIZE = cfg["batch_size"]
EPOCHS = cfg["epochs"]

# -----------------------------
# Collect files
# -----------------------------

image_files = sorted(IMAGE_DIR.glob("*"))
mask_files = sorted(MASK_DIR.glob("*"))

assert len(image_files) == len(mask_files), "Image and mask count mismatch."

print(f"Found {len(image_files)} image-mask pairs.")

# -----------------------------
# Train / Validation split
# -----------------------------

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_files,
    mask_files,
    test_size=cfg["val_split"],
    random_state=cfg["seed"],
    shuffle=True,
)

# Convert Path objects to strings for TensorFlow
train_imgs = [str(p) for p in train_imgs]
val_imgs = [str(p) for p in val_imgs]
train_masks = [str(p) for p in train_masks]
val_masks = [str(p) for p in val_masks]

print(f"Train: {len(train_imgs)}")
print(f"Validation: {len(val_imgs)}")

# -----------------------------
# Data loader
# -----------------------------

def load_pair(img_path, mask_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    mask = tf.cast(mask > 127, tf.float32)

    return img, mask


def make_dataset(images, masks):

    ds = tf.data.Dataset.from_tensor_slices((images, masks))

    def _load(i, m):
        return tf.py_function(
            func=load_pair,
            inp=[i, m],
            Tout=(tf.float32, tf.float32),
        )

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    def _shape(img, mask):
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        mask.set_shape((IMG_SIZE, IMG_SIZE, 1))
        return img, mask

    ds = ds.map(_shape)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(train_imgs, train_masks)
val_ds = make_dataset(val_imgs, val_masks)

# -----------------------------
# Dice Metric
# -----------------------------

def dice_coef(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (2.0 * intersection + 1.0) / (union + 1.0)


def dice_loss(y_true, y_pred):

    smooth = 1.0

    intersection = tf.reduce_sum(y_true * y_pred)

    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice


def bce_dice_loss(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    return bce + dice_loss(y_true, y_pred)

# -----------------------------
# U-Net Model
# -----------------------------

def conv_block(x, filters):

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)

    return x


def encoder_block(x, filters):

    c = conv_block(x, filters)
    p = tf.keras.layers.MaxPooling2D()(c)

    return c, p


def decoder_block(x, skip, filters):

    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)

    return x


def build_unet(input_shape=(256, 256, 3)):

    inputs = tf.keras.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    bridge = conv_block(p4, 1024)

    d1 = decoder_block(bridge, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        dtype="float32",
    )(d4)

    return tf.keras.Model(inputs, outputs)


model = build_unet((IMG_SIZE, IMG_SIZE, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
    loss=bce_dice_loss,
    metrics=[dice_coef, "accuracy"],
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(WEIGHTS_DIR / "unet_best.keras"),
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        mode="max",
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
    ),
]

# -----------------------------
# Train
# -----------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# -----------------------------
# Save final model
# -----------------------------

model.save(str(WEIGHTS_DIR / "unet_final.keras"))

print("\nTraining completed.")
print("Best model:", WEIGHTS_DIR / "unet_best.keras")