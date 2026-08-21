from pathlib import Path
import yaml
import tensorflow as tf

# ---------- Paths ----------

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "weights").mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / "configs" / "densenet121.yaml"))

TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"

IMG_SIZE = cfg["image_size"]
BATCH_SIZE = cfg["batch_size"]

# ---------- Mixed Precision ----------

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ---------- Datasets ----------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=cfg["seed"]
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------- Model ----------

base = tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = tf.keras.applications.densenet.preprocess_input(inputs)

x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(cfg["dropout"])(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    dtype="float32"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

# ---------- Callbacks ----------

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ROOT / "weights" / "densenet121_tb_best.keras"),
        save_best_only=True,
        monitor="val_auc",
        mode="max"
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    )
]

# ---------- Summary ----------

model.summary()

# ---------- Training ----------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg["epochs"],
    callbacks=callbacks
)

# ---------- Save Final Model ----------

model.save(str(ROOT / "weights" / "densenet121_tb_final.keras"))

print("\nTraining completed.")