from pathlib import Path
from typing import cast
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "weights" / "densenet121_tb_best.keras"
VAL_DIR = ROOT / "data" / "val"

REPORT_DIR = ROOT / "reports" / "evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_FILE = ROOT / "evaluation" / "threshold.json"
THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------------------------------------
# Validation Dataset (Identical to Training Directory Loader)
# ------------------------------------------------------------

if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation folder not found at {VAL_DIR}")

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

# Match classifier preprocessing (0–255 float32)
val_dataset = val_dataset.map(
    lambda x, y: (tf.cast(x, tf.float32), y),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

print("\nRunning batched inference...")

y_prob = cast(np.ndarray, model.predict(val_dataset, verbose=1).flatten())

# Collect true labels in the exact same unshuffled order
y_true = np.concatenate([y.numpy().flatten() for _, y in val_dataset])

# ------------------------------------------------------------
# Threshold Optimization (Youden's J Statistic)
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)

best_threshold = float(thresholds[best_idx])

pred_labels = cast(np.ndarray, (y_prob >= best_threshold).astype(np.int32))

auc = roc_auc_score(y_true, y_prob)
f1 = f1_score(y_true, pred_labels)

precision, recall, _ = precision_recall_curve(y_true, y_prob)

# ------------------------------------------------------------
# Save Threshold & Metrics
# ------------------------------------------------------------

with open(THRESHOLD_FILE, "w") as f:
    json.dump(
        {
            "threshold": round(best_threshold, 4),
            "auc": round(float(auc), 4),
            "f1": round(float(f1), 4),
        },
        f,
        indent=4,
    )

# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "roc_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# PR Curve
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(REPORT_DIR / "pr_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_true, pred_labels)

disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "TB"])
disp.plot()
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("MODEL THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 55)
print(f"Evaluation Samples : {len(y_true)}")
print(f"Optimal Threshold  : {best_threshold:.3f}")
print(f"AUC Score          : {auc:.3f}")
print(f"F1 Score           : {f1:.3f}")
print("=" * 55)

print("\nFiles Generated:")
print(f"• {THRESHOLD_FILE}")
print(f"• {REPORT_DIR / 'roc_curve.png'}")
print(f"• {REPORT_DIR / 'pr_curve.png'}")
print(f"• {REPORT_DIR / 'confusion_matrix.png'}")