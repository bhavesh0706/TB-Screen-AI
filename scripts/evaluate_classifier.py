from pathlib import Path
from typing import cast
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16

TEST_DIR = ROOT / "data" / "test"

# Load dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

# Load model
model = tf.keras.models.load_model(str(ROOT / "weights" / "densenet121_tb_best.keras"))

# Predictions
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print("\n===== TEST RESULTS =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

# Save classification report
report = cast(str, classification_report(y_true, y_pred, output_dict=False))
with open(RESULTS / "classification_report.txt", "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Tuberculosis"])
disp.plot(cmap="Blues")
plt.savefig(RESULTS / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(y_true, y_prob)
plt.savefig(RESULTS / "roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved to results/")
print(" - confusion_matrix.png")
print(" - roc_curve.png")
print(" - classification_report.txt")