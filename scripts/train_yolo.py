from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT / "configs" / "tbx11k_yolo.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

print("=" * 60)
print("PulmoTB AI — YOLOv8 Lesion Detection Training")
print("=" * 60)
print(f"Dataset: {DATA_YAML}")
print()

# Load pretrained YOLOv8 Nano
model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=8,
    workers=0,
    device="cpu",      # Windows TensorFlow setup is CPU-only
    project="runs/detect",
    name="tbx11k",
    exist_ok=True,
)

print("\nTraining completed.")
print(f"Best weights: {ROOT / 'runs' / 'detect' / 'tbx11k' / 'weights' / 'best.pt'}")