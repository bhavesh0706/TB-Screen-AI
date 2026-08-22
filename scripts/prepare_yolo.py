from pathlib import Path
import json
import shutil
import random

ROOT = Path(__file__).resolve().parents[1]

JSON_DIR = ROOT / "data" / "raw" / "TBX11K" / "annotations" / "json"
IMG_ROOT = ROOT / "data" / "raw" / "TBX11K" / "imgs"

YOLO_ROOT = ROOT / "data" / "yolo"

TRAIN_IMG = YOLO_ROOT / "images" / "train"
VAL_IMG = YOLO_ROOT / "images" / "val"
TRAIN_LBL = YOLO_ROOT / "labels" / "train"
VAL_LBL = YOLO_ROOT / "labels" / "val"

for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    p.mkdir(parents=True, exist_ok=True)

random.seed(42)

records = {}

# -----------------------------
# Read all COCO annotation files
# -----------------------------

json_files = sorted(JSON_DIR.glob("*.json"))

print(f"Found {len(json_files)} COCO annotation files.")

for jf in json_files:

    with open(jf) as f:
        data = json.load(f)

    image_map = {img["id"]: img for img in data["images"]}

    for ann in data["annotations"]:

        img = image_map[ann["image_id"]]

        file_name = img["file_name"]

        w = img["width"]
        h = img["height"]

        x, y, bw, bh = ann["bbox"]

        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw /= w
        bh /= h

        records.setdefault(file_name, []).append(
            f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

# -----------------------------
# Train / Validation split
# -----------------------------

files = list(records.keys())
random.shuffle(files)

split = int(0.8 * len(files))

train_files = files[:split]
val_files = files[split:]

print(f"Train images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")

# -----------------------------
# Copy images and labels
# -----------------------------

def export(file_list, img_dst, lbl_dst):

    copied = 0

    for rel in file_list:

        src = IMG_ROOT / rel

        if not src.exists():
            continue

        shutil.copy2(src, img_dst / src.name)

        with open(lbl_dst / f"{src.stem}.txt", "w") as f:
            f.write("\n".join(records[rel]))

        copied += 1

    return copied

train_count = export(train_files, TRAIN_IMG, TRAIN_LBL)
val_count = export(val_files, VAL_IMG, VAL_LBL)

# -----------------------------
# Write YAML automatically
# -----------------------------

CONFIG_DIR = ROOT / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

yaml_text = """path: data/yolo

train: images/train
val: images/val

names:
  0: tuberculosis
"""

yaml_path = CONFIG_DIR / "tbx11k_yolo.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("\nYOLO dataset created successfully.")
print(f"Train exported: {train_count}")
print(f"Val exported: {val_count}")
print(f"YAML saved: {yaml_path}")