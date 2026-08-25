from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

SRC_IMAGES = ROOT / "data" / "raw" / "LungSegmentation704" / "image"
SRC_MASKS = ROOT / "data" / "raw" / "LungSegmentation704" / "mask"

DST_IMAGES = ROOT / "data" / "processed" / "LungSegmentation704" / "image"
DST_MASKS = ROOT / "data" / "processed" / "LungSegmentation704" / "mask"

DST_IMAGES.mkdir(parents=True, exist_ok=True)
DST_MASKS.mkdir(parents=True, exist_ok=True)

matched = 0

for img in SRC_IMAGES.glob("*.png"):

    # Example:
    # CHNCXR_0001_0.png
    # -> CHNCXR_0001_0_mask.png
    mask_name = img.stem + "_mask.png"
    mask_path = SRC_MASKS / mask_name

    if mask_path.exists():
        shutil.copy2(img, DST_IMAGES / img.name)
        shutil.copy2(mask_path, DST_MASKS / img.name)
        matched += 1

print("=" * 50)
print(f"Matched pairs : {matched}")
print("=" * 50)