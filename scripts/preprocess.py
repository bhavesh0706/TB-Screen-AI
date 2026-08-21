from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/dataset.yaml"))

SIZE = cfg["image_size"]

manifest = pd.read_csv(ROOT / cfg["output"]["manifest"])
processed = ROOT / cfg["output"]["processed"]

total = len(manifest[manifest.status == "OK"])
count = 0

for _, row in manifest[manifest.status == "OK"].iterrows():
    src = ROOT / row.path
    dst = processed / row.dataset / row.filename
    dst.parent.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(src).convert("RGB"))
    img = cv2.resize(img, (SIZE, SIZE))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    Image.fromarray(img).save(dst)

    count += 1
    if count % 100 == 0 or count == total:
        print(f"Processed {count}/{total}")