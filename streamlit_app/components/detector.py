from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
YOLO_PATH = ROOT / "weights" / "yolov8_tbx11k_best.pt"

# Load model once
model = YOLO(str(YOLO_PATH))


def detect_lesions(image):
    """
    Runs YOLOv8 TB lesion detection.

    Args:
        image (PIL.Image)

    Returns:
        detected_img (PIL.Image)
        lesion_found (bool)
        detections (list)
    """

    rgb = image.convert("RGB")
    frame = np.array(rgb)

    results = model.predict(
        source=frame,
        conf=0.30,
        iou=0.45,
        verbose=False,
    )

    output = frame.copy()
    detections = []

    if len(results) == 0:
        return Image.fromarray(output), False, detections

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return Image.fromarray(output), False, detections

    # Highest confidence first
    order = np.argsort(boxes.conf.cpu().numpy())[::-1]

    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, box_idx in enumerate(order, start=1):

        box = boxes[box_idx]

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])

        detections.append({
            "id": idx,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })

        # Bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 60, 60), 4)

        # -------- Region ID --------
        id_text = f"#{idx}"
        (tw, th), _ = cv2.getTextSize(id_text, font, 0.9, 2)

        cv2.rectangle(
            output,
            (x1, max(0, y1 - th - 12)),
            (x1 + tw + 10, y1),
            (255, 60, 60),
            -1,
        )

        cv2.putText(
            output,
            id_text,
            (x1 + 5, y1 - 6),
            font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # -------- Confidence --------
        conf_text = f"{conf:.2f}"
        (cw, ch), _ = cv2.getTextSize(conf_text, font, 0.75, 2)

        by = min(output.shape[0] - 5, y2 + ch + 12)

        cv2.rectangle(
            output,
            (x1, by - ch - 8),
            (x1 + cw + 10, by + 2),
            (30, 30, 50),
            -1,
        )

        cv2.putText(
            output,
            conf_text,
            (x1 + 5, by - 4),
            font,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(output), True, detections