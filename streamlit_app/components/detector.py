from pathlib import Path
from typing import cast

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "tbx11k"
    / "weights"
    / "best.pt"
)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def detect_lesions(image: Image.Image) -> Image.Image:
    """
    Run YOLOv8 lesion detection and return the annotated image.
    """
    model = load_model()

    results = model.predict(
        source=np.array(image),
        conf=0.25,
        verbose=False,
    )

    # Get the first prediction in a Pylance-friendly way
    result = cast(Results, next(iter(results)))
    plotted = result.plot()

    return Image.fromarray(plotted)