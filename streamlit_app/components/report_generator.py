from io import BytesIO
from datetime import datetime

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


def pil_to_buffer(img: PILImage.Image):
    """Convert PIL image to ReportLab image buffer."""
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def create_report(session_state):
    """
    Generates the final patient screening PDF.
    Returns PDF bytes.
    """

    cls = session_state["classification"]
    original = session_state["uploaded_image"]
    mask = session_state["segmented_image"]
    gradcam = session_state["gradcam_image"]
    yolo = session_state["detected_image"]

    patient_id = session_state.get("patient_id", "0001")

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=(8.27 * inch, 11.69 * inch),  # A4
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER
    title_style.textColor = colors.HexColor("#0F172A")

    heading = styles["Heading2"]
    heading.textColor = colors.HexColor("#1D4ED8")

    normal = styles["BodyText"]

    story = []

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    story.append(Paragraph("<b>AI TB Screening System</b>", title_style))
    story.append(Paragraph("Chest X-ray Tuberculosis Screening Report", normal))
    story.append(Spacer(1, 0.25 * inch))

    # --------------------------------------------------
    # Patient Info
    # --------------------------------------------------

    story.append(Paragraph("<b>Patient Information</b>", heading))

    info = Table(
        [
            ["Patient ID", patient_id],
            ["Date", datetime.now().strftime("%d %B %Y")],
            ["Time", datetime.now().strftime("%I:%M %p")],
        ],
        colWidths=[1.8 * inch, 4.8 * inch],
    )

    info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5E7EB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(info)
    story.append(Spacer(1, 0.2 * inch))

    # --------------------------------------------------
    # AI Summary
    # --------------------------------------------------

    story.append(Paragraph("<b>AI Screening Summary</b>", heading))

    summary = Table(
        [
            ["Prediction", cls["label"]],
            ["Confidence", f"{cls['confidence']*100:.2f}%"],
            ["TB Probability", f"{cls['tb_probability']*100:.2f}%"],
        ],
        colWidths=[2.2 * inch, 4.4 * inch],
    )

    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(summary)
    story.append(Spacer(1, 0.25 * inch))

    # --------------------------------------------------
    # Images
    # --------------------------------------------------

    story.append(Paragraph("<b>AI Visual Analysis</b>", heading))

    def rl_img(img):
        return Image(pil_to_buffer(img), width=2.8 * inch, height=2.8 * inch)

    image_table = Table(
        [
            [rl_img(original), rl_img(gradcam)],
            [Paragraph("<b>Original X-ray</b>", normal),
             Paragraph("<b>Grad-CAM</b>", normal)],
            [rl_img(mask), rl_img(yolo)],
            [Paragraph("<b>Lung Segmentation</b>", normal),
             Paragraph("<b>YOLO Lesion Detection</b>", normal)],
        ],
        colWidths=[3.1 * inch, 3.1 * inch],
    )

    image_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(image_table)
    story.append(Spacer(1, 0.2 * inch))

    # --------------------------------------------------
    # Clinical Interpretation
    # --------------------------------------------------

    story.append(Paragraph("<b>Clinical Interpretation</b>", heading))

    if cls["label"] == "Tuberculosis":
        text = """
        The DenseNet121 classification model detected tuberculosis-like
        radiographic patterns in the uploaded chest X-ray.

        Grad-CAM highlights the image regions that contributed most to
        the classification.

        YOLOv8 attempted to localize suspicious lesions using
        bounding-box detection.
        """
    else:
        text = """
        The DenseNet121 classification model did not detect significant
        tuberculosis-like radiographic patterns.

        Grad-CAM shows the regions that influenced the model's decision.

        YOLOv8 did not identify strong localized lesion patterns.
        """

    story.append(Paragraph(text, normal))
    story.append(Spacer(1, 0.2 * inch))

    # --------------------------------------------------
    # Model Pipeline
    # --------------------------------------------------

    story.append(Paragraph("<b>AI Pipeline Used</b>", heading))

    pipeline = Table(
        [
            ["DenseNet121", "TB Classification"],
            ["U-Net", "Lung Segmentation"],
            ["Grad-CAM", "Model Explainability"],
            ["YOLOv8", "Lesion Localization"],
        ],
        colWidths=[2.3 * inch, 4.3 * inch],
    )

    pipeline.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(pipeline)
    story.append(Spacer(1, 0.25 * inch))

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    story.append(Paragraph("<b>Screening Disclaimer</b>", heading))

    disclaimer = """
    This report is generated by an AI-assisted tuberculosis screening
    system developed for research and screening purposes.

    The results should be interpreted by qualified healthcare
    professionals and confirmed using appropriate clinical and
    laboratory investigations.
    """

    story.append(Paragraph(disclaimer, normal))

    # --------------------------------------------------

    doc.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()