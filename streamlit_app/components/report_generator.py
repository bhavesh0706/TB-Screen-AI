from io import BytesIO
from datetime import datetime

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    ListFlowable,
    ListItem,
    HRFlowable,
)

from components.recommendation import get_patient_recommendation


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

    # FIX: `.get(key, default)` only falls back when the key is missing,
    # not when it's an empty string — use `or` so blank input falls back too.
    patient_id = session_state.get("patient_id") or "0001"
    patient_name = session_state.get("patient_name") or "Not provided"
    patient_age = session_state.get("patient_age", "-")
    patient_gender = session_state.get("patient_gender", "-")

    tb_probability_pct = cls["tb_probability"] * 100

    # Same rule-based engine the dashboard uses, so the PDF and the
    # on-screen report always agree.
    rec = get_patient_recommendation(
        tb_probability=tb_probability_pct,
        age=patient_age if isinstance(patient_age, int) else 30,
        gender=patient_gender if isinstance(patient_gender, str) else "Male",
    )

    risk_color = colors.HexColor(rec["color"])

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

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        fontSize=11,
    )

    heading = styles["Heading2"]
    heading.textColor = colors.HexColor("#1D4ED8")

    normal = styles["BodyText"]
    normal.leading = 15

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=normal,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    story.append(Paragraph("<b>🫁 PulmoTB AI</b>", title_style))
    story.append(Paragraph("AI-Assisted Chest X-ray Tuberculosis Screening Report", subtitle_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#93C5FD")))
    story.append(Spacer(1, 0.2 * inch))

    # --------------------------------------------------
    # Patient Info
    # --------------------------------------------------

    story.append(Paragraph("<b>Patient Information</b>", heading))

    info = Table(
        [
            ["Patient ID", patient_id],
            ["Patient Name", patient_name],
            ["Age", str(patient_age)],
            ["Gender", str(patient_gender)],
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
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ]
        )
    )

    story.append(info)
    story.append(Spacer(1, 0.25 * inch))

    # --------------------------------------------------
    # AI Summary + Risk Badge
    # --------------------------------------------------

    story.append(Paragraph("<b>AI Screening Summary</b>", heading))

    summary = Table(
        [
            ["Prediction", cls["label"]],
            ["Confidence", f"{cls['confidence']*100:.2f}%"],
            ["TB Probability", f"{tb_probability_pct:.2f}%"],
            ["Risk Level", rec["risk"]],
        ],
        colWidths=[2.2 * inch, 4.4 * inch],
    )

    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                ("BACKGROUND", (0, 3), (-1, 3), colors.Color(
                    risk_color.red, risk_color.green, risk_color.blue, alpha=0.15
                )),
                ("TEXTCOLOR", (1, 3), (1, 3), risk_color),
                ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(summary)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(rec["summary"], normal))
    story.append(Spacer(1, 0.25 * inch))

    # --------------------------------------------------
    # Recommended Next Steps
    # --------------------------------------------------

    story.append(Paragraph("<b>Recommended Next Steps</b>", heading))

    step_items = [
        ListItem(Paragraph(step, bullet_style), bulletColor=risk_color)
        for step in rec["steps"]
    ]

    story.append(
        ListFlowable(
            step_items,
            bulletType="bullet",
            start="circle",
            leftIndent=18,
            bulletFontSize=8,
        )
    )
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
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(pipeline)
    story.append(Spacer(1, 0.25 * inch))

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 0.15 * inch))
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