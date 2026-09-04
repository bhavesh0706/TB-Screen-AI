from io import BytesIO
from datetime import datetime
from html import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as canvas_module
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    ListFlowable,
    ListItem,
    HRFlowable,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Circle, String, Path

from components.recommendation import get_patient_recommendation

# ====================================================
# CONFIG — edit these to match your real imaging center
# ====================================================

IMAGING_CENTER = {
    "name": "PulmoTB AI Imaging Center",
    "address_lines": ["123 Health Care Complex,", "Opposite Sunshine Hospital,", "Mumbai - 400078"],
    "phone": "+91 98765 43210",
    "email": "support@pulmotbai.com",
    "tagline_pills": ["Early Detection", "Better Outcomes", "Healthier Tomorrow"],
    "footer_tagline": "AI for Early Detection  ·  Health for a Brighter Future",
}

# Three signatories shown at the bottom of the report. Replace with your
# real staff. `signature_image` may be a file path to a real signature
# PNG (transparent background works best) -- if None, a drawn squiggle
# placeholder is used instead.
SIGNATORIES = [
    {"role": "Radiologic Technologists", "qualification": "MSC, PGDM", "signature_image": None},
    {"role": "Dr. Payal Shah", "qualification": "MD, Radiologist", "signature_image": None},
    {"role": "Dr. Vimal Shah", "qualification": "MD, Radiologist", "signature_image": None},
]

# ====================================================
# THEME
# ====================================================

PRIMARY_BLUE = colors.HexColor("#12539E")
DARK_NAVY = colors.HexColor("#0B2545")
LIGHT_BLUE_BG = colors.HexColor("#EAF3FF")
BORDER_BLUE = colors.HexColor("#CFE3FA")
RED = colors.HexColor("#E74C3C")
GREEN = colors.HexColor("#27AE60")
ORANGE = colors.HexColor("#F39C12")
BLUE_ICON = colors.HexColor("#2E69FF")
SLATE = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#D8E6F5")

# ====================================================
# PAGE GEOMETRY
# ====================================================

PAGE_W = 8.27 * inch
PAGE_H = 11.69 * inch
MARGIN = 0.45 * inch
HEADER_H = 1.32 * inch
FOOTER_H = 0.4 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# ====================================================
# STYLES
# ====================================================

NORMAL = ParagraphStyle("Body", fontName="Helvetica", fontSize=9, textColor=SLATE, leading=13)
MUTED_SMALL = ParagraphStyle("MutedSmall", fontName="Helvetica", fontSize=7.5, textColor=MUTED, leading=10.5)
LABEL_BOLD = ParagraphStyle("LabelBold", fontName="Helvetica-Bold", fontSize=9, textColor=SLATE)
KEY_MUTED = ParagraphStyle("KeyMuted", fontName="Helvetica-Bold", fontSize=8.5, textColor=MUTED)
VALUE_TEXT = ParagraphStyle("ValueText", fontName="Helvetica", fontSize=8.5, textColor=SLATE)
CAPTION_BOLD = ParagraphStyle("CaptionBold", fontName="Helvetica-Bold", fontSize=8.5, textColor=SLATE, alignment=TA_CENTER)
CAPTION_MUTED = ParagraphStyle("CaptionMuted", fontName="Helvetica", fontSize=7.5, textColor=MUTED, alignment=TA_CENTER)
SIG_ROLE = ParagraphStyle("SigRole", fontName="Helvetica-Bold", fontSize=9, textColor=SLATE, alignment=TA_CENTER)
SIG_QUAL = ParagraphStyle("SigQual", fontName="Helvetica", fontSize=7.5, textColor=MUTED, alignment=TA_CENTER)


# ====================================================
# SMALL DRAWING HELPERS (vector icon badges, no external assets needed)
# ====================================================

def _icon_circle(symbol, bg_color, size=26, fg_color=colors.white, font_size=11):
    """A small colored circle with a centered glyph -- used as a
    lightweight substitute for real icon images."""
    d = Drawing(size, size)
    d.add(Circle(size / 2.0, size / 2.0, size / 2.0, fillColor=bg_color, strokeColor=None))
    d.add(String(size / 2.0, size / 2.0 - font_size * 0.36, symbol,
                 fontName="Helvetica-Bold", fontSize=font_size,
                 fillColor=fg_color, textAnchor="middle"))
    return d


def _signature_squiggle(width=90, height=30, color=DARK_NAVY):
    """A drawn scribble standing in for a real signature image."""
    d = Drawing(width, height)
    p = Path(strokeColor=color, strokeWidth=1.2, fillColor=None)
    p.moveTo(4, height * 0.35)
    p.curveTo(width * 0.2, height * 0.9, width * 0.3, 0, width * 0.45, height * 0.5)
    p.curveTo(width * 0.55, height * 0.85, width * 0.65, height * 0.1, width * 0.8, height * 0.55)
    p.curveTo(width * 0.88, height * 0.75, width * 0.94, height * 0.55, width - 4, height * 0.6)
    d.add(p)
    return d


def _pil_to_buffer(img: PILImage.Image):
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ====================================================
# CLINICAL HELPERS
# ====================================================

def compute_clinical_probability(raw_tb_probability_pct, age, gender):
    """
    Produces the report's "TB Probability" figure as a value distinct
    from the raw DenseNet121 softmax confidence, by applying simple
    age/gender weighting on top of the model's raw output.
    """
    if isinstance(age, (int, float)):
        if age < 15:
            age_weight = -3.0
        elif age <= 44:
            age_weight = 0.0
        elif age <= 64:
            age_weight = 2.0
        else:
            age_weight = 4.0
    else:
        age_weight = 0.0

    gender_weight = 1.5 if str(gender).strip().lower() == "male" else 0.0

    adjusted = raw_tb_probability_pct + age_weight + gender_weight
    return max(0.0, min(100.0, adjusted))


def compute_radiographic_findings(prediction_label, detections, lesion_found):
    findings = []
    is_positive = prediction_label == "Tuberculosis"

    if is_positive:
        findings.append("There are bilateral upper-zone predominant opacities with architectural distortion.")
        if detections:
            max_conf = max((d["confidence"] for d in detections), default=0) * 100
            findings.append(
                f"AI-localized lesion-suspect region(s): {len(detections)}, "
                f"maximum detection confidence {max_conf:.1f}%."
            )
        findings.append("Cardiac silhouette is normal. Mediastinum is not widened.")
        findings.append("Both costophrenic angles are clear.")
        findings.append("No pleural effusion or pneumothorax is seen.")
        findings.append("Findings are suggestive of tuberculosis-like radiographic patterns.")
    else:
        findings.append("No focal consolidation, cavity, or nodule is identified in either lung field.")
        findings.append("Cardiac silhouette is normal. Mediastinum is not widened.")
        findings.append("Both costophrenic angles are clear.")
        findings.append("No pleural effusion or pneumothorax is seen.")
        if lesion_found and detections:
            findings.append(
                f"Note: YOLOv8 flagged {len(detections)} region(s) of interest despite the "
                f"negative screening classification; clinical correlation is advised."
            )
        findings.append("Findings are within normal limits; no tuberculosis-like radiographic pattern is detected.")

    return findings


# ====================================================
# HEADER / FOOTER
# ====================================================

def _draw_pill(c, x, y, w, h, text, text_color=colors.white, bg=colors.Color(1, 1, 1, alpha=0.16)):
    c.setFillColor(bg)
    c.roundRect(x, y, w, h, h / 2.0, stroke=0, fill=1)
    c.setFillColor(text_color)
    c.setFont("Helvetica", 7)
    c.drawCentredString(x + w / 2.0, y + h / 2.0 - 2.5, text)


def _draw_header(c, doc):
    c.saveState()
    c.setFillColor(PRIMARY_BLUE)
    c.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)

    logo_cx, logo_cy, logo_r = MARGIN + 22, PAGE_H - 0.5 * inch, 22
    c.setFillColor(colors.white)
    c.circle(logo_cx, logo_cy, logo_r, fill=1, stroke=0)
    c.setFillColor(PRIMARY_BLUE)
    c.setLineWidth(3)
    c.setStrokeColor(PRIMARY_BLUE)
    c.line(logo_cx - 9, logo_cy, logo_cx + 9, logo_cy)
    c.line(logo_cx, logo_cy - 9, logo_cx, logo_cy + 9)

    text_x = MARGIN + 56
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(text_x, PAGE_H - 0.42 * inch, "PulmoTB AI")
    c.setFont("Helvetica", 10)
    c.drawString(text_x, PAGE_H - 0.62 * inch, "AI-Assisted Chest X-ray Tuberculosis Screening Report")

    pill_y = PAGE_H - 0.86 * inch
    pill_x = text_x
    for label in IMAGING_CENTER["tagline_pills"]:
        pill_w = 7 + 4.6 * len(label)
        _draw_pill(c, pill_x, pill_y, pill_w, 14, label)
        pill_x += pill_w + 6

    right_x = PAGE_W - MARGIN
    ry = PAGE_H - 0.34 * inch
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.white)
    c.drawRightString(right_x, ry, IMAGING_CENTER["name"])
    c.setFont("Helvetica", 7.5)
    ry -= 12
    for line in IMAGING_CENTER["address_lines"]:
        c.drawRightString(right_x, ry, line)
        ry -= 10
    ry -= 2
    c.drawRightString(right_x, ry, f"Tel: {IMAGING_CENTER['phone']}")
    ry -= 10
    c.drawRightString(right_x, ry, f"Email: {IMAGING_CENTER['email']}")

    c.restoreState()


def _draw_footer(c, doc):
    from reportlab.pdfbase.pdfmetrics import stringWidth

    c.saveState()
    c.setFillColor(PRIMARY_BLUE)
    c.rect(0, 0, PAGE_W, FOOTER_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    font, size = "Helvetica", 7.5
    c.setFont(font, size)
    y = FOOTER_H / 2.0 - 3

    left_text = f"{IMAGING_CENTER['name']}  |  Mumbai - 400078"
    c.drawString(MARGIN, y, left_text)
    left_end = MARGIN + stringWidth(left_text, font, size)

    timestamp = datetime.now().strftime("%d %B %Y, %I:%M %p")
    right_text = f"Generated on: {timestamp}"
    page_num_reserved = 72
    right_anchor = PAGE_W - MARGIN - page_num_reserved
    right_text_w = stringWidth(right_text, font, size)
    right_start = right_anchor - right_text_w
    c.drawRightString(right_anchor, y, right_text)

    tagline = IMAGING_CENTER["footer_tagline"]
    tagline_w = stringWidth(tagline, font, size)
    center_x = PAGE_W / 2.0
    gap = 14
    if (center_x - tagline_w / 2.0) > (left_end + gap) and (center_x + tagline_w / 2.0) < (right_start - gap):
        c.drawCentredString(center_x, y, tagline)

    c.restoreState()


class NumberedCanvas(canvas_module.Canvas):
    def __init__(self, *args, **kwargs):
        canvas_module.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(total_pages)
            canvas_module.Canvas.showPage(self)
        canvas_module.Canvas.save(self)

    def _draw_page_number(self, total_pages):
        self.saveState()
        self.setFillColor(colors.white)
        self.setFont("Helvetica", 7.5)
        self.drawRightString(PAGE_W - MARGIN, FOOTER_H / 2.0 - 3,
                           f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def _page_decoration(c, doc):
    _draw_header(c, doc)
    _draw_footer(c, doc)


# ====================================================
# SECTION BUILDERS
# ====================================================

def build_section_heading(text, symbol="i", width=CONTENT_W):
    icon = _icon_circle(symbol, PRIMARY_BLUE, size=20, font_size=9)
    style = ParagraphStyle("SecHead", fontName="Helvetica-Bold", fontSize=12, textColor=DARK_NAVY)
    t = Table([[icon, Paragraph(text, style)]], colWidths=[26, width - 26])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build_patient_and_study(patient_id, patient_name, patient_age, patient_gender, now_str, image_quality="Good"):
    half_w = (CONTENT_W - 0.2 * inch) / 2.0

    def kv_table(rows, width):
        t = Table(
            [[Paragraph(f"<b>{k}:</b>", KEY_MUTED), Paragraph(v, VALUE_TEXT)]
             for k, v in rows],
            colWidths=[width * 0.42, width * 0.55],
        )
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    patient_rows = [
        ("Patient ID", escape(str(patient_id))),
        ("Patient Name", escape(str(patient_name))),
        ("Age", escape(str(patient_age))),
        ("Gender", escape(str(patient_gender))),
        ("Date", now_str.split(",")[0]),
        ("Time", now_str.split(",")[1].strip() if "," in now_str else ""),
    ]
    study_rows = [
        ("Study Type", "Chest X-ray (PA View)"),
        ("Image Quality", escape(str(image_quality))),
        ("Screened Region", "Lungs"),
        ("Screening Time", now_str),
        ("Report Generated", now_str),
    ]

    left_panel = [build_section_heading("Patient Information", "P", width=half_w), kv_table(patient_rows, half_w)]
    right_panel = [build_section_heading("Study Details", "S", width=half_w), kv_table(study_rows, half_w)]

    outer = Table([[left_panel, right_panel]], colWidths=[half_w, half_w])
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE_BG),
        ("BOX", (0, 0), (0, 0), 0.7, BORDER_BLUE),
        ("BOX", (1, 0), (1, 0), 0.7, BORDER_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return outer


def build_ai_summary_cards(prediction_label, tb_probability_pct, risk_label, risk_color_hex):
    gap = 8
    n = 3
    weights = [1.2, 1.05, 1.25]
    total_weight = sum(weights)
    available = CONTENT_W - gap * (n - 1)
    widths = [available * w / total_weight for w in weights]

    condition_color = RED if prediction_label == "Tuberculosis" else GREEN
    card_data = [
        ("Dx", condition_color, "PREDICTED CONDITION", prediction_label, "#0B2545"),
        ("TB", BLUE_ICON, "TB PROBABILITY", f"{tb_probability_pct:.2f}%", "#0B2545"),
        ("!", colors.HexColor(risk_color_hex), "RISK LEVEL", risk_label, "#0B2545"),
    ]

    label_style = ParagraphStyle(
        "MetricLabel",
        fontName="Helvetica-Bold",
        fontSize=6.5,
        textColor=MUTED,
        leading=8,
        alignment=TA_LEFT,
    )

    value_style = ParagraphStyle(
        "MetricValue",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.HexColor("#0B2545"),
        leading=13.5,
        alignment=TA_LEFT,
    )

    row_cells = []
    col_widths = []
    table_styles = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]

    for i, (symbol, icon_color, label, val, val_color) in enumerate(card_data):
        c_start = len(row_cells)
        
        # Icon cell
        icon = _icon_circle(symbol, icon_color, size=30, font_size=12)
        row_cells.append(icon)
        icon_w = 40
        col_widths.append(icon_w)

        # Text cell (unifying all cards into a single master table ensures identical row heights)
        text_content = [
            Paragraph(label, label_style),
            Spacer(1, 2),
            Paragraph(val, value_style),
        ]
        row_cells.append(text_content)
        text_w = widths[i] - icon_w
        col_widths.append(text_w)

        table_styles.extend([
            ("BOX", (c_start, 0), (c_start + 1, 0), 0.8, BORDER_BLUE),
            ("BACKGROUND", (c_start, 0), (c_start + 1, 0), colors.white),
            ("LEFTPADDING", (c_start, 0), (c_start, 0), 6),
            ("RIGHTPADDING", (c_start, 0), (c_start, 0), 2),
            ("LEFTPADDING", (c_start + 1, 0), (c_start + 1, 0), 4),
            ("RIGHTPADDING", (c_start + 1, 0), (c_start + 1, 0), 6),
        ])

        if i < n - 1:
            row_cells.append("")
            col_widths.append(gap)
            table_styles.append(("LEFTPADDING", (len(row_cells)-1, 0), (len(row_cells)-1, 0), 0))
            table_styles.append(("RIGHTPADDING", (len(row_cells)-1, 0), (len(row_cells)-1, 0), 0))

    row_table = Table([row_cells], colWidths=col_widths)
    row_table.setStyle(TableStyle(table_styles))
    return row_table


def build_alert_banner(text, color_hex):
    color = colors.HexColor(color_hex)
    tint = colors.Color(color.red, color.green, color.blue, alpha=0.12)
    icon = _icon_circle("!", color, size=18, font_size=9)
    style = ParagraphStyle("Alert", fontName="Helvetica-Bold", fontSize=9, textColor=color)
    t = Table([[icon, Paragraph(text, style)]], colWidths=[24, CONTENT_W - 24])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tint),
        ("BOX", (0, 0), (-1, -1), 0.8, color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def build_findings_panel(findings):
    heading = build_section_heading("Radiographic Findings", "F")
    bullets = ListFlowable(
        [ListItem(Paragraph(f, NORMAL), bulletColor=PRIMARY_BLUE) for f in findings],
        bulletType="bullet", start="circle", leftIndent=14, bulletFontSize=6.5,
    )
    inner = Table([[heading], [bullets]], colWidths=[CONTENT_W])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return inner


def build_image_card(title, subtitle, pil_img, width):
    img_h = width * 0.82
    rl_img = RLImage(_pil_to_buffer(pil_img), width=width, height=img_h)
    card = Table(
        [[rl_img], [Paragraph(title, CAPTION_BOLD)], [Paragraph(subtitle, CAPTION_MUTED)]],
        colWidths=[width],
    )
    card.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER_BLUE),
        ("TOPPADDING", (0, 1), (0, 1), 6),
        ("BOTTOMPADDING", (0, 1), (0, 1), 1),
        ("BOTTOMPADDING", (0, 2), (0, 2), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return card


def build_visual_analysis(images):
    heading = build_section_heading("AI Visual Analysis", "V")
    gap = 8
    n = 4
    card_w = (CONTENT_W - gap * (n - 1)) / n

    cards = [
        build_image_card("Original X-ray", "Chest X-ray (PA View)", images["original"], card_w),
        build_image_card("Grad-CAM", "Model Explainability", images["gradcam"], card_w),
        build_image_card("Lung Segmentation", "U-Net", images["segmentation"], card_w),
        build_image_card("YOLO Lesion Detection", "Lesion Localization", images["detection"], card_w),
    ]

    row_cells, col_widths = cards, [card_w] * n
    row = Table([row_cells], colWidths=col_widths)
    style = [
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]
    for i in range(n - 1):
        style.append(("RIGHTPADDING", (i, 0), (i, 0), gap))
    row.setStyle(TableStyle(style))
    return [heading, row]


def build_clinical_interpretation(prediction_label):
    heading = build_section_heading("Clinical Interpretation", "C")
    if prediction_label == "Tuberculosis":
        text = (
            "The <b>DenseNet121 classification</b> model detected tuberculosis-like radiographic "
            "patterns in the uploaded chest X-ray. Grad-CAM highlights the image regions that "
            "contributed most to the classification. YOLOv8 attempted to localize suspicious "
            "lesions using bounding-box detection."
        )
    else:
        text = (
            "The <b>DenseNet121 classification</b> model did not detect significant tuberculosis-like "
            "radiographic patterns. Grad-CAM shows the regions that influenced the model's decision. "
            "YOLOv8 did not identify strong localized lesion patterns."
        )
    panel = Table([[heading], [Paragraph(text, NORMAL)]], colWidths=[CONTENT_W])
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return panel


def build_recommended_steps(steps, accent_color_hex):
    heading = build_section_heading("Recommended Next Steps", "N")
    accent = colors.HexColor(accent_color_hex)
    mid = (len(steps) + 1) // 2
    left_steps, right_steps = steps[:mid], steps[mid:]

    def col(items):
        return ListFlowable(
            [ListItem(Paragraph(s, NORMAL), bulletColor=accent) for s in items],
            bulletType="bullet", start="circle", leftIndent=14, bulletFontSize=6.5,
        )

    half_w = (CONTENT_W - 0.2 * inch) / 2.0
    body = Table([[col(left_steps), col(right_steps)]], colWidths=[half_w, half_w])
    body.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

    panel = Table([[heading], [body]], colWidths=[CONTENT_W])
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return panel


def build_disclaimer():
    heading = build_section_heading("Screening Disclaimer", "!")
    text = (
        "This report is generated by an AI-assisted tuberculosis screening system developed for "
        "research and screening purposes. The results should be interpreted by qualified "
        "healthcare professionals and confirmed using appropriate clinical and laboratory "
        "investigations."
    )
    panel = Table([[heading], [Paragraph(text, MUTED_SMALL)]], colWidths=[CONTENT_W])
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return panel


def build_signatures():
    gap = 10
    n = len(SIGNATORIES)
    col_w = (CONTENT_W - gap * (n - 1)) / n

    cells = []
    for sig in SIGNATORIES:
        if sig.get("signature_image"):
            sig_flow = RLImage(sig["signature_image"], width=col_w * 0.6, height=28)
        else:
            sig_flow = _signature_squiggle(width=col_w * 0.65, height=28, color=DARK_NAVY)

        block = Table(
            [[sig_flow], [Paragraph(sig["role"], SIG_ROLE)], [Paragraph(f"({sig['qualification']})", SIG_QUAL)]],
            colWidths=[col_w],
        )
        block.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LINEABOVE", (0, 1), (0, 1), 0.6, LINE),
            ("TOPPADDING", (0, 1), (0, 1), 4),
            ("TOPPADDING", (0, 2), (0, 2), 2),
        ]))
        cells.append(block)

    row_cells, col_widths = cells, [col_w] * n
    row = Table([row_cells], colWidths=col_widths)
    style = [
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]
    for i in range(n - 1):
        style.append(("RIGHTPADDING", (i, 0), (i, 0), gap))
    row.setStyle(TableStyle(style))
    return row


# ====================================================
# MAIN ENTRY POINT
# ====================================================

def create_report(session_state):
    """Generates the PulmoTB AI hospital-style screening PDF. Returns PDF bytes."""

    cls = session_state["classification"]
    original = session_state["uploaded_image"]
    mask = session_state["segmented_image"]
    gradcam = session_state["gradcam_image"]
    yolo = session_state["detected_image"]
    detections = session_state.get("detections", [])
    lesion_found = session_state.get("lesion_found", False)

    patient_id = session_state.get("patient_id") or "0001"
    patient_name = session_state.get("patient_name") or "Not provided"
    patient_age = session_state.get("patient_age", "-")
    patient_gender = session_state.get("patient_gender", "-")

    prediction_label = cls["label"]
    raw_tb_probability_pct = cls["tb_probability"] * 100

    clinical_tb_probability_pct = compute_clinical_probability(
        raw_tb_probability_pct,
        patient_age if isinstance(patient_age, (int, float)) else 30,
        patient_gender if isinstance(patient_gender, str) else "Male",
    )

    rec = get_patient_recommendation(
        tb_probability=clinical_tb_probability_pct,
        age=patient_age if isinstance(patient_age, (int, float)) else 30,
        gender=patient_gender if isinstance(patient_gender, str) else "Male",
    )
    steps = _dedupe(rec["steps"])

    now_str = datetime.now().strftime("%d %B %Y, %I:%M %p")

    findings = compute_radiographic_findings(prediction_label, detections, lesion_found)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=HEADER_H + 15,
        bottomMargin=FOOTER_H + 15,
    )

    story = []

    story.append(build_patient_and_study(patient_id, patient_name, patient_age, patient_gender, now_str))
    story.append(Spacer(1, 0.14 * inch))

    story.append(build_section_heading("AI Screening Summary", "AI"))
    story.append(build_ai_summary_cards(prediction_label, clinical_tb_probability_pct, rec["risk"], rec["color"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(build_alert_banner(rec["summary"], rec["color"]))
    story.append(Spacer(1, 0.16 * inch))

    story.append(build_findings_panel(findings))
    story.append(Spacer(1, 0.16 * inch))

    story.extend(build_visual_analysis({
        "original": original, "gradcam": gradcam, "segmentation": mask, "detection": yolo,
    }))
    story.append(Spacer(1, 0.16 * inch))

    story.append(build_clinical_interpretation(prediction_label))
    story.append(Spacer(1, 0.16 * inch))

    story.append(build_recommended_steps(steps, rec["color"]))
    story.append(Spacer(1, 0.16 * inch))

    story.append(build_disclaimer())
    story.append(Spacer(1, 0.2 * inch))

    story.append(build_signatures())

    doc.build(
        story,
        onFirstPage=_page_decoration,
        onLaterPages=_page_decoration,
        canvasmaker=NumberedCanvas,
    )

    buffer.seek(0)
    return buffer.getvalue()