import streamlit as st
from PIL import Image

from components.classifier import predict_tb
from components.segmenter import segment_lungs
from components.gradcam import generate_gradcam
from components.detector import detect_lesions
from components.report_generator import create_report
from components.recommendation import get_patient_recommendation

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="PulmoTB AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# CSS
# -------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
}

.stApp{
    background:
        radial-gradient(circle at top left,#0B82D9 0%,#06285D 28%,#020B22 62%,#010611 100%);
    color:white;
}

.block-container{
    max-width:1200px;
    padding-top:1rem;
    padding-bottom:2rem;
}

header[data-testid="stHeader"]{
    background:rgba(0,0,0,0);
}

#MainMenu,
footer,
[data-testid="stSidebar"]{
    visibility:hidden;
    display:none !important;
}

/* Hero Header */
.hero-card{
    background:linear-gradient(135deg,#0A1836 0%,#0F2D63 45%,#1E56D8 100%);
    border:1px solid rgba(59,130,246,.25);
    border-radius:26px;
    padding:28px;
    box-shadow:0 20px 45px rgba(0,0,0,.45);
    margin-bottom:18px;
}

.hero-title{
    font-size:38px;
    font-weight:800;
    color:white;
    margin:0;
}

.hero-sub{
    color:#BFD7FF;
    font-size:16px;
    margin-top:4px;
}

.status-pill{
    display:inline-block;
    padding:8px 16px;
    border-radius:999px;
    background:rgba(74,222,128,.12);
    border:1px solid rgba(74,222,128,.35);
    color:#4ADE80;
    font-weight:700;
    font-size:13px;
}

/* Section title bar (self-contained: opens AND closes in one call, no stray content) */
.section-title-bar{
    background:rgba(4,15,35,.72);
    border:1px solid rgba(75,145,255,.16);
    border-radius:22px;
    padding:18px 22px;
    margin:18px 0 14px 0;
    backdrop-filter:blur(18px);
    box-shadow:
        0 0 25px rgba(40,110,255,.12),
        inset 0 0 14px rgba(255,255,255,.02);
}

.section-title{
    font-size:1.35rem;
    font-weight:800;
    color:white;
}

/* Glass Cards */
.report-card{
    background:rgba(7,19,38,.82);
    border:1px solid rgba(46,105,255,.18);
    border-radius:16px;
    box-shadow:
        0 0 18px rgba(46,105,255,.08),
        inset 0 0 8px rgba(255,255,255,.02);
}

.report-card:hover{
    border-color:#2E69FF;
    box-shadow:
        0 0 28px rgba(46,105,255,.18),
        inset 0 0 12px rgba(255,255,255,.03);
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #0B1730 !important;
    color: white !important;
    border: 1px solid rgba(90,150,255,.20) !important;
    border-radius: 14px !important;
}

.stNumberInput button {
    background: #0B1730 !important;
    color: #D6E6FF !important;
    border: none !important;
}

.stNumberInput button:hover {
    background: #13264A !important;
    color: white !important;
}

.stNumberInput div[data-baseweb="input"] {
    background: #0B1730 !important;
    border-radius: 14px !important;
}

.stSelectbox div[data-baseweb="select"] * {
    background: #0B1730 !important;
    color: white !important;
}

.stSelectbox svg {
    fill: #BFD7FF !important;
}

[data-testid="stFileUploader"] button {
    background: #0B1730 !important;
    color: white !important;
    border: 1px solid rgba(90,150,255,.25) !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #13264A !important;
}

/* Upload Panel */
.upload-card{
    background:rgba(6,18,42,.72);
    border:1px solid rgba(70,150,255,.18);
    border-radius:18px;
    padding:20px;
    height:100%;
    backdrop-filter:blur(12px);
}

.upload-title{
    color:white;
    font-size:1.2rem;
    font-weight:800;
    margin-bottom:16px;
}

.upload-info{
    background:#0A1835;
    border:1px solid rgba(80,150,255,.18);
    border-radius:12px;
    padding:10px 14px;
    color:#BFD8FF;
    font-size:.8rem;
    margin-top:12px;
}

.success-strip{
    background:linear-gradient(90deg,#083D2E,#0A5B42);
    border:1px solid #18A16C;
    color:#68F3A8;
    border-radius:12px;
    padding:10px 14px;
    font-size:.85rem;
    font-weight:600;
    margin-top:12px;
}

/* Image Cards */
.img-card{
    background:rgba(10,22,50,.92);
    border:1px solid rgba(59,130,246,.18);
    border-radius:16px;
    padding:10px 14px;
    margin-bottom:8px;
    transition:.25s;
}

.img-card:hover{
    transform:translateY(-2px);
    border-color:#2E69FF;
}

.img-title{
    color:white;
    font-size:13px;
    font-weight:700;
}

[data-testid="stImage"] img{
    border-radius:14px !important;
    max-height:260px !important;
    width:auto !important;
    max-width:100% !important;
    object-fit:contain !important;
    background:#090D16;
}

[data-testid="stFileUploaderDropzone"]{
    background:#111827;
    border:2px dashed rgba(90,150,255,.28);
    border-radius:18px;
    padding:24px;
    min-height:160px;
}

/* Download Button */
.stDownloadButton button{
    width:100%;
    border-radius:14px;
    border:none;
    color:white;
    font-weight:700;
    background:linear-gradient(90deg,#2563EB,#3B82F6);
    box-shadow:0 10px 24px rgba(37,99,235,.35);
    transition:.2s;
    height:50px;
}

.stDownloadButton button:hover{
    transform:translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 1. HERO
# -------------------------------------------------

st.markdown("""
<div class="hero-card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:18px;">
        <div>
            <div class="hero-title">🫁 PulmoTB AI</div>
            <div class="hero-sub">AI-Assisted Pulmonary Tuberculosis Screening Platform</div>
        </div>
        <div class="status-pill">● System Ready</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 2. PATIENT INFORMATION
# -------------------------------------------------

st.markdown(
    '<div class="section-title-bar"><div class="section-title">Patient Information</div></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    patient_id = st.text_input("Patient ID", max_chars=4, placeholder="0001")
    if patient_id and not patient_id.isdigit():
        st.error("Patient ID must contain only 4 digits.")
        st.stop()

with c2:
    patient_name = st.text_input("Patient Name", placeholder="Enter name")
with c3:
    patient_age = st.number_input("Age", 1, 120, 30)
with c4:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# -------------------------------------------------
# 3. CHEST X-RAY ACQUISITION  (+ Grad-CAM alongside the original)
# -------------------------------------------------

st.markdown(
    '<div class="section-title-bar"><div class="section-title">Chest X-ray Acquisition</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<div class="upload-title">Upload Chest Radiograph</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PNG / JPG / JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="upload-info"><b>Formats:</b> PNG · JPG · JPEG <br>'
        '<b>Minimum Resolution:</b> 512×512</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file:
        st.markdown('<div class="success-strip">✓ Radiograph processed successfully</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

with right:
    if image is not None:
        with st.spinner("Executing screening & localization models..."):
            classification = predict_tb(image)
            segmented = segment_lungs(image)

            try:
                gradcam_img = generate_gradcam(image, lung_mask=segmented)
            except Exception as e:
                st.warning(f"Grad-CAM unavailable: {e}")
                gradcam_img = image

            detected_img, lesion_found, detections = detect_lesions(image)

        # persist for the report generator / PDF
        st.session_state["patient_id"] = patient_id
        st.session_state["patient_name"] = patient_name
        st.session_state["patient_age"] = patient_age
        st.session_state["patient_gender"] = patient_gender
        st.session_state["uploaded_image"] = image
        st.session_state["classification"] = classification
        st.session_state["segmented_image"] = segmented
        st.session_state["gradcam_image"] = gradcam_img
        st.session_state["detected_image"] = detected_img
        st.session_state["lesion_found"] = lesion_found
        st.session_state["detections"] = detections
        st.session_state["lesion_count"] = len(detections)

        img_col1, img_col2 = st.columns(2, gap="medium")
        with img_col1:
            st.markdown(
                '<div class="img-card"><div class="img-title">🩻 Original Radiograph</div></div>',
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)
        with img_col2:
            st.markdown(
                '<div class="img-card"><div class="img-title">🔥 Grad-CAM Heatmap</div></div>',
                unsafe_allow_html=True,
            )
            st.image(gradcam_img, use_container_width=True)
    else:
        st.markdown(
            '<div style="height:230px;display:flex;align-items:center;justify-content:center;'
            'color:#8CA8D6;border:2px dashed rgba(90,150,255,.25);border-radius:18px;'
            'background:rgba(6,18,42,.72);">Upload a chest X-ray to generate live Grad-CAM saliency.</div>',
            unsafe_allow_html=True,
        )

if uploaded_file is None:
    st.stop()

prediction = classification["label"]
confidence = classification["confidence"] * 100
tb_probability = classification["tb_probability"] * 100

rec = get_patient_recommendation(
    tb_probability=tb_probability,
    age=patient_age,
    gender=patient_gender,
)

# -------------------------------------------------
# 4. ANATOMICAL & LESION LOCALIZATION
# -------------------------------------------------

st.markdown(
    '<div class="section-title-bar"><div class="section-title">Anatomical & Lesion Localization</div></div>',
    unsafe_allow_html=True,
)

loc1, loc2 = st.columns(2, gap="medium")

with loc1:
    st.markdown(
        '<div class="img-card"><div class="img-title">🫁 Lung Segmentation (U-Net)</div></div>',
        unsafe_allow_html=True,
    )
    st.image(st.session_state["segmented_image"], use_container_width=True)

with loc2:
    st.markdown(
        '<div class="img-card"><div class="img-title">🎯 Lesion Localization (YOLOv8)</div></div>',
        unsafe_allow_html=True,
    )
    st.image(st.session_state["detected_image"], use_container_width=True)
    if not st.session_state.get("lesion_found", True):
        st.info("No localized lesions detected by YOLOv8.")

# -------------------------------------------------
# 5. AI SCREENING REPORT (single, RetinexAI style)
# -------------------------------------------------

st.markdown(
    '<div class="section-title-bar"><div class="section-title">AI Screening Report</div></div>',
    unsafe_allow_html=True,
)

report_left, report_right = st.columns([1, 1.15], gap="large")

# ---------------- LEFT PANEL ----------------

with report_left:

    condition_text = "TB Positive" if prediction == "Tuberculosis" else "TB Negative"
    st.markdown(
        '<div class="report-card" style="padding:22px;margin-bottom:16px;">'
        '<div style="color:#8EA8CF;font-size:13px;font-weight:700;">Predicted Condition</div>'
        f'<div style="font-size:30px;font-weight:800;color:white;margin-top:8px;">{condition_text}</div>'
        '<div style="color:#9FB5D9;font-size:13px;margin-top:6px;">AI-assisted screening output</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    m_c1, m_c2 = st.columns(2)

    with m_c1:
        st.markdown(
            '<div class="report-card" style="padding:18px;text-align:center;">'
            '<div style="color:#8EA8CF;font-size:12px;">Confidence</div>'
            f'<div style="font-size:24px;font-weight:800;color:white;margin-top:6px;">{confidence:.1f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with m_c2:
        st.markdown(
            '<div class="report-card" style="padding:18px;text-align:center;">'
            '<div style="color:#8EA8CF;font-size:12px;">TB Probability</div>'
            f'<div style="font-size:24px;font-weight:800;color:white;margin-top:6px;">{tb_probability:.1f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    badge_style = (
        "margin-top:10px;display:inline-block;padding:8px 18px;border-radius:999px;"
        f"background:{rec['color']}22;border:1px solid {rec['color']};color:{rec['color']};font-weight:700;"
    )
    st.markdown(
        '<div class="report-card" style="padding:20px;margin-top:16px;">'
        '<div style="color:#8EA8CF;font-size:12px;">Risk Assessment</div>'
        f'<div style="{badge_style}">{rec["risk"]}</div>'
        f'<div style="margin-top:18px;color:#D6E6FF;line-height:1.7;">{rec["summary"]}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-card" style="padding:20px;margin-top:16px;">'
        '<div style="color:white;font-weight:700;font-size:17px;margin-bottom:14px;">Patient Category</div>'
        '<div style="display:flex;gap:18px;">'
            '<div style="flex:1;background:#08152B;border:1px solid rgba(59,130,246,.16);'
            'border-radius:14px;padding:14px;">'
                '<div style="color:#8EA8CF;font-size:12px;">Age Group</div>'
                f'<div style="color:white;font-size:20px;font-weight:800;margin-top:4px;">{rec["age_group"]}</div>'
            '</div>'
            '<div style="flex:1;background:#08152B;border:1px solid rgba(59,130,246,.16);'
            'border-radius:14px;padding:14px;">'
                '<div style="color:#8EA8CF;font-size:12px;">Gender</div>'
                f'<div style="color:white;font-size:20px;font-weight:800;margin-top:4px;">{rec["gender"]}</div>'
            '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------- RIGHT PANEL ----------------

with report_right:

    st.markdown(
        '<div style="font-size:22px;font-weight:800;color:white;margin-bottom:14px;">'
        'Recommended Next Steps</div>',
        unsafe_allow_html=True,
    )

    icon_map = [
        ("🚑", "#EF4444"),
        ("🧪", "#8B5CF6"),
        ("📄", "#2563EB"),
        ("🩺", "#06B6D4"),
        ("🫁", "#EAB308"),
        ("🏥", "#10B981"),
        ("👨‍⚕️", "#3B82F6"),
    ]

    rows_html = ""
    total = len(rec["steps"])
    for i, step in enumerate(rec["steps"]):
        icon, accent = icon_map[i % len(icon_map)]
        is_last = i == total - 1
        border = "" if is_last else "border-bottom:1px solid rgba(90,150,255,.10);"
        rows_html += (
            f'<div style="display:flex;gap:12px;align-items:center;padding:10px 4px;{border}">'
                f'<div style="width:30px;height:30px;min-width:30px;border-radius:50%;background:{accent}22;'
                f'display:flex;align-items:center;justify-content:center;font-size:14px;'
                f'border:1px solid {accent};">{icon}</div>'
                f'<div style="color:white;font-size:13.5px;font-weight:600;line-height:1.35;">{step}</div>'
            '</div>'
        )

    st.markdown(
        f'<div class="report-card" style="padding:6px 16px;">{rows_html}</div>',
        unsafe_allow_html=True,
    )

pdf_bytes = create_report(st.session_state)

st.download_button(
    "⬇ Download Official Clinical Screening Report (PDF)",
    pdf_bytes,
    file_name=f"PulmoTB_AI_Report_{patient_id or '0001'}.pdf",
    mime="application/pdf",
    use_container_width=True,
)