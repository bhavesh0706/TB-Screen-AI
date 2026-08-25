# streamlit_app/pages/1_Dashboard.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from components.classifier import predict_tb
from components.segmenter import segment_lungs
from components.gradcam import generate_gradcam
from components.detector import detect_lesions
from components.report_generator import create_report

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
# PHASE 5: COMMERCIAL HOSPITAL-GRADE CSS
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
    padding-top:0rem;
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
    margin-bottom:20px;
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

/* Glass Section Container */
.section-card{
    background:rgba(4,15,35,.72);
    border:1px solid rgba(75,145,255,.16);
    border-radius:22px;
    padding:22px;
    margin:18px 0;
    backdrop-filter:blur(18px);
    box-shadow:
        0 0 25px rgba(40,110,255,.12),
        inset 0 0 14px rgba(255,255,255,.02);
}

.section-title{
    font-size:1.35rem;
    font-weight:800;
    margin-bottom:16px;
    color:white;
}

/* Form Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]>div{
    background:#101B31 !important;
    border-radius:12px !important;
    border:1px solid rgba(90,150,255,.18) !important;
    color:white !important;
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
    max-height:220px !important;
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
# HELPER: CIRCULAR TB RISK GAUGE
# -------------------------------------------------

def draw_risk_gauge(score: float):
    fig, ax = plt.subplots(figsize=(3.4, 3.4), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)

    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 10)
    ax.axis("off")

    theta = np.linspace(0, np.pi, 200)
    ax.plot(theta, np.full_like(theta, 9), linewidth=18, color="#12305F")

    if score < 40:
        c = "#4ADE80"
    elif score < 70:
        c = "#FACC15"
    else:
        c = "#EF4444"

    end = np.pi * (score / 100)
    ax.plot(np.linspace(0, end, 120), np.full(120, 9), linewidth=18, color=c)

    ax.text(0, 2, f"{score:.0f}%", ha="center", va="center", color="white", fontsize=22, fontweight="bold")
    ax.text(0, -1.5, "TB Risk", ha="center", color="#8EA8CF", fontsize=11)

    return fig

# -------------------------------------------------
# 1. HERO DASHBOARD WITH SYSTEM STATUS
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
# 2. AI PIPELINE TIMELINE
# -------------------------------------------------

st.markdown("""
<div class="section-card" style="padding:14px 22px;margin-bottom:18px;">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <div style="text-align:center;min-width:110px;">
            <div style="font-size:26px;">📤</div>
            <div style="color:#BFD7FF;font-size:12px;font-weight:600;">1. Upload</div>
        </div>
        <div style="color:#2E69FF;font-size:22px;font-weight:bold;">→</div>
        <div style="text-align:center;min-width:110px;">
            <div style="font-size:26px;">🧠</div>
            <div style="color:#BFD7FF;font-size:12px;font-weight:600;">2. Analysis</div>
        </div>
        <div style="color:#2E69FF;font-size:22px;font-weight:bold;">→</div>
        <div style="text-align:center;min-width:110px;">
            <div style="font-size:26px;">🫁</div>
            <div style="color:#BFD7FF;font-size:12px;font-weight:600;">3. Localization</div>
        </div>
        <div style="color:#2E69FF;font-size:22px;font-weight:bold;">→</div>
        <div style="text-align:center;min-width:110px;">
            <div style="font-size:26px;">📄</div>
            <div style="color:#BFD7FF;font-size:12px;font-weight:600;">4. Report</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 3. PATIENT INFORMATION
# -------------------------------------------------

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    patient_id = st.text_input(
        "Patient ID",
        max_chars=4,
        placeholder="0001",
    )

    if patient_id and not patient_id.isdigit():
        st.error("Patient ID must contain only 4 digits.")
        st.stop()

with c2:
    patient_name = st.text_input("Patient Name", placeholder="Enter name")
with c3:
    patient_age = st.number_input("Age", 1, 120, 30)
with c4:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 4. CHEST X-RAY ACQUISITION
# -------------------------------------------------

st.markdown("""
<div class="section-card">
<div class="section-title">Chest X-ray Acquisition</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<div class="upload-title">Upload Chest Radiograph</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PNG / JPG / JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="upload-info">
        <b>Formats:</b> PNG · JPG · JPEG <br>
        <b>Minimum Resolution:</b> 512×512
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        st.markdown('<div class="success-strip">✓ Radiograph processed successfully</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="img-card"><div class="img-title">🩻 Original Radiograph</div></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    else:
        st.markdown("""
        <div style="height:230px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:#8CA8D6;
                    border:2px dashed rgba(90,150,255,.25);
                    border-radius:18px;
                    background:rgba(6,18,42,.72);">
            Upload a chest X-ray to generate live Grad-CAM saliency.
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.stop()

# -------------------------------------------------
# PIPELINE EXECUTION
# -------------------------------------------------

with st.spinner("Executing screening & localization models..."):
    classification = predict_tb(image)
    segmented = segment_lungs(image)
    
    try:
        gradcam_img = generate_gradcam(image, lung_mask=segmented)
    except Exception as e:
        st.warning(f"Grad-CAM unavailable: {e}")
        gradcam_img = image

    detected_img, lesion_found, detections = detect_lesions(image)

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

prediction = classification["label"]
confidence = classification["confidence"] * 100
tb_probability = classification["tb_probability"] * 100

# -------------------------------------------------
# 5. AI SCREENING SUMMARY + CIRCULAR GAUGE
# -------------------------------------------------

risk = (
    "High Risk"
    if tb_probability >= 70
    else "Moderate Risk"
    if tb_probability >= 40
    else "Low Risk"
)

risk_color = (
    "#EF4444"
    if risk == "High Risk"
    else "#FACC15"
    if risk == "Moderate Risk"
    else "#4ADE80"
)

risk_bg = f"{risk_color}22"
prediction_label = "TB Positive" if prediction == "Tuberculosis" else "TB Negative"

st.markdown(f"""
<div class="section-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
        <div>
            <div class="section-title" style="margin:0;">AI Screening Summary</div>
            <div style="color:#9fb5d9;font-size:13px;margin-top:2px;">Automated decision output</div>
        </div>
        <div style="
            background:{risk_bg};
            border:1px solid {risk_color};
            color:{risk_color};
            padding:8px 18px;
            border-radius:999px;
            font-weight:700;
            font-size:13px;">
            {risk}
        </div>
    </div>
""", unsafe_allow_html=True)

sum_col_left, sum_col_right = st.columns([1.6, 1], gap="medium")

with sum_col_left:
    st.markdown(f"""
    <div style="
        background:linear-gradient(145deg,#0b1b36,#0c2347);
        border:1px solid rgba(59,130,246,.22);
        border-radius:18px;
        padding:20px;
        text-align:center;
        margin-bottom:16px;">
        <div style="color:#8ea8cf;font-size:13px;">Primary AI Decision</div>
        <div style="font-size:32px;font-weight:800;color:white;margin-top:4px;">{prediction_label}</div>
        <div style="color:#9fb5d9;font-size:12px;margin-top:4px;">Validated screening threshold</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div style="background:#08152b;border:1px solid rgba(59,130,246,.18);border-radius:14px;padding:14px;text-align:center;">
            <div style="color:#8ea8cf;font-size:12px;">Confidence</div>
            <div style="font-size:22px;font-weight:800;color:white;margin-top:4px;">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background:#08152b;border:1px solid rgba(59,130,246,.18);border-radius:14px;padding:14px;text-align:center;">
            <div style="color:#8ea8cf;font-size:12px;">TB Prob</div>
            <div style="font-size:22px;font-weight:800;color:white;margin-top:4px;">{tb_probability:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div style="background:#08152b;border:1px solid rgba(59,130,246,.18);border-radius:14px;padding:14px;text-align:center;">
            <div style="color:#8ea8cf;font-size:12px;">Status</div>
            <div style="font-size:18px;font-weight:700;color:#4ADE80;margin-top:6px;">Complete</div>
        </div>
        """, unsafe_allow_html=True)

with sum_col_right:
    gauge_fig = draw_risk_gauge(tb_probability)
    st.pyplot(gauge_fig, use_container_width=True)
    plt.close(gauge_fig)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# 6. GRAD-CAM ATTENTION MAP
# -------------------------------------------------

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Grad-CAM Attention Map</div>', unsafe_allow_html=True)
st.markdown('<div class="img-card"><div class="img-title">🔥 Grad-CAM Saliency (Constrained to Lung Mask)</div></div>', unsafe_allow_html=True)
st.image(st.session_state["gradcam_image"], use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 7. ANATOMICAL & LESION LOCALIZATION
# -------------------------------------------------

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Anatomical & Lesion Localization</div>', unsafe_allow_html=True)

loc1, loc2 = st.columns(2, gap="medium")

with loc1:
    st.markdown('<div class="img-card"><div class="img-title">🫁 Lung Segmentation (U-Net)</div></div>', unsafe_allow_html=True)
    st.image(st.session_state["segmented_image"], use_container_width=True)

with loc2:
    st.markdown('<div class="img-card"><div class="img-title">🎯 Lesion Localization (YOLOv8)</div></div>', unsafe_allow_html=True)
    st.image(st.session_state["detected_image"], use_container_width=True)
    if not st.session_state.get("lesion_found", True):
        st.info("No localized lesions detected by YOLOv8.")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 8. WHO-STYLE CLINICAL RECOMMENDATION & REPORT
# -------------------------------------------------

rec_color = "#EF4444" if tb_probability >= 70 else "#FACC15" if tb_probability >= 40 else "#4ADE80"
rec_summary = (
    "The AI screening indicates a high likelihood of pulmonary tuberculosis. Immediate microbiological evaluation is advised."
    if tb_probability >= 50
    else "No definitive radiographic hallmarks of active pulmonary tuberculosis detected. Clinical correlation advised."
)

st.markdown(
    f"""
<div style="background:linear-gradient(135deg,#0A1836,#0A1020);
border-left:6px solid {rec_color};
border-radius:18px;
padding:22px;
margin-bottom:20px;
box-shadow:0 8px 25px rgba(0,0,0,.35);">

<h3 style="margin:0;color:white;font-size:1.25rem;">
Clinical Recommendation (WHO/NTEP Protocol)
</h3>

<p style="color:#BFD7FF;margin-top:8px;font-size:14px;line-height:1.5;">
{rec_summary}
</p>

<hr style="border-color:rgba(255,255,255,.08);margin:12px 0;">

<div style="color:white;font-weight:700;font-size:13px;">
Recommended Diagnostic Protocol:
</div>

<ul style="color:#D6E6FF;line-height:1.8;margin-top:8px;font-size:13px;">
<li>Immediate physician consultation & symptom review</li>
<li>Confirmatory GeneXpert / Sputum Smear Microscopy</li>
<li>Follow National Tuberculosis Elimination Program (NTEP) guidelines</li>
</ul>

</div>
""",
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