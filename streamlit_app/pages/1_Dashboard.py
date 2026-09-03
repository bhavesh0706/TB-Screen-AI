from PIL import Image
import streamlit as st

from components.classifier import predict_tb
from components.detector import detect_lesions
from components.gradcam import generate_gradcam
from components.recommendation import get_patient_recommendation
from components.report_generator import create_report
from components.segmenter import segment_lungs

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PulmoTB AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
defaults = {
    "uploaded_image": None,
    "classification": None,
    "segmented_image": None,
    "gradcam_image": None,
    "detected_image": None,
    "lesion_found": False,
    "detections": [],
    "lesion_count": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# GLOBAL CSS — compact RetinexAI theme
# --------------------------------------------------
# NOTE: every HTML string in this file is either one line or flush-left
# with zero leading whitespace per line. Indented multi-line HTML inside
# st.markdown() gets misread by Streamlit's markdown parser as a code
# block (4+ leading spaces = code block), which is why raw tags were
# printing as visible text in the previous version.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(circle at top left,#1489E7 0%,#0B4D99 22%,#031C4C 55%,#010817 100%);color:white;}
.block-container{max-width:1520px;padding-top:0.6rem;padding-bottom:1rem;padding-left:2rem;padding-right:2rem;}
header[data-testid="stHeader"]{background:transparent;height:2.2rem;}
#MainMenu,footer,[data-testid="stSidebar"]{display:none !important;}
.hero-card{background:linear-gradient(135deg,#08132D,#0F2C63,#2554CC);border:1px solid rgba(72,138,255,.22);border-radius:16px;padding:12px 18px;box-shadow:0 10px 24px rgba(0,0,0,.35);margin-bottom:10px;}
.hero-title{font-size:22px;font-weight:800;color:white;margin:0;}
.hero-sub{color:#BFD8FF;font-size:12px;margin-top:2px;}
.status-pill{padding:5px 12px;border-radius:999px;background:rgba(74,222,128,.14);border:1px solid rgba(74,222,128,.4);color:#4ADE80;font-size:11px;font-weight:700;white-space:nowrap;}
.section-title-bar{background:rgba(4,14,34,.78);border:1px solid rgba(72,138,255,.14);border-radius:12px;padding:8px 16px;margin:12px 0 8px;}
.section-title{font-size:0.95rem;font-weight:800;letter-spacing:.01em;}
.report-card{background:rgba(7,19,40,.84);border:1px solid rgba(59,130,246,.18);border-radius:12px;padding:12px 14px;}
.card-label{color:#8EA8CF;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;}
.card-value-lg{font-size:20px;font-weight:800;color:white;margin-top:4px;}
.card-value-md{font-size:16px;font-weight:800;color:white;margin-top:3px;}
.card-note{color:#9FB5D9;font-size:11px;margin-top:3px;}
.risk-pill{display:inline-block;padding:4px 12px;border-radius:999px;font-size:11.5px;font-weight:700;margin-top:4px;}
.prob-row{display:flex;align-items:center;gap:8px;margin-top:6px;}
.prob-label{width:78px;font-size:11px;color:#BFD8FF;flex-shrink:0;}
.prob-track{flex:1;height:10px;border-radius:6px;background:#0E1D3D;overflow:hidden;}
.prob-fill{height:100%;border-radius:6px;}
.prob-pct{width:44px;text-align:right;font-size:11px;font-weight:700;color:white;flex-shrink:0;}
.subcard{flex:1;background:#08152B;border:1px solid rgba(59,130,246,.16);border-radius:10px;padding:8px 10px;}
.subcard-label{color:#8EA8CF;font-size:10px;}
.subcard-value{color:white;font-size:14px;font-weight:800;margin-top:2px;}
.step-row{display:flex;gap:10px;align-items:center;padding:7px 4px;border-bottom:1px solid rgba(90,150,255,.10);}
.step-row:last-child{border-bottom:none;}
.step-icon{width:24px;height:24px;min-width:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;}
.step-text{color:white;font-size:12px;font-weight:600;line-height:1.3;}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"]>div{background:#08162F !important;color:white !important;border:1px solid rgba(59,130,246,.20) !important;border-radius:10px !important;height:2.2rem !important;}
.stNumberInput div[data-baseweb="input"]{background:#08162F !important;border-radius:10px !important;}
.stNumberInput button{background:#08162F !important;color:white !important;border:none !important;}
.stSelectbox div[data-baseweb="select"] *{background:#08162F !important;color:white !important;}
.stSelectbox svg{fill:#BFD8FF !important;}
label{font-size:12px !important;}
[data-testid="stFileUploaderDropzone"]{background:#0B1835;border:2px dashed rgba(59,130,246,.28);border-radius:12px;min-height:90px;padding:6px;}
[data-testid="stFileUploader"] button{background:#08162F !important;color:white !important;border-radius:8px !important;border:1px solid rgba(59,130,246,.25) !important;padding:2px 10px !important;}
[data-testid="stFileUploader"] section{padding:6px !important;}
[data-testid="stFileUploader"] small{font-size:10px !important;}
[data-testid="stImage"] img{border-radius:10px !important;max-height:180px !important;width:auto !important;max-width:100% !important;object-fit:contain !important;background:#090D18;margin:0 auto;display:block;}
.img-label{font-size:12px;font-weight:700;color:white;margin-bottom:6px;}
[data-testid="stExpander"]{background:rgba(7,19,40,.6);border:1px solid rgba(59,130,246,.15);border-radius:10px;}
.stDownloadButton button{width:100%;height:42px;border:none;border-radius:10px;font-weight:700;font-size:13px;color:white;background:linear-gradient(90deg,#2563EB,#3B82F6);box-shadow:0 8px 18px rgba(37,99,235,.3);}
div[data-testid="stVerticalBlock"] > div{gap:0.35rem;}
hr{margin:0.4rem 0;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown('<div class="hero-card"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;"><div><div class="hero-title">🫁 PulmoTB AI</div><div class="hero-sub">AI-Assisted Pulmonary Tuberculosis Screening Platform</div></div><div class="status-pill">● System Ready</div></div></div>', unsafe_allow_html=True)

# --------------------------------------------------
# PATIENT INFORMATION
# --------------------------------------------------
st.markdown('<div class="section-title-bar"><div class="section-title">Patient Information</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    patient_id = st.text_input("Patient ID", placeholder="0001", max_chars=4)
    if patient_id and not patient_id.isdigit():
        st.error("Patient ID must contain only 4 digits.")
        st.stop()
with c2:
    patient_name = st.text_input("Patient Name", placeholder="Enter name")
with c3:
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=30)
with c4:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# --------------------------------------------------
# CHEST X-RAY ACQUISITION
# --------------------------------------------------
st.markdown('<div class="section-title-bar"><div class="section-title">Chest X-ray Acquisition</div></div>', unsafe_allow_html=True)

up_col, info_col = st.columns([1.4, 3], gap="medium")
with up_col:
    uploaded_file = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
with info_col:
    if uploaded_file:
        st.markdown('<div style="margin-top:8px;background:linear-gradient(90deg,#063A2C,#0B6B4D);border:1px solid #19C37D;border-radius:8px;padding:6px 10px;color:#72F0B2;font-size:11px;font-weight:600;display:inline-block;">✓ Uploaded successfully</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-top:8px;color:#8EA8CF;font-size:11px;">PNG · JPG · JPEG, min 512×512 — upload to run the full screening pipeline.</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

# --------------------------------------------------
# AI PIPELINE EXECUTION
# --------------------------------------------------
with st.spinner("Running PulmoTB AI pipeline..."):
    classification = predict_tb(image)
    segmented_img = segment_lungs(image)

    try:
        gradcam_img = generate_gradcam(image, lung_mask=segmented_img)
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
st.session_state["segmented_image"] = segmented_img
st.session_state["gradcam_image"] = gradcam_img
st.session_state["detected_image"] = detected_img
st.session_state["lesion_found"] = lesion_found
st.session_state["detections"] = detections
st.session_state["lesion_count"] = len(detections)

prediction = classification["label"]
confidence = classification["confidence"] * 100
tb_probability = classification["tb_probability"] * 100
normal_probability = 100 - tb_probability

rec = get_patient_recommendation(tb_probability=tb_probability, age=patient_age, gender=patient_gender)
prediction_label = "TB Positive" if prediction == "Tuberculosis" else "TB Negative"

# --------------------------------------------------
# IMAGING OVERVIEW — all 4 visuals in a single row
# --------------------------------------------------
st.markdown('<div class="section-title-bar"><div class="section-title">Imaging Overview</div></div>', unsafe_allow_html=True)

img1, img2, img3, img4 = st.columns(4, gap="small")
with img1:
    st.markdown('<div class="img-label">🩻 Original</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
with img2:
    st.markdown('<div class="img-label">🔥 Grad-CAM</div>', unsafe_allow_html=True)
    st.image(gradcam_img, use_container_width=True)
with img3:
    st.markdown('<div class="img-label">🫁 Segmentation</div>', unsafe_allow_html=True)
    st.image(segmented_img, use_container_width=True)
with img4:
    st.markdown('<div class="img-label">🎯 Lesion Localization</div>', unsafe_allow_html=True)
    st.image(detected_img, use_container_width=True)

lesion_status = "Detected" if lesion_found else "None"
lesion_color = "#EF4444" if lesion_found else "#4ADE80"
highest = max((d["confidence"] for d in detections), default=0) * 100
st.markdown(f'<div style="margin-top:8px;font-size:11.5px;color:#BFD8FF;">Regions: <b style="color:white;">{len(detections)}</b> &nbsp;·&nbsp; Status: <b style="color:{lesion_color};">{lesion_status}</b> &nbsp;·&nbsp; Highest: <b style="color:white;">{highest:.1f}%</b></div>', unsafe_allow_html=True)

if detections:
    with st.expander("View detected region coordinates"):
        for region in detections:
            x1, y1, x2, y2 = region["bbox"]
            st.markdown(f'<div style="font-size:12px;color:#BFD8FF;padding:3px 0;">Region #{region["id"]} — bbox ({x1}, {y1}) → ({x2}, {y2}) — confidence {region["confidence"]*100:.1f}%</div>', unsafe_allow_html=True)

# --------------------------------------------------
# AI SCREENING REPORT (single consolidated block)
# --------------------------------------------------
st.markdown('<div class="section-title-bar"><div class="section-title">AI Screening Report</div></div>', unsafe_allow_html=True)

rep_left, rep_right = st.columns([1, 1.1], gap="medium")

with rep_left:
    st.markdown(f'<div class="report-card"><div class="card-label">Predicted Condition</div><div class="card-value-lg">{prediction_label}</div><div class="card-note">AI-assisted screening output</div></div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f'<div class="report-card" style="text-align:center;margin-top:8px;"><div class="card-label">Confidence</div><div class="card-value-md">{confidence:.1f}%</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="report-card" style="text-align:center;margin-top:8px;"><div class="card-label">TB Probability</div><div class="card-value-md">{tb_probability:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="report-card" style="margin-top:8px;"><div class="card-label">Class Probabilities</div><div class="prob-row"><div class="prob-label">Tuberculosis</div><div class="prob-track"><div class="prob-fill" style="width:{tb_probability:.1f}%;background:#3B82F6;"></div></div><div class="prob-pct">{tb_probability:.1f}%</div></div><div class="prob-row"><div class="prob-label">Normal</div><div class="prob-track"><div class="prob-fill" style="width:{normal_probability:.1f}%;background:#64748B;"></div></div><div class="prob-pct">{normal_probability:.1f}%</div></div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="report-card" style="margin-top:8px;"><div class="card-label">Risk Assessment</div><div class="risk-pill" style="background:{rec["color"]}22;border:1px solid {rec["color"]};color:{rec["color"]};">{rec["risk"]}</div><div style="margin-top:8px;color:#D6E6FF;font-size:12px;line-height:1.5;">{rec["summary"]}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="report-card" style="margin-top:8px;"><div class="card-label" style="margin-bottom:6px;">Patient Category</div><div style="display:flex;gap:10px;"><div class="subcard"><div class="subcard-label">Age Group</div><div class="subcard-value">{rec["age_group"]}</div></div><div class="subcard"><div class="subcard-label">Gender</div><div class="subcard-value">{rec["gender"]}</div></div></div></div>', unsafe_allow_html=True)

with rep_right:
    st.markdown('<div style="font-size:14px;font-weight:800;color:white;margin-bottom:6px;">Recommended Next Steps</div>', unsafe_allow_html=True)

    icon_map = [
        ("🚑", "#EF4444"), ("🧪", "#8B5CF6"), ("📄", "#2563EB"),
        ("🩺", "#06B6D4"), ("🫁", "#EAB308"), ("🏥", "#10B981"), ("👨‍⚕️", "#3B82F6"),
    ]
    rows_html = ""
    for i, step in enumerate(rec["steps"]):
        icon, accent = icon_map[i % len(icon_map)]
        rows_html += f'<div class="step-row"><div class="step-icon" style="background:{accent}22;border:1px solid {accent};">{icon}</div><div class="step-text">{step}</div></div>'
    st.markdown(f'<div class="report-card" style="padding:4px 12px;">{rows_html}</div>', unsafe_allow_html=True)

# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------
st.markdown('<div class="section-title-bar"><div class="section-title">Clinical Report</div></div>', unsafe_allow_html=True)

pdf_bytes = create_report(st.session_state)

st.download_button(
    "⬇ Download Official Clinical Screening Report (PDF)",
    data=pdf_bytes,
    file_name=f"PulmoTB_AI_Report_{patient_id or '0001'}.pdf",
    mime="application/pdf",
    use_container_width=True,
)
