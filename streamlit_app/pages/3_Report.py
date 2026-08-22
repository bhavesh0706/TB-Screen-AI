import streamlit as st
from components.report_generator import create_report

st.set_page_config(page_title="TB Screening Report", layout="wide")

st.title("Patient Screening Report")
st.caption("AI-assisted tuberculosis screening summary")

# ---------------- Check Session ----------------

required = [
    "classification",
    "uploaded_image",
    "segmented_image",
    "gradcam_image",
    "detected_image",
]

missing = [k for k in required if k not in st.session_state]

if missing:
    st.warning("Please complete Screening first.")
    if st.button("Go to Screening"):
        st.switch_page("pages/2_Screening.py")
    st.stop()

# ---------------- Load Data ----------------

classification = st.session_state["classification"]
original = st.session_state["uploaded_image"]
mask = st.session_state["segmented_image"]
gradcam = st.session_state["gradcam_image"]
yolo = st.session_state["detected_image"]

# ---------------- Summary ----------------

risk = (
    "High Risk (TB Detected)"
    if classification["label"] == "Tuberculosis"
    else "Low Risk (Normal)"
)

st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Prediction", classification["label"])

with c2:
    st.metric("Confidence", f"{classification['confidence']*100:.2f}%")

with c3:
    st.metric("TB Probability", f"{classification['tb_probability']*100:.2f}%")

st.markdown(f"### {risk}")

st.markdown("---")

# ---------------- Images ----------------

row1col1, row1col2 = st.columns(2)

with row1col1:
    st.subheader("Original X-ray")
    st.image(original, use_container_width=True)

with row1col2:
    st.subheader("Grad-CAM")
    st.image(gradcam, use_container_width=True)

row2col1, row2col2 = st.columns(2)

with row2col1:
    st.subheader("Lung Segmentation")
    st.image(mask, use_container_width=True)

with row2col2:
    st.subheader("YOLO Lesion Detection")
    st.image(yolo, use_container_width=True)

st.markdown("---")

# ---------------- Clinical Interpretation ----------------

st.subheader("Clinical Interpretation")

if classification["label"] == "Tuberculosis":
    st.error("DenseNet121 detected a tuberculosis-like pattern.")
else:
    st.success("DenseNet121 detected a normal chest X-ray pattern.")

st.markdown(f"""
| Component | Result |
|---|---|
| DenseNet121 | **{classification['label']}** |
| Confidence | **{classification['confidence']*100:.2f}%** |
| TB Probability | **{classification['tb_probability']*100:.2f}%** |
| U-Net | Lung region extracted |
| Grad-CAM | Attention heatmap generated |
| YOLOv8 | Lesion localization completed |
""")

st.markdown("---")

# ---------------- PDF ----------------

pdf_bytes = create_report(st.session_state)

st.download_button(
    "Download Patient Report (PDF)",
    pdf_bytes,
    file_name="TB_Screening_Report.pdf",
    mime="application/pdf",
    use_container_width=True,
)

st.markdown("---")

if st.button("⬅ Back to Screening", use_container_width=True):
    st.switch_page("pages/2_Screening.py")