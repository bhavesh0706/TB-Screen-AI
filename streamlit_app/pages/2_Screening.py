import streamlit as st
from components.classifier import predict_tb
from components.segmenter import segment_lungs
from components.gradcam import generate_gradcam
from components.detector import detect_lesions

st.markdown("""
<style>

.dashboard-card{
    background:#161B22;
    border:1px solid #30363D;
    border-radius:18px;
    padding:20px;
    margin-bottom:18px;
}

.image-card{
    background:#161B22;
    border:1px solid #30363D;
    border-radius:18px;
    padding:15px;
    height:100%;
}

.metric-box{
    background:#0D1117;
    border:1px solid #30363D;
    border-radius:14px;
    padding:18px;
    text-align:center;
}

.risk-high{
    color:#EF4444;
    font-weight:bold;
    font-size:20px;
}

.risk-low{
    color:#2EA043;
    font-weight:bold;
    font-size:20px;
}

.section-title{
    font-size:24px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">🔬 AI Tuberculosis Screening Dashboard</div>', unsafe_allow_html=True)
st.caption("DenseNet121 · U-Net · Grad-CAM · YOLOv8")

if "uploaded_image" not in st.session_state:
    st.warning("Upload a chest X-ray first.")
    st.stop()

image = st.session_state["uploaded_image"]

# ---------------- AI PIPELINE ----------------

with st.spinner("Running AI Pipeline..."):
    classification = predict_tb(image)
    segmented = segment_lungs(image)
    gradcam = generate_gradcam(image)
    detection = detect_lesions(image)

st.session_state["classification"] = classification
st.session_state["segmented_image"] = segmented
st.session_state["gradcam_image"] = gradcam
st.session_state["detected_image"] = detection

# ---------------- TOP SUMMARY ----------------

risk = "High Risk" if classification["label"] == "Tuberculosis" else "Low Risk"

st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

left,right = st.columns([3,1])

with left:
    st.markdown("## AI Screening Summary")

with right:
    if risk == "High Risk":
        st.markdown('<div class="risk-high">High Risk</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low">Low Risk</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

m1,m2,m3=st.columns(3)

with m1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Prediction", classification["label"])
    st.markdown("</div>", unsafe_allow_html=True)

with m2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Confidence", f"{classification['confidence']*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with m3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("TB Probability", f"{classification['tb_probability']*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- IMAGE GRID ----------------

row1col1,row1col2=st.columns(2)

with row1col1:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.subheader("Original X-ray")
    st.image(image,use_container_width=True)
    st.caption("Uploaded chest X-ray")
    st.markdown("</div>", unsafe_allow_html=True)

with row1col2:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.subheader("Grad-CAM")
    st.image(gradcam,use_container_width=True)
    st.caption("Model attention heatmap")
    st.markdown("</div>", unsafe_allow_html=True)

row2col1,row2col2=st.columns(2)

with row2col1:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.subheader("Lung Segmentation")
    st.image(segmented,use_container_width=True)
    st.caption("U-Net extracted lung region")
    st.markdown("</div>", unsafe_allow_html=True)

with row2col2:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.subheader("YOLO Lesion Detection")
    st.image(detection,use_container_width=True)
    st.caption("Localized suspicious lesion")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INTERPRETATION ----------------

st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

st.markdown("## Clinical Interpretation")

if classification["label"]=="Tuberculosis":
    st.error("DenseNet121 detected a strong tuberculosis-like pattern.")
else:
    st.success("DenseNet121 detected a normal chest X-ray pattern.")

st.markdown(f"""
**DenseNet121 Confidence**

- **Prediction:** {classification["label"]}
- **Confidence:** {classification["confidence"]*100:.2f}%
- **TB Probability:** {classification["tb_probability"]*100:.2f}%

**U-Net**

- Automatically isolated the lung region for focused analysis.

**Grad-CAM**

- Highlights the image regions that influenced the classification decision.

**YOLOv8**

- Localized suspicious lesions inside the lungs using bounding-box detection.
""")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

col1,col2=st.columns(2)

with col1:
    if st.button("⬅ Upload Another X-ray",use_container_width=True):
        st.switch_page("pages/1_Upload.py")

with col2:
    if st.button("Generate Report ➜",type="primary",use_container_width=True):
        st.switch_page("pages/3_Report.py")