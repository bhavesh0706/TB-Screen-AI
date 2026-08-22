import streamlit as st

st.set_page_config(
    page_title="TB-Screen AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{
    padding-top:1.2rem;
    padding-bottom:2rem;
    max-width:1400px;
}

section[data-testid="stSidebar"]{
    background:#161B22;
}

.sidebar-title{
    font-size:26px;
    font-weight:700;
    color:white;
}

.sidebar-sub{
    color:#8B949E;
    font-size:13px;
}

.main-title{
    font-size:36px;
    font-weight:700;
    color:white;
}

.subtitle{
    color:#8B949E;
    font-size:18px;
}

.card{
    background:#161B22;
    border:1px solid #30363D;
    border-radius:18px;
    padding:18px;
    margin-bottom:18px;
}

.metric-card{
    background:#161B22;
    border:1px solid #30363D;
    border-radius:18px;
    padding:18px;
    text-align:center;
}

.image-card{
    background:#161B22;
    border:1px solid #30363D;
    border-radius:18px;
    padding:16px;
}

div.stButton>button{
    width:100%;
    border-radius:12px;
    height:48px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🫁 TB-Screen AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-sub">AI-Powered Tuberculosis Screening Platform</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("""
**AI Pipeline**

- DenseNet121
- U-Net
- Grad-CAM
- YOLOv8
""")

    st.divider()

    st.markdown("""
**Project Status**

✅ Classification

✅ Segmentation

✅ Explainability

✅ Lesion Detection

✅ PDF Report
""")

# ---------- Home ----------

st.markdown('<div class="main-title">TB-Screen AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Intelligent Chest X-ray Screening System for Early Tuberculosis Detection</div>',
    unsafe_allow_html=True
)

st.divider()

col1,col2,col3,col4=st.columns(4)

with col1:
    st.markdown("""
<div class="metric-card">
<h3>DenseNet121</h3>
Classification
</div>
""",unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="metric-card">
<h3>U-Net</h3>
Lung Segmentation
</div>
""",unsafe_allow_html=True)

with col3:
    st.markdown("""
<div class="metric-card">
<h3>Grad-CAM</h3>
Explainability
</div>
""",unsafe_allow_html=True)

with col4:
    st.markdown("""
<div class="metric-card">
<h3>YOLOv8</h3>
Lesion Detection
</div>
""",unsafe_allow_html=True)

st.divider()

st.markdown("""
<div class="card">

### Workflow

1. Upload a Chest X-ray.
2. DenseNet121 predicts Tuberculosis risk.
3. U-Net isolates the lung region.
4. Grad-CAM explains the prediction.
5. YOLOv8 localizes suspicious lesions.
6. Generate a professional PDF report.

</div>
""",unsafe_allow_html=True)