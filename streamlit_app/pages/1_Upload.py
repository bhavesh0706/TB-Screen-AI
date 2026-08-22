import streamlit as st
from PIL import Image

st.markdown("""
<style>

.upload-card{
    background:#161B22;
    border:2px dashed #3B82F6;
    border-radius:20px;
    padding:35px;
    text-align:center;
    margin-bottom:20px;
}

.preview-card{
    background:#161B22;
    border:1px solid #30363D;
    border-radius:18px;
    padding:18px;
}

.info-card{
    background:#161B22;
    border:1px solid #30363D;
    border-radius:18px;
    padding:20px;
}

.big-title{
    font-size:30px;
    font-weight:700;
}

.small-text{
    color:#8B949E;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📤 Upload Chest X-ray</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="small-text">Upload a frontal PA chest X-ray for AI-powered tuberculosis screening.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown("""
<div class="upload-card">

## Drag & Drop Chest X-ray

Supported formats:

**PNG · JPG · JPEG**

Recommended resolution: **512×512 or higher**

</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "",
    type=["png","jpg","jpeg"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.session_state["uploaded_image"] = image

    st.success("Chest X-ray uploaded successfully.")

    left,right = st.columns([1.1,0.9])

    with left:

        st.markdown("""
<div class="preview-card">

### X-ray Preview

</div>
""", unsafe_allow_html=True)

        st.image(image, use_container_width=True)

    with right:

        st.markdown("""
<div class="info-card">

### Image Information

</div>
""", unsafe_allow_html=True)

        width,height = image.size

        st.metric("Width", width)

        st.metric("Height", height)

        st.metric("Mode", image.mode)

        st.metric("Aspect Ratio", f"{width/height:.2f}")

        st.markdown("---")

        st.markdown("### Ready for Analysis")

        st.markdown(
            "The uploaded X-ray is stored securely for AI processing."
        )

        if st.button(
            "Start AI Screening →",
            type="primary",
            use_container_width=True
        ):
            st.switch_page("pages/2_Screening.py")

else:

    st.info("Upload a chest X-ray to begin screening.")