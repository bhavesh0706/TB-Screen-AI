import streamlit as st

st.set_page_config(
    page_title="TB-Screen AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit's multipage navigation and default menu
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

[data-testid="stSidebarNav"] {
    display: none;
}

section[data-testid="stSidebar"] {
    display: none;
}

.block-container{
    padding-top:0rem;
    padding-bottom:0rem;
    max-width:100%;
}
</style>
""", unsafe_allow_html=True)

# Open the only page in the application
st.switch_page("pages/1_Dashboard.py")