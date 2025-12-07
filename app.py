import streamlit as st
import joblib
from PIL import Image
import time

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Car Issue Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- CUSTOM BG ------------------------
background_img = "background.jpg"  # replace your image
if background_img:
    bg_url = f"""
    <style>
    .stApp {{
        background-image: url('https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_url, unsafe_allow_html=True)

# ---------------------- HEADER ------------------------
st.markdown(
    "<h1 style='text-align:center; color:white;'>CAR ISSUE PREDICTOR </h1>",
    unsafe_allow_html=True
)

# ---------------------- LOAD MODEL ----------------------
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    st.error("❌ Model or Vectorizer missing!")
    st.stop()

# ---------------------- USER INPUT ----------------------
st.write("")
complaint = st.text_area(
    "Describe your car complaint:",
    height=120,
    placeholder="Example: Strange noise from engine..."
)

# Hide extra empty box
st.markdown("<style> .css-1es6m3g {display:none;}</style>", unsafe_allow_html=True)

# ---------------------- PREDICT ----------------------
if st.button("🔍 Predict Issue"):
    if complaint.strip() == "":
        st.warning("⚠️ Please type something!")
    else:
        # Spinner (spanner) animation
        with st.spinner("🔧 Fixing... please wait"):
            time.sleep(2)  # for effect
            vect_text = vectorizer.transform([complaint])
            prediction = model.predict(vect_text)[0]

        st.success(f"🚀 PREDICTED ISSUE: **{prediction}**")

        # Static spanner icon
        st.markdown("<center><h2>🛠️</h2></center>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("<br><br><center>🔧 Powered by Machine Learning</center>", unsafe_allow_html=True)
