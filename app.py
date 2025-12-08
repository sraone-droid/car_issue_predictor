# app.py
import streamlit as st
import joblib
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Car Issue Predictor", layout="centered")

# -------------------- BACKGROUND ---------------------
bg_url = """
<style>
.stApp {
    background-image: url("https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(bg_url, unsafe_allow_html=True)

# -------------------- HEADER -------------------------
st.markdown(
    "<h1 style='text-align:center; color:white; text-shadow:2px 2px 5px black;'>🚗 Car Issue Predictor</h1>",
    unsafe_allow_html=True
)

# -------------------- LOAD MODEL ---------------------
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
except:
    st.error("❌ Model or vectorizer not found! Upload model.pkl & vectorizer.pkl.")
    st.stop()

# -------------------- INPUT --------------------------
complaint = st.text_area(
    "Describe your car complaint:",
    height=140,
    placeholder="Example: Strange noise coming from engine..."
)

THRESHOLD = 0.5  # Adjust between 0.4–0.7

# -------------------- PREDICT ------------------------
if st.button("🚀 Predict"):
    if not complaint or len(complaint.strip().split()) < 2:
        st.warning("⚠️ Please write a longer complaint (2+ words).")
    else:
        x = vectorizer.transform([complaint])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            idx = np.argmax(probs)
            pred = model.classes_[idx]
            conf = probs[idx]

            if conf < THRESHOLD:
                st.warning("❓ Unable to confidently predict. Please describe the issue more clearly.")
            else:
                st.success(f"🔧 Predicted Problem: **{pred}**")
                st.info(f"📊 Confidence: **{conf:.2f}**")
        else:
            pred = model.predict(x)[0]
            st.success(f"🔧 Predicted Problem: **{pred}**")

# -------------------- FOOTER -------------------------
st.markdown(
    "<br><center style='color:white;'>🛠️ Powered by Machine Learning</center>",
    unsafe_allow_html=True
)
