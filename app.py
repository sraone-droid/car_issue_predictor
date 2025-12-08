import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Issue Predictor", layout="centered")

# ---------------- BACKGROUND GIF ----------------
bg_style = """
<style>
.stApp {
    background-image: url('https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(bg_style, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center; color:white;'>🚗 Car Issue Predictor</h1>",
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
except:
    st.error("❌ model.pkl or vectorizer.pkl missing!")
    st.stop()

# ---------------- INPUT ----------------
complaint = st.text_area(
    "Describe your car complaint:",
    height=140,
    placeholder="Example: Engine overheats while driving..."
)

# ---------------- PREDICT ----------------
if st.button("🔍 Predict"):
    if not complaint or len(complaint.strip().split()) < 2:
        st.warning("⚠️ Please type more details.")
    else:
        x = vectorizer.transform([complaint])

        # If model supports probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            idx = np.argmax(probs)
            pred = model.classes_[idx]
            conf = probs[idx]

            st.success(f"🚀 Predicted Issue: **{pred}** (Confidence: {conf:.2f})")

        else:
            pred = model.predict(x)[0]
            st.success(f"🚀 Predicted Issue: **{pred}**")

# ---------------- FOOTER ----------------
st.markdown(
    "<br><center>🔧 Powered by Machine Learning</center>",
    unsafe_allow_html=True
)
