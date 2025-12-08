# app.py
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Car Issue Predictor", layout="centered")

st.markdown("<h1 style='text-align:center;'>🚗 Car Issue Predictor</h1>", unsafe_allow_html=True)

# load trained objects (must be present in repo)
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("Model/vectorizer not found or failed to load. Run training locally and upload model.pkl & vectorizer.pkl.")
    st.stop()

complaint = st.text_area("Describe your car complaint:", height=140)

THRESHOLD = 0.5  # adjust (0.4..0.7) depending on desired strictness

if st.button("Predict"):
    if not complaint or len(complaint.strip().split()) < 2:
        st.warning("Please write a longer complaint (2+ words).")
    else:
        x = vectorizer.transform([complaint])
        # use predict_proba if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            idx = np.argmax(probs)
            pred = model.classes_[idx]
            conf = probs[idx]
            if conf < THRESHOLD:
                st.warning("❓ Unable to confidently predict. Please describe the issue more clearly.")
            else:
                st.success(f"🚀 Predicted Problem: **{pred}** (Confidence: {conf:.2f})")
        else:
            # fallback if model doesn't support probs
            pred = model.predict(x)[0]
            st.success(f"🚀 Predicted Problem: **{pred}**")
