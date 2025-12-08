import streamlit as st
import joblib
import numpy as np

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Car Issue Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- BACKGROUND ----------------------
bg = """
<style>
.stApp {
    background-image: url('https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(bg, unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown(
    "<h1 style='text-align:center; color:white;'>🚗 Car Issue Predictor</h1>",
    unsafe_allow_html=True
)

# ---------------------- LOAD MODEL ----------------------
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
except:
    st.error("❌ Model or vectorizer missing! Upload model.pkl & vectorizer.pkl.")
    st.stop()

# ---------------------- INPUT ----------------------
complaint = st.text_area(
    "Describe your car complaint:",
    height=130,
    placeholder="Example: Engine turns off suddenly while driving, no warning lights..."
)

THRESHOLD = 0.50  # adjust sensitivity

# ---------------------- PREDICT ----------------------
if st.button("Predict"):
    if not complaint or len(complaint.strip().split()) < 2:
        st.warning("⚠️ Please write a longer complaint (2+ words).")
    else:
        x = vectorizer.transform([complaint])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]

            # get top 3 predictions
            top3_idx = np.argsort(probs)[::-1][:3]

            st.subheader("🔮 Top Predictions")
            for i in top3_idx:
                st.write(f"➡️ **{model.classes_[i]}** — Confidence: `{probs[i]:.2f}`")

            # best prediction
            best = top3_idx[0]
            best_conf = probs[best]

            if best_conf < THRESHOLD:
                st.warning("❓ Unable to confidently predict. Please describe the issue more clearly.")
            else:
                st.success(
                    f"🚀 Predicted Problem: **{model.classes_[best]}** "
                    f"(Confidence: {best_conf:.2f})"
                )

        else:
            pred = model.predict(x)[0]
            st.success(f"🚀 Predicted Problem: **{pred}**")

# ---------------------- FOOTER ----------------------
st.markdown(
    "<br><center>🔧 Powered by Machine Learning</center>",
    unsafe_allow_html=True
)
