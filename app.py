import streamlit as st
import joblib

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Car Issue Predictor",
    page_icon="🚗",
    layout="centered"
)

# -----------------------
# CSS FOR BACKGROUND IMAGE + GLASS EFFECT
# -----------------------
st.markdown("""
<style>
/* Transparent background image */
.stApp {
    background: 
        linear-gradient(rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.35)),
        url("https://i.pinimg.com/1200x/51/a4/82/51a4821b0ea7eb4dc6ae58a85a13839b.jpg");
    background-size: cover;
    background-position: center;
}

/* Title */
h1 {
    text-align: center;
    color: white;
    font-family: monospace;
    font-weight: 900;
    text-shadow: 0px 0px 10px black;
}

# /* Glassmorphism Card */
# .glass-box {
    # background: rgba(255, 255, 255, 0.18);
    # padding: 35px;
    # border-radius: 20px;
    # border: 1px solid rgba(255, 255, 255, 0.3);
    # box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
    # backdrop-filter: blur(10px);
# }

/* Textarea */
textarea {
    background: rgba(255, 255, 255, 0.18);
    padding: 35px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    border-radius: 12px;
    color: white;
    padding: 10px 25px;
    border: none;
    font-size: 18px;
    width: 100%;
    font-weight: bold;
    transition: 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 10px rgba(0,150,255,0.7);
}

/* Prediction box */
.pred-box {
    margin-top: 20px;
    background: rgba(0, 200, 255, 0.25);
    padding: 20px;
    border-radius: 15px;
    color: #e8f8ff;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    border: 1px solid rgba(0,200,255,0.5);
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# UI CONTENT
# -----------------------
st.markdown("<h1> CAR ISSUE PREDICTOR </h1>", unsafe_allow_html=True)

st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

user_input = st.text_area("Describe your car complaint:")

if st.button("Predict Issue"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]

        st.markdown(
            f"<div class='pred-box'>🚀 Predicted Problem:<br>{prediction}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
