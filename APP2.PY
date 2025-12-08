import streamlit as st
import time
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Car Issue Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- CUSTOM BG ------------------------
background_img = "background.jpg"
if background_img:
    bg_url = """
    <style>
    .stApp {
        background-image: url('https://cdn.dribbble.com/userupload/22797976/file/original-3b362f19987e09fbeb2b092dc029db17.gif');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """
    st.markdown(bg_url, unsafe_allow_html=True)

# ---------------------- HEADER ------------------------
st.markdown(
    "<h1 style='text-align:center; color:white;'>CAR ISSUE PREDICTOR</h1>",
    unsafe_allow_html=True
)

# ---------------------- EMBEDDING MODEL ----------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Known issue examples
issue_examples = {
    "Weak Battery": ["car not starting", "slow crank", "battery weak"],
    "Radiator Problem": ["engine overheating", "coolant leak", "radiator issue"],
    "Brake Issue": ["brake noise", "brakes not working", "vibration when braking"],
    "Unbalanced Wheels": ["steering vibration", "wobbling", "wheel shaking"],
    "Faulty Spark Plug": ["engine misfire", "jerking", "spark plug issue"]
}

# Pre-calc embeddings
issue_embeddings = {
    label: embedder.encode(examples) for label, examples in issue_examples.items()
}

# ---------------------- USER INPUT ----------------------
complaint = st.text_area(
    "Describe your car complaint:",
    height=120,
    placeholder="Example: Strange noise from engine..."
)

# Hide empty box
st.markdown("<style> .css-1es6m3g {display:none;}</style>", unsafe_allow_html=True)

# ---------------------- PREDICT FUNCTION ----------------------
def predict_issue(text):
    if len(text.strip()) < 4:
        return "⚠️ Unable to predict (input too short)"

    input_emb = embedder.encode(text)
    best_label = None
    best_score = 0

    for label, example_embs in issue_embeddings.items():
        similarity = util.cos_sim(torch.tensor(input_emb), torch.tensor(example_embs))
        score = similarity.max().item()

        if score > best_score:
            best_score = score
            best_label = label

    if best_score < 0.45:  # confidence threshold
        return "❓ Unknown / Cannot Predict"

    return best_label

# ---------------------- BUTTON ----------------------
if st.button("🔍 Predict Issue"):
    if complaint.strip() == "":
        st.warning("⚠️ Please type something!")
    else:
        with st.spinner("🔧 Checking..."):
            time.sleep(2)
            result = predict_issue(complaint)

        st.success(f"🚀 PREDICTED ISSUE: **{result}**")
        st.markdown("<center><h2>🛠️</h2></center>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("<br><br><center>🔧 Powered by Machine Learning</center>", unsafe_allow_html=True)
