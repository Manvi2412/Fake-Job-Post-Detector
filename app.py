import streamlit as st
import joblib
import os

# -------------------------------------------------------
# 🧩 Page Setup
# -------------------------------------------------------
st.set_page_config(
    page_title="Fake Job Post Detector",
    page_icon="🕵️‍♀️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# 🧠 Load Model and Vectorizer
# -------------------------------------------------------
st.write("📁 Current working directory:", os.getcwd())
st.write("📂 Files found:", os.listdir())

try:
    model_path = os.path.join(os.getcwd(), "fake_job_post_detector.pkl")
    vectorizer_path = os.path.join(os.getcwd(), "tfidf_vectorizer.pkl")

    st.write("🔍 Loading model from:", model_path)
    st.write("🔍 Loading vectorizer from:", vectorizer_path)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    st.success("✅ Model and vectorizer loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Failed to load model/vectorizer: {e}")
    st.stop()

# -------------------------------------------------------
# 🎨 Styling (modern dark UI)
# -------------------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #2b1055, #7597de);
        color: white;
    }
    .main {
        background-color: rgba(20, 20, 20, 0.85);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .stTextArea textarea {
        background-color: #222;
        color: white;
        border-radius: 10px;
        border: 1px solid #555;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a60c3, #5f3ea1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🕵️‍♀️ App UI
# -------------------------------------------------------
st.title("🕵️‍♀️ Fake Job Post Detector")
st.markdown("### Detect whether a job posting is **Real** or **Fake** using Machine Learning")

# -------------------------------------------------------
# ✍️ Input Section
# -------------------------------------------------------
user_input = st.text_area("Paste the job description below:", height=200, placeholder="Enter job description here...")

# -------------------------------------------------------
# 🔍 Prediction Logic (with Confidence)
# -------------------------------------------------------
if st.button("Analyze Job Post"):
    if user_input.strip() == "":
        st.warning("Please enter a job description first.")
    else:
        try:
            features = vectorizer.transform([user_input])
            prediction = model.predict(features)[0]
            
            # Confidence Score (probability)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                fake_prob = float(probabilities[1]) * 100  # probability of fake
                real_prob = float(probabilities[0]) * 100
            else:
                # fallback if model doesn't support predict_proba
                fake_prob = 50.0
                real_prob = 50.0

            st.subheader("Prediction Result:")
            if prediction == 1:
                st.error("🚨 This job post seems **FAKE**!")
                st.write(f"Confidence: **{fake_prob:.2f}% Fake**, {real_prob:.2f}% Real**")
                st.progress(int(fake_prob))
            else:
                st.success("✅ This job post seems **REAL**!")
                st.write(f"Confidence: **{real_prob:.2f}% Real**, {fake_prob:.2f}% Fake**")
                st.progress(int(real_prob))

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# -------------------------------------------------------
# 💬 Footer
# -------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Machine Learning.")
