from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")
st.title("📧 Email Spam Detection")
st.write("Enter an email message to check whether it looks like spam.")

if not MODEL_PATH.exists():
    st.warning("The trained model is not available yet. Add your dataset and run `python train_model.py`.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

email_text = st.text_area("Email content", height=180, placeholder="Paste the email text here...")

if st.button("Check email", type="primary"):
    if not email_text.strip():
        st.error("Please enter some email text.")
    else:
        model = load_model()
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        confidence = max(probabilities) * 100

        if prediction == "spam":
            st.error("⚠️ This email is classified as SPAM.")
        else:
            st.success("✅ This email is classified as NOT SPAM.")
        st.caption(f"Model confidence: {confidence:.2f}%")
        st.info("This is a machine-learning prediction, not a guarantee. Review suspicious emails carefully.")
