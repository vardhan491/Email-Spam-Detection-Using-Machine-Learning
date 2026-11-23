from pathlib import Path
import argparse
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"

def main():
    parser = argparse.ArgumentParser(description="Classify an email as spam or ham.")
    parser.add_argument("email", nargs="+", help="Email text to classify")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python train_model.py")

    model = joblib.load(MODEL_PATH)
    text = " ".join(args.email)
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities) * 100

    print(f"Prediction: {'SPAM' if prediction == 'spam' else 'NOT SPAM'}")
    print(f"Model confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
