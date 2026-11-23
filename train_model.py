from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from text_preprocessing import tokenize

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "emails.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"

def find_columns(df):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    label_candidates = ["label", "category", "class", "target", "v1"]
    text_candidates = ["text", "message", "email", "email_text", "content", "v2"]
    label_col = next((normalized[c] for c in label_candidates if c in normalized), None)
    text_col = next((normalized[c] for c in text_candidates if c in normalized), None)
    if label_col is None or text_col is None:
        raise ValueError(
            "Could not identify columns. Use columns named 'label' and 'text' "
            "(or common alternatives such as Category/Message or v1/v2)."
        )
    return label_col, text_col

def normalize_label(value):
    value = str(value).strip().lower()
    if value in {"spam", "1", "junk"}:
        return "spam"
    if value in {"ham", "not spam", "not_spam", "0", "legitimate", "normal"}:
        return "ham"
    return None

def main():
    parser = argparse.ArgumentParser(description="Train the email spam classifier.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to CSV dataset")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Add your CSV file at data/emails.csv or pass --data path/to/file.csv."
        )

    df = pd.read_csv(data_path).dropna()
    label_col, text_col = find_columns(df)
    data = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df[label_col].map(normalize_label)
    }).dropna()
    data = data[data["text"].str.strip().ne("")].drop_duplicates()

    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both spam and ham examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.20,
        random_state=42, stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset rows used: {len(data)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
