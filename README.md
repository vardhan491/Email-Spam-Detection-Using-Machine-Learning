# Email Spam Detection using Machine Learning

A beginner-friendly machine-learning project that classifies email or message text as **Spam** or **Not Spam (Ham)**.

## How it works

1. Load a labelled email dataset from a CSV file.
2. Clean the text: lowercase, remove common stopwords, tokenize and lemmatize.
3. Convert text into numerical features using **TF-IDF**.
4. Train a **Multinomial Naive Bayes** classifier.
5. Evaluate the model using accuracy, precision, recall, F1-score and a confusion matrix.
6. Enter a new email in the Streamlit app to get a prediction.

## Project structure

```text
email-spam-detection/
├── data/
│   └── emails.csv             # Add your dataset here (not included)
├── models/
│   └── spam_classifier.joblib # Generated after training (not committed)
├── app.py                     # Streamlit user interface
├── predict.py                 # Command-line prediction
├── text_preprocessing.py      # Text cleaning and tokenization
├── setup_nltk.py              # Downloads NLTK WordNet resources
├── train_model.py             # Training and evaluation
├── requirements.txt
└── README.md
```

## 1. Requirements

- Python 3.9 or newer
- pip

## 2. Install dependencies

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```bat
.venv\Scripts\activate
```

Install packages and NLTK data:

```bash
pip install -r requirements.txt
python setup_nltk.py
```

## 3. Add your dataset

Create `data/emails.csv`. The project accepts common column names such as:

- `label` and `text`
- `Category` and `Message`
- `v1` and `v2` (common in SMS spam datasets)

Example format:

| label | text |
|---|---|
| ham | Hi, are we meeting today? |
| spam | Congratulations! You have won a free prize. |

Use a properly labelled email dataset for your actual project evaluation. The example rows above only explain the format; they are not a training dataset. Do not report an accuracy score until you have trained and evaluated the model on your dataset.

## 4. Train and evaluate

Place the CSV at `data/emails.csv`, then run:

```bash
python train_model.py
```

Or specify another CSV path:

```bash
python train_model.py --data path/to/your_dataset.csv
```

The script prints accuracy, precision, recall, F1-score and the confusion matrix. It saves the trained model to `models/spam_classifier.joblib`.

## 5. Run the web app

```bash
streamlit run app.py
```

A browser tab will open. Paste an email and click **Check email**.

## 6. Predict from the terminal

```bash
python predict.py "Congratulations! You have won a free gift"
```

## Algorithms and concepts

- **TF-IDF (Term Frequency–Inverse Document Frequency):** represents words based on their importance in a document relative to the dataset.
- **Multinomial Naive Bayes:** a probabilistic classification algorithm commonly used for text classification.
- **Train-test split:** keeps a portion of the labelled data aside to evaluate performance on unseen examples.
- **Evaluation metrics:** accuracy, precision, recall, F1-score and confusion matrix.

## Important limitations

- A legitimate email can contain words often seen in spam, and a spam email can look convincing. The model can make mistakes.
- The confidence shown by the app is the model's predicted probability, not a guarantee that the prediction is correct.
- Model performance depends on dataset quality, class balance, preprocessing and the test split.
- This project is a learning/demo classifier, not a replacement for email-provider security filters.

## Future improvements

- Compare Naive Bayes with Logistic Regression, SVM and other classifiers.
- Add cross-validation and hyperparameter tuning.
- Improve handling of URLs, sender details, HTML and email headers.
- Deploy the app to a hosting platform.

## Author

Add your name, GitHub profile and project details here.
