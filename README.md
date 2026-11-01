# Verity

Verity is a Python app for screening news text. It checks for potentially misleading language patterns and can optionally learn from a labeled news dataset. Its results are clues to investigate, not fact-checking verdicts.

## Features

- Analyze a headline, claim, or article in the Streamlit web interface.
- See a screening score and the language signals behind it.
- Train a TF-IDF and logistic regression classifier on labeled articles.
- Review held-out accuracy and F1 after training.
- Review and export analysis history for the current session.
- Save a trained model locally as `model.joblib`.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown in the terminal, usually `http://localhost:8501`.

## Train with a Kaggle dataset

In the **Train model** page, upload a labeled CSV with:

- An article text column named `text`, `content`, `article`, `news`, or `title`.
- A label column named `label`, `labels`, `class`, or `target`.

The app also recognizes the common split format `datasets/fake.csv` and `datasets/true.csv` (or `datasets/real.csv`). It assigns labels from the filenames and combines title and article text when both are available.

Supported labels include `fake`/`real`, `false`/`true`, and `0`/`1`. Numeric meanings vary by dataset, so check the Kaggle dataset description before training.

## How detection works

Before you train a model, Verity uses transparent language-pattern checks such as sensational wording, heavy capitalization, repeated exclamation marks, and pressure to share. After training, it uses TF-IDF to convert text into word and phrase features, then logistic regression to learn patterns from the labeled examples.

The training process holds out part of the dataset to report accuracy and F1. Dataset quality affects these scores, and results may not generalize to new sources or topics. A screening score is not a verified probability that a story is true or false.

## Project files

- `app.py` - Streamlit interface and navigation.
- `detector.py` - Screening rules, dataset preparation, model training, and predictions.
- `requirements.txt` - Python dependencies.
- `datasets/` - Optional location for Kaggle CSV files.
- `model.joblib` - Created locally after model training.
