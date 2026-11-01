"""Text screening and trainable fake-news classification utilities."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_PATH = Path(__file__).parent / "model.joblib"

MISLEADING_TERMS = {
    "shocking", "secret", "exposed", "miracle", "urgent", "unbelievable",
    "they don't want you to know", "they don't want you to see", "share before",
    "100% proven", "wake up", "must read", "breaking!!!",
}


@dataclass
class DetectionResult:
    label: str
    score: float
    method: str
    signals: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def heuristic_screen(text: str) -> DetectionResult:
    """Return a transparent language-pattern screen, not a truth verdict."""
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    signals: list[str] = []
    score = 0.0

    found_terms = [term for term in MISLEADING_TERMS if term in lowered]
    if found_terms:
        signals.append("Sensational or urgency-focused wording detected")
        score += min(0.35, 0.12 + 0.06 * (len(found_terms) - 1))

    words = re.findall(r"\b[A-Z]{3,}\b", normalized)
    letters = re.findall(r"[A-Za-z]", normalized)
    if len(words) >= 2 or (len(letters) >= 30 and sum(char.isupper() for char in letters) / len(letters) > 0.3):
        signals.append("Unusually heavy capitalization")
        score += 0.2

    if normalized.count("!") >= 2:
        signals.append("Repeated exclamation marks")
        score += 0.15

    if re.search(r"\b(share this|forward this|before it is deleted|before they remove)\b", lowered):
        signals.append("Pressure to share or act immediately")
        score += 0.2

    has_attribution = bool(re.search(r"https?://|\b(according to|reported by|said in a|study published)\b", lowered))
    if len(normalized.split()) > 35 and not has_attribution:
        signals.append("No obvious source attribution in the submitted text")
        score += 0.1

    score = min(score, 0.95)
    label = "Review recommended" if score >= 0.3 else "Few warning signals"
    explanation = (
        "These language patterns are screening clues only. Check the original source and independent reporting."
    )
    if not signals:
        signals.append("No listed wording patterns detected")

    return DetectionResult(label, score, "Language-pattern screening", signals, explanation)


def normalize_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label in {"fake", "false", "misleading", "0", "unreliable"}:
        return "misleading"
    if label in {"real", "true", "reliable", "1", "credible"}:
        return "credible"
    raise ValueError(f"Unsupported label: {value!r}. Use fake/misleading or real/credible.")


def train_model(data: pd.DataFrame) -> dict[str, Any]:
    """Train a text model from a CSV-like DataFrame with text and label columns."""
    required = {"text", "label"}
    columns = {str(column).strip().lower(): column for column in data.columns}
    text_column = next((columns[name] for name in ("text", "content", "article", "news", "title") if name in columns), None)
    label_column = next((columns[name] for name in ("label", "labels", "class", "target") if name in columns), None)
    if text_column is None or label_column is None:
        raise ValueError("Dataset needs a text/article/title column and a label/class column.")

    selected_columns = [text_column]
    title_column = columns.get("title")
    if title_column is not None and title_column != text_column:
        selected_columns.append(title_column)
    cleaned = data.loc[:, selected_columns + [label_column]].dropna(subset=[text_column, label_column]).copy()
    cleaned["text"] = cleaned[selected_columns].astype(str).agg(" ".join, axis=1).str.strip()
    cleaned = cleaned[cleaned["text"].str.len() >= 20]
    try:
        cleaned["label"] = cleaned[label_column].map(normalize_label)
    except ValueError as error:
        raise ValueError(f"{error} Labels 0/FAKE and 1/REAL are supported for common Kaggle datasets.") from error

    if cleaned["label"].nunique() != 2:
        raise ValueError("Training data must include both credible and misleading examples.")
    if len(cleaned) < 20:
        raise ValueError("At least 20 usable examples are required; 100+ per class is recommended.")
    if cleaned["label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two examples for a stratified evaluation split.")

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned["text"], cleaned["label"], test_size=0.2, random_state=42, stratify=cleaned["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=1, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    joblib.dump(pipeline, MODEL_PATH)

    return {
        "examples": len(cleaned),
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, pos_label="misleading", zero_division=0),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }


def load_split_dataset(directory: Path) -> pd.DataFrame:
    """Load common Kaggle datasets split into fake.csv and true/real.csv."""
    files = {path.stem.strip().lower(): path for path in directory.glob("*.csv")}
    fake_path = files.get("fake")
    real_path = files.get("true") or files.get("real")
    if fake_path is None or real_path is None:
        raise ValueError("Expected both fake.csv and true.csv (or real.csv) in the datasets folder.")

    frames = []
    for path, label in ((fake_path, "fake"), (real_path, "real")):
        frame = pd.read_csv(path, low_memory=False)
        frame["label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_model() -> Pipeline | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, EOFError, ValueError, TypeError):
        return None


def analyze(text: str, model: Pipeline | None = None) -> DetectionResult:
    cleaned = " ".join(text.split())
    if len(cleaned) < 20:
        raise ValueError("Enter at least 20 characters so there is enough text to screen.")

    if model is None:
        return heuristic_screen(cleaned)

    probabilities = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    misleading_score = float(probabilities[classes.index("misleading")])
    label = "Higher model score for misleading" if misleading_score >= 0.5 else "Higher model score for credible"
    return DetectionResult(
        label=label,
        score=misleading_score,
        method="TF-IDF + logistic regression",
        signals=["Prediction based on patterns learned from your uploaded labeled dataset"],
        explanation="The score is a model estimate, not a verified probability that the claim is true or false.",
    )
