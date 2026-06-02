from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def _resolve_dataset_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [base_dir / "spam.csv", base_dir / "dist" / "spam.csv"]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find spam.csv. Expected at "
        f"{candidates[0]} or {candidates[1]}"
    )


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset() -> pd.DataFrame:
    dataset_path = _resolve_dataset_path()
    dataframe = pd.read_csv(dataset_path, encoding="latin-1", usecols=["v1", "v2"])
    dataframe.columns = ["label", "message"]
    dataframe["message"] = dataframe["message"].astype(str)
    dataframe["label"] = dataframe["label"].map({"ham": 0, "spam": 1})
    return dataframe.dropna(subset=["label"])


def create_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    preprocessor=_clean_text,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(test_size: float = 0.2, random_state: int = 42) -> Tuple[Pipeline, float]:
    dataframe = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe["message"],
        dataframe["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe["label"],
    )

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return pipeline, accuracy
