from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
REAL_DATA_ZIP = BASE_DIR.parent / "archive.zip"
REAL_DATA_CSV = BASE_DIR.parent / "spam.csv"
RANDOM_STATE = 42


def load_real_dataset() -> pd.DataFrame:
    if REAL_DATA_ZIP.exists():
        with zipfile.ZipFile(REAL_DATA_ZIP) as archive:
            csv_name = next((name for name in archive.namelist() if name.endswith(".csv")), None)
            if csv_name is None:
                raise FileNotFoundError(f"No CSV found inside {REAL_DATA_ZIP}")
            with archive.open(csv_name) as file_handle:
                df = pd.read_csv(file_handle, encoding="latin-1")
    elif REAL_DATA_CSV.exists():
        df = pd.read_csv(REAL_DATA_CSV, encoding="latin-1")
    else:
        raise FileNotFoundError(
            f"Could not find {REAL_DATA_ZIP} or {REAL_DATA_CSV}. "
            "Copy the real spam dataset into the parent folder first."
        )

    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df = df.drop_duplicates().reset_index(drop=True)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                stop_words="english",
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 5),
                                min_df=2,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )


def main() -> None:
    df = load_real_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label_num"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Rows: {len(df)}")
    print(f"Test split accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, target_names=["ham", "spam"]))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
