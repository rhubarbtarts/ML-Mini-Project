from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "tracks_clean.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

FEATURES = [
    "popularity", "duration_ms", "explicit", "danceability",
    "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "time_signature",
]
TARGET = "track_genre"


def main():
    df = pd.read_csv(DATA_FILE)

    # Balanced sample: at most 100 tracks per genre.
    df = (
        df.groupby(TARGET, group_keys=False)
        .sample(n=100, random_state=42)
        .reset_index(drop=True)
    )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale")
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")
    print(f"Genres: {y.nunique()}")
    print("Training RBF SVM...")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "baseline_svm.joblib")

    metrics = {
        "model": "RBF SVM",
        "dataset": "Spotify track genre dataset",
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "number_of_genres": int(y.nunique()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "training_seconds": training_seconds,
        "random_state": 42,
    }

    with (RESULTS_DIR / "baseline_svm_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Model saved: {MODEL_DIR / 'baseline_svm.joblib'}")
    print(f"Metrics saved: {RESULTS_DIR / 'baseline_svm_metrics.json'}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
