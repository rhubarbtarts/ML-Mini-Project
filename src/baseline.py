from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

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
SEED = 42


def main():
    df = pd.read_csv(DATA_FILE)

    # Use exactly 100 tracks per genre for a balanced experiment.
    df = (
        df.groupby(TARGET, group_keys=False)
        .sample(n=100, random_state=SEED)
        .reset_index(drop=True)
    )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # Scaling is fitted separately within each CV training fold.
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf"),
    )

    param_grid = {
        "svc__C": [1, 10],
        "svc__gamma": ["scale", 0.01],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print(f"Total sampled tracks: {len(df):,}")
    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")
    print(f"Number of genres: {y.nunique()}")
    print("Starting five-fold SVM hyperparameter tuning...")

    start = time.perf_counter()
    search.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    # GridSearchCV refits the best model on all training data.
    model = search.best_estimator_
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "baseline_svm.joblib"
    metrics_path = RESULTS_DIR / "baseline_svm_metrics.json"
    report_path = RESULTS_DIR / "baseline_svm_report.txt"
    cv_path = RESULTS_DIR / "baseline_svm_cv_results.csv"

    joblib.dump(model, model_path)

    metrics = {
        "model": "Tuned RBF SVM",
        "target": TARGET,
        "dataset": "Spotify track genre dataset",
        "sample_size": len(df),
        "tracks_per_genre": 100,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "number_of_genres": int(y.nunique()),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "best_cv_accuracy": float(search.best_score_),
        "best_params": search.best_params_,
        "training_seconds": training_seconds,
        "random_state": SEED,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(search.cv_results_).to_csv(
        cv_path, index=False
    )

    report = classification_report(
        y_test, predictions, zero_division=0
    )
    report_path.write_text(report, encoding="utf-8")

    print("\n--- SVM RESULTS ---")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best 5-fold CV accuracy: {search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Training and tuning time: {training_seconds:.2f}s")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved CV results: {cv_path}")
    print(f"Saved classification report: {report_path}")


if __name__ == "__main__":
    main()
