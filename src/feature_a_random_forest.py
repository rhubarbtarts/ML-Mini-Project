
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from data import load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42


def main():
    # Load the same dataset and split used by our other models.
    X_train, X_test, y_train, y_test, feature_names, genre_names = load_data(SEED)

    # Best hyperparameters from Experiment 2.
    best_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    }

    # Define the Random Forest.
    model = RandomForestClassifier(
        **best_params,
        random_state=SEED,
        n_jobs=-1
    )

    # Five-fold stratified cross-validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")
    print(f"Number of genres: {len(genre_names)}")
    print("Starting 5-fold cross-validation...")

    start = time.perf_counter()

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    print(f"Cross-validation scores: {cv_scores}")

    # Train the final model on the complete training set.
    print("\nTraining final Random Forest model...")
    model.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start

    # Predictions on training and test data.
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate evaluation metrics.
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0
    )

    mean_cv_accuracy = cv_scores.mean()

    # Create output directories.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Output file paths.
    model_path = MODEL_DIR / "feature_a_random_forest.joblib"
    metrics_path = RESULTS_DIR / "feature_a_random_forest_metrics.json"
    report_path = RESULTS_DIR / "feature_a_random_forest_report.txt"
    cv_path = RESULTS_DIR / "feature_a_random_forest_cv_results.csv"
    importance_path = RESULTS_DIR / "feature_a_feature_importance.csv"

    # Save trained model.
    joblib.dump(model, model_path)

    # Save metrics.
    metrics = {
        "model": "Random Forest",
        "feature": "Feature A",
        "target": "track_genre",
        "sample_size": len(X_train) + len(X_test),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "number_of_genres": len(genre_names),
        "train_accuracy": float(train_accuracy),
        "accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
        "best_cv_accuracy": float(mean_cv_accuracy),
        "cv_fold_scores": [float(score) for score in cv_scores],
        "best_params": best_params,
        "training_seconds": training_seconds,
        "random_state": SEED,
        "features": feature_names
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save cross-validation results.
    cv_results = pd.DataFrame({
        "fold": range(1, len(cv_scores) + 1),
        "accuracy": cv_scores
    })
    cv_results.to_csv(cv_path, index=False)

    # Save classification report.
    report = classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )

    report_path.write_text(report, encoding="utf-8")

    # Calculate and save feature importance.
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(
        by="importance",
        ascending=False
    )

    importance.to_csv(importance_path, index=False)

    # Display results.
    print("\n--- RANDOM FOREST RESULTS ---")
    print(f"Best parameters: {best_params}")
    print(f"Mean 5-fold CV accuracy: {mean_cv_accuracy:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Training time: {training_seconds:.2f}s")

    print("\n--- FEATURE IMPORTANCE ---")
    print(importance.to_string(index=False))

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved CV results: {cv_path}")
    print(f"Saved feature importance: {importance_path}")


if __name__ == "__main__":
    main()
