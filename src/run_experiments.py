from pathlib import Path
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, silhouette_score
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "tracks_clean.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

FEATURES = [
    "popularity", "duration_ms", "explicit", "danceability",
    "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "time_signature"
]
TARGET = "track_genre"


def main():
    df = pd.read_csv(DATA)

    # Use the same balanced sample and split for comparable models.
    df = (
        df.groupby(TARGET, group_keys=False)
        .sample(n=100, random_state=42)
        .reset_index(drop=True)
    )
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    metrics = []
    predictions_by_model = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1,
        class_weight="balanced_subsample"
    )
    print("Training Random Forest...", flush=True)
    start = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - start
    pred = rf.predict(X_test)
    predictions_by_model["Random Forest"] = pred
    metrics.append({
        "model": "Random Forest",
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "training_seconds": rf_time
    })
    joblib.dump(rf, MODELS / "random_forest.joblib")

    importance = pd.Series(
        rf.feature_importances_, index=FEATURES
    ).sort_values()
    ax = importance.plot(kind="barh", figsize=(9, 6))
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=150)
    plt.close()

    # One-hidden-layer neural network
    nn = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=35,
            early_stopping=True, random_state=42
        )
    )
    print("Training neural network...", flush=True)
    start = time.perf_counter()
    nn.fit(X_train, y_train)
    nn_time = time.perf_counter() - start
    pred = nn.predict(X_test)
    predictions_by_model["Neural Network"] = pred
    metrics.append({
        "model": "Neural Network",
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "training_seconds": nn_time
    })
    joblib.dump(nn, MODELS / "neural_network.joblib")

    # Unsupervised segmentation experiment using K-means.
    print("Running clustering/segmentation...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    start = time.perf_counter()
    clusters = kmeans.fit_predict(X_scaled)
    cluster_time = time.perf_counter() - start
    silhouette = silhouette_score(
        X_scaled, clusters, sample_size=min(1500, len(X_scaled)),
        random_state=42
    )
    joblib.dump(
        {"scaler": scaler, "kmeans": kmeans},
        MODELS / "kmeans_segmentation.joblib"
    )
    pd.DataFrame({
        "genre": y,
        "cluster": clusters
    }).to_csv(RESULTS / "cluster_assignments.csv", index=False)

    cluster_summary = pd.crosstab(
        pd.Series(clusters, name="cluster"),
        pd.Series(y, name="genre")
    )
    cluster_summary.to_csv(RESULTS / "cluster_genre_counts.csv")

    # Save the RF confusion matrix for the 12 most frequent test genres.
    labels = sorted(y_test.unique())[:12]
    cm = confusion_matrix(y_test, predictions_by_model["Random Forest"],
                          labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title("Random Forest Confusion Matrix (12 genres)")
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrix_rf.png", dpi=150)
    plt.close()

    # Include the previously completed SVM metrics, if present.
    svm_file = RESULTS / "baseline_svm_metrics.json"
    if svm_file.exists():
        with svm_file.open(encoding="utf-8") as f:
            old = json.load(f)
        metrics.insert(0, {
            "model": "RBF SVM",
            "accuracy": old["accuracy"],
            "macro_f1": old["macro_f1"],
            "training_seconds": old["training_seconds"]
        })

    comparison = pd.DataFrame(metrics)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    ax = comparison.set_index("model")[["accuracy", "macro_f1"]].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1)
    )
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=150)
    plt.close()

    summary = {
        "dataset": "Spotify track genre dataset",
        "task": "Genre classification; not original playlist-label classification",
        "sample_size": len(df),
        "genres": int(y.nunique()),
        "features": FEATURES,
        "random_state": 42,
        "models": metrics,
        "kmeans_clusters": 8,
        "kmeans_silhouette_score": silhouette,
        "kmeans_fit_seconds": cluster_time,
        "note": "Silhouette measures cluster separation, not classification accuracy."
    }
    with (RESULTS / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MODEL COMPARISON ===")
    print(comparison.round(4).to_string(index=False))
    print(f"\nK-means silhouette score: {silhouette:.4f}")
    print("\nSaved outputs in:", RESULTS)
    print("Models saved in:", MODELS)


if __name__ == "__main__":
    main()
