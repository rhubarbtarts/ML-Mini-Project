
from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def read_json(filename):
    path = RESULTS_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # Load existing experiment metrics.
    svm = read_json("baseline_svm_metrics.json")
    nn = read_json("baseline_nn_metrics.json")
    rf = read_json("feature_a_random_forest_metrics.json")
    clustering = read_json("feature_b_metrics.json")

    # Build supervised model comparison.
    comparison = pd.DataFrame([
        {
            "model": "RBF SVM",
            "accuracy": svm["accuracy"],
            "macro_f1": svm["macro_f1"],
            "training_seconds": svm["training_seconds"]
        },
        {
            "model": "Neural Network",
            "accuracy": nn["test_accuracy"],
            "macro_f1": nn["macro_f1"],
            "training_seconds": nn["training_seconds"]
        },
        {
            "model": "Random Forest (Feature A)",
            "accuracy": rf["accuracy"],
            "macro_f1": rf["macro_f1"],
            "training_seconds": rf["training_seconds"]
        }
    ])

    comparison = comparison.sort_values(
        by="accuracy",
        ascending=False
    )

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    # Generate model comparison chart.
    ax = comparison.set_index("model")[
        ["accuracy", "macro_f1"]
    ].plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1)
    )

    ax.set_title("Supervised Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "model_comparison.png",
        dpi=300
    )
    plt.close()

    # Save combined experiment summary.
    summary = {
        "dataset": "Spotify track genre dataset",
        "sample_size": svm["sample_size"],
        "number_of_genres": svm["number_of_genres"],
        "features": svm["features"],
        "random_state": 42,
        "classification_models": comparison.to_dict(
            orient="records"
        ),
        "best_classification_model": comparison.iloc[0]["model"],
        "best_classification_accuracy": float(
            comparison.iloc[0]["accuracy"]
        ),
        "feature_a": {
            "best_parameters": rf["best_params"],
            "cross_validation_accuracy": rf["best_cv_accuracy"],
            "training_accuracy": rf["train_accuracy"],
            "test_accuracy": rf["accuracy"],
            "macro_f1": rf["macro_f1"]
        },
        "feature_b": {
            "best_number_of_clusters": clustering[
                "best_number_of_clusters"
            ],
            "silhouette_score": clustering[
                "best_silhouette_score"
            ],
            "tracks_clustered": clustering["tracks_clustered"],
            "genre_labels_used_for_clustering": clustering[
                "genre_labels_used_for_clustering"
            ]
        },
        "note": (
            "Classification accuracy and clustering silhouette "
            "scores measure different objectives and should not "
            "be compared directly."
        )
    }

    with (
        RESULTS_DIR / "experiment_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n--- MODEL COMPARISON ---")
    print(comparison.round(4).to_string(index=False))

    print("\n--- FEATURE B ---")
    print(
        "Best clusters:",
        clustering["best_number_of_clusters"]
    )
    print(
        "Silhouette score:",
        round(clustering["best_silhouette_score"], 4)
    )

    print("\nGenerated:")
    print("results/model_comparison.csv")
    print("results/model_comparison.png")
    print("results/experiment_summary.json")


if __name__ == "__main__":
    main()
