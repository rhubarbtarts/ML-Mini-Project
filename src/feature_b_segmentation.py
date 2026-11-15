from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data import load_data

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def main():
    X_train, X_test, y_train, y_test, feature_names, genre_names = load_data(SEED)

    X = pd.concat([X_train, X_test]).sort_index()
    y = pd.concat([y_train, y_test]).sort_index()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cluster_results = []
    best_score = -1
    best_k = None

    for k in range(2, 11):
        start_time = time.time()

        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=3000,
            random_state=SEED
        )

        elapsed = time.time() - start_time

        cluster_results.append({
            "n_clusters": k,
            "silhouette_score": score,
            "inertia": model.inertia_,
            "time_seconds": elapsed
        })

        print(
            f"K={k} | Silhouette score={score:.4f} | "
            f"Time={elapsed:.2f}s"
        )

        if score > best_score:
            best_score = score
            best_k = k

    results_df = pd.DataFrame(cluster_results)
    results_df.to_csv(
        RESULTS_DIR / "feature_b_cluster_scores.csv",
        index=False
    )

    best_model = KMeans(
        n_clusters=best_k,
        random_state=SEED,
        n_init=10
    )

    labels = best_model.fit_predict(X_scaled)

    assignments = X.copy()
    assignments["track_genre"] = y
    assignments["cluster"] = labels
    assignments.to_csv(
        RESULTS_DIR / "feature_b_cluster_assignments.csv",
        index=False
    )

    profiles = X.copy()
    profiles["cluster"] = labels
    cluster_profiles = profiles.groupby("cluster")[feature_names].mean()
    cluster_profiles.to_csv(
        RESULTS_DIR / "feature_b_cluster_profiles.csv"
    )

    genre_distribution = pd.crosstab(
        assignments["cluster"],
        assignments["track_genre"]
    )

    genre_distribution.to_csv(
        RESULTS_DIR / "feature_b_genre_distribution.csv"
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        results_df["n_clusters"],
        results_df["silhouette_score"],
        marker="o"
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("K-Means Silhouette Scores")
    plt.xticks(range(2, 11))
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "feature_b_silhouette_scores.png",
        dpi=300
    )
    plt.close()

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    plot_data = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": labels
    })

    plt.figure(figsize=(9, 6))
    plt.scatter(
        plot_data["PC1"],
        plot_data["PC2"],
        c=plot_data["cluster"],
        cmap="tab10",
        s=8,
        alpha=0.6
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"K-Means Track Segments (K={best_k})")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "feature_b_cluster_visualization.png",
        dpi=300
    )
    plt.close()

    joblib.dump(
        best_model,
        MODELS_DIR / "feature_b_kmeans.joblib"
    )
    joblib.dump(
        scaler,
        MODELS_DIR / "feature_b_scaler.joblib"
    )

    summary = {
        "best_number_of_clusters": int(best_k),
        "best_silhouette_score": float(best_score),
        "tracks_clustered": int(len(X)),
        "features_used": feature_names,
        "genre_labels_used_for_clustering": False,
        "pca_explained_variance": [
            float(value) for value in pca.explained_variance_ratio_
        ]
    }

    with open(
        RESULTS_DIR / "feature_b_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nSegmentation complete.")
    print("Best number of clusters:", best_k)
    print("Best silhouette score:", round(best_score, 4))
    print("Tracks clustered:", len(X))
    print("Cluster sizes:")
    print(assignments["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()