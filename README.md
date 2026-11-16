# Spotify Music Genre Classification and Playlist Segmentation

## 1. Project Overview

This project explores machine learning techniques for analysing Spotify tracks using their audio features.

The project has two main objectives:

1. **Genre Classification:** Predict the genre of a track using supervised machine learning.
2. **Playlist Segmentation:** Group tracks into clusters based on similarities in their audio characteristics using unsupervised learning.

The project also compares multiple classification models to evaluate their performance.

## 2. Dataset

The project uses a Spotify track genre dataset.

A balanced sample is created using 100 tracks per genre, resulting in:

- Total tracks: 11,400
- Number of genres: 114
- Training samples: 9,120
- Testing samples: 2,280

The dataset is split using an 80:20 stratified train-test split with random state 42.

### Features used

The following 15 audio and track-related features are used:

- Popularity
- Duration
- Explicit
- Danceability
- Energy
- Key
- Loudness
- Mode
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Time signature

The target variable for classification is `track_genre`.

## 3. Baseline Models

Two supervised learning models were implemented as baselines.

### 3.1 Support Vector Machine (SVM)

- Algorithm: RBF Kernel SVM
- Hyperparameter tuning using GridSearchCV
- Feature scaling using StandardScaler
- Evaluation using accuracy and macro F1-score

Best parameters:
- C = 10
- Gamma = 0.01

Results:
- Test accuracy: 21.80%
- Macro F1-score: 19.38%
- Cross-validation accuracy: 20.08%

### 3.2 Neural Network

A feed-forward neural network was implemented using PyTorch.

Architecture:
- Input layer
- Hidden layer with 128 neurons
- Sigmoid activation
- Output layer with 114 classes
- LogSoftmax output

Training configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Negative Log Likelihood Loss
- Epochs: 500

Results:
- Training accuracy: 21.58%
- Test accuracy: 20.79%
- Macro F1-score: 17.23%

## 4. Feature A – Random Forest Classification

Feature A investigates whether a Random Forest classifier can improve genre prediction performance.

### Methodology

A Random Forest classifier was trained using the same balanced dataset and train-test split as the baseline models.

Hyperparameter tuning was performed through multiple experiments using five-fold stratified cross-validation.

The final selected configuration was:

| Parameter | Value |
|---|---|
| Number of estimators | 200 |
| Maximum depth | 20 |
| Minimum samples split | 10 |
| Minimum samples leaf | 2 |
| Maximum features | sqrt |

### Final results

- Cross-validation accuracy: 27.05%
- Training accuracy: 94.90%
- Test accuracy: 29.56%
- Macro F1-score: 26.94%

### Feature Importance

The most influential features according to the trained Random Forest were:

1. Popularity
2. Acousticness
3. Duration
4. Danceability
5. Speechiness
6. Valence
7. Loudness
8. Energy

The feature importance results indicate that popularity and several audio characteristics contribute substantially to the model's predictions.

### Observation

The Random Forest achieved the highest test accuracy among the classification models.

However, the substantial difference between training accuracy (94.90%) and test accuracy (29.56%) indicates significant overfitting.

## 5. Feature B – Playlist Segmentation

Feature B explores unsupervised learning through K-Means clustering.

Unlike genre classification, clustering does not use genre labels to train the model. Instead, tracks are grouped according to similarities in their numerical audio features.

### Methodology

1. Load the balanced dataset.
2. Standardize the numerical features using StandardScaler.
3. Evaluate K-Means clustering for cluster counts from 2 to 10.
4. Calculate silhouette scores to evaluate cluster separation.
5. Select the number of clusters with the highest silhouette score.
6. Analyse cluster profiles and genre distributions.
7. Apply Principal Component Analysis (PCA) to visualize the clusters in two dimensions.

### Results

- Tracks clustered: 11,400
- Best number of clusters: 3
- Silhouette score: 0.1932

The project also generates:
- Cluster assignments
- Cluster profiles
- Genre distribution across clusters
- Silhouette score visualization
- PCA-based cluster visualization

### Observation

The clustering experiment identifies three groups based on audio-feature similarities.

The silhouette score suggests relatively limited separation between the clusters, indicating that the tracks do not form strongly distinct groups in the selected feature space.

## 6. Model Comparison

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| RBF SVM | 21.80% | 19.38% |
| Neural Network | 20.79% | 17.23% |
| Random Forest | **29.56%** | **26.94%** |

Random Forest achieved the strongest classification performance among the three supervised models.

K-Means is evaluated separately using the silhouette score because it is an unsupervised clustering algorithm.

## 7. Limitations

- The classification task involves 114 genres, making it a challenging multiclass prediction problem.
- Several genres have overlapping audio characteristics.
- The models use only 15 numerical features and do not incorporate audio waveforms or textual metadata.
- The Random Forest exhibits substantial overfitting.
- The K-Means silhouette score indicates limited cluster separation.
- The classification task predicts Spotify genre labels; it does not directly reproduce the original playlist-membership task described in the research paper.

## 8. Project Structure

```text
ML-Mini-Project/
│
├── data/
│   └── processed/
│       └── tracks_clean.csv
│
├── models/
│   ├── baseline_nn.pt
│   └── baseline_svm.joblib
│
├── results/
│   ├── baseline_nn_metrics.json
│   ├── baseline_svm_metrics.json
│   ├── feature_a_random_forest_metrics.json
│   ├── feature_a_random_forest_report.txt
│   ├── feature_a_feature_importance.csv
│   ├── feature_b_metrics.json
│   ├── feature_b_cluster_profiles.csv
│   ├── feature_b_cluster_assignments.csv
│   └── feature_b visualizations
│
├── src/
│   ├── data.py
│   ├── baseline.py
│   ├── baseline_nn.py
│   ├── feature_a_random_forest.py
│   ├── feature_b_segmentation.py
│   └── run_experiments.py
│
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Requirements

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- PyTorch
- joblib

### Install dependencies

```bash
pip install -r requirements.txt
```

Ensure the processed dataset is available at:

```text
data/processed/tracks_clean.csv
```

### Run experiments

Run the following scripts from the project root:

```bash
python src/baseline.py
python src/baseline_nn.py
python src/feature_a_random_forest.py
python src/feature_b_segmentation.py
```

Generated metrics, reports, models and visualizations are saved in the `results/` and `models/` directories.

## 10. Conclusion

The project demonstrates the application of supervised and unsupervised machine learning techniques to Spotify track data.

Among the classification approaches, Random Forest achieved the best test accuracy and macro F1-score. The feature importance analysis also provided insights into which track characteristics contributed most to genre prediction.

The K-Means experiment demonstrated how tracks can be segmented using audio characteristics without relying on genre labels.

Overall, the project highlights the potential and limitations of using numerical audio features for music genre classification and playlist segmentation.