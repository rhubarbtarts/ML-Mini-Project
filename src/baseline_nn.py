from pathlib import Path
import json
import time

import joblib
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import load_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42
EPOCHS = 500
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)


class MusicClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, HIDDEN_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_SIZE, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


def main():
    X_train, X_test, y_train, y_test, feature_names, playlist_names = load_data(SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    model = MusicClassifier(
        n_features=X_train_tensor.shape[1],
        n_classes=len(encoder.classes_)
    )

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")
    print(f"Number of genres: {len(encoder.classes_)}")
    print(f"Starting neural network training for {EPOCHS} epochs...")

    start = time.perf_counter()

    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    training_seconds = time.perf_counter() - start

    model.eval()

    with torch.no_grad():
        train_predictions = model(X_train_tensor).argmax(dim=1).numpy()
        test_predictions = model(X_test_tensor).argmax(dim=1).numpy()

    train_accuracy = accuracy_score(y_train_encoded, train_predictions)
    test_accuracy = accuracy_score(y_test_encoded, test_predictions)
    macro_f1 = f1_score(
        y_test_encoded,
        test_predictions,
        average="macro",
        zero_division=0
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "baseline_nn.pt"
    metrics_path = RESULTS_DIR / "baseline_nn_metrics.json"
    report_path = RESULTS_DIR / "baseline_nn_report.txt"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, MODEL_DIR / "baseline_nn_scaler.joblib")
    joblib.dump(encoder, MODEL_DIR / "baseline_nn_encoder.joblib")

    metrics = {
        "model": "Neural Network",
        "target": "track_genre",
        "sample_size": len(X_train) + len(X_test),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "number_of_genres": len(encoder.classes_),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
        "training_seconds": training_seconds,
        "epochs": EPOCHS,
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_state": SEED
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        y_test_encoded,
        test_predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )

    report_path.write_text(report, encoding="utf-8")

    print("\n--- NEURAL NETWORK RESULTS ---")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Training time: {training_seconds:.2f}s")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
