
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = ROOT / "data" / "processed" / "tracks_clean.csv"

FEATURES = [
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

TARGET = "track_genre"



def load_data(seed=42):
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {PROCESSED_FILE}")

    df = pd.read_csv(PROCESSED_FILE)
    df = df.dropna(subset=FEATURES + [TARGET])

    df = (
        df.groupby(TARGET, group_keys=False)
        .sample(n=100, random_state=seed)
        .reset_index(drop=True)
    )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        FEATURES,
        sorted(y.unique().tolist())
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, playlist_names = load_data()

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Features:", len(feature_names))
    print("Genres:", len(playlist_names))
