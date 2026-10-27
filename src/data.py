from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "spotify_tracks.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

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


def prepare_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()

    # Convert Boolean explicit values to numeric values.
    df["explicit"] = df["explicit"].astype(int)
    
    # Keep only rows with complete, valid numeric features.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat invalid zero values as missing.
    df.loc[df["tempo"] <= 0, "tempo"] = float("nan")
    df.loc[df["duration_ms"] <= 0, "duration_ms"] = float("nan")

    # Drop rows with missing values in any feature.
    df = df.dropna(subset=FEATURES)


    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "tracks_clean.csv"
    df.to_csv(output, index=False)

    print(f"Rows prepared: {len(df):,}")
    print(f"Number of genres: {df[TARGET].nunique()}")
    print(f"Features: {len(FEATURES)}")
    print(f"Saved to: {output}")
    print("\nMost common genres:")
    print(df[TARGET].value_counts().head(10).to_string())


if __name__ == "__main__":
    prepare_data()
