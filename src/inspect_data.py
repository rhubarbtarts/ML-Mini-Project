import json
from pathlib import Path

path = Path(
    r"C:\Users\Ananya V\Documents\GitHub\playlistr-ml-v1"
    r"\get_toy_set\results\toy_data_set_playlists_full.json"
)

with path.open("r", encoding="utf-8") as f:
    playlists = json.load(f)

all_ids = []
first_track = None

for playlist in playlists:
    tracks = playlist.get("tracks", [])
    for item in tracks:
        track = item.get("track", item) if isinstance(item, dict) else {}
        if not first_track and track:
            first_track = track
        track_id = track.get("id")
        if track_id:
            all_ids.append(track_id)

print("Playlist count:", len(playlists))
print("Total track entries:", len(all_ids))
print("Unique track IDs:", len(set(all_ids)))
print("Track IDs appearing more than once:", len(all_ids) - len(set(all_ids)))

if first_track:
    print("\nExample song:", first_track.get("name"))
    print("Song fields:", list(first_track.keys()))
    print("Audio features included:", any(
        key in first_track for key in [
            "danceability", "energy", "valence",
            "acousticness", "instrumentalness"
        ]
    ))
else:
    print("No track records found in the expected format.")
