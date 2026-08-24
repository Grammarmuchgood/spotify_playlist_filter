from datetime import datetime, timezone

from auth.spotify_oauth import get_spotify_client
from db.database import get_connection
from db.models import init_db

TARGET_PLAYLIST_ID = "4Jlag9nPT6xEKjNa515hUB"  # "When"


def fetch_playlist_items(playlist_id: str) -> list[dict]:
    sp = get_spotify_client()
    results = sp._get(f"playlists/{playlist_id}/items", limit=100)
    items = results["items"]
    while results["next"]:
        results = sp.next(results)
        items.extend(results["items"])
    return items


def save_tracks(items: list[dict]) -> int:
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for entry in items:
        track = entry.get("item")
        if track is None or track.get("id") is None:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO songs (track_id, name, artist, album, duration_ms, isrc, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track["id"],
                track["name"],
                ", ".join(a["name"] for a in track["artists"] if a.get("name")),
                track["album"]["name"],
                track["duration_ms"],
                track.get("external_ids", {}).get("isrc"),
                now,
            ),
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def fetch_and_store(playlist_id: str = TARGET_PLAYLIST_ID) -> int:
    items = fetch_playlist_items(playlist_id)
    return save_tracks(items)
