from datetime import datetime, timezone

from auth.spotify_oauth import get_spotify_client
from db.database import get_connection
from db.models import init_db

TARGET_PLAYLIST_ID = "4Jlag9nPT6xEKjNa515hUB"  # "When"


def fetch_playlist_items(playlist_id: str) -> list[dict]:
    sp = get_spotify_client()
    # Spotify's Feb 2026 migration renamed this endpoint from /tracks to
    # /items - sp._get() is used (instead of a spotipy named helper)
    # because spotipy hasn't been updated for the rename yet.
    results = sp._get(f"playlists/{playlist_id}/items", limit=100)
    items = results["items"]
    # Spotify paginates results at 100 items per page; results["next"]
    # holds the URL for the next page (or falsy if there isn't one).
    # sp.next() just follows that URL - this loop runs until every page
    # has been collected. "When" has 653 tracks, so ~7 iterations.
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
        # The same Feb 2026 migration also renamed the "track" key inside
        # each item to "item". entry.get() (not entry["item"]) returns
        # None instead of crashing when it's missing entirely - which
        # happens for local files and tracks removed from Spotify's
        # catalog since being added to the playlist.
        track = entry.get("item")
        if track is None or track.get("id") is None:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO songs (track_id, name, artist, album, release_date, duration_ms, isrc, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track["id"],
                track["name"],
                # A handful of artist entries have a null name (a real,
                # confirmed Spotify data gap) - filtered out here so
                # join() doesn't crash trying to join a None.
                ", ".join(a["name"] for a in track["artists"] if a.get("name")),
                track["album"]["name"],
                track["album"].get("release_date"),
                track["duration_ms"],
                track.get("external_ids", {}).get("isrc"),
                now,
            ),
        )
        # OR REPLACE: if this track_id already exists, overwrite the row
        # instead of erroring - makes this function safely re-runnable.
        # The ?, ?, ?... placeholders (with values passed separately, not
        # string-formatted into the SQL) is parameterized SQL - the safe
        # way to insert data, avoiding SQL injection.
        count += 1
    # SQLite doesn't persist changes to disk until commit() is called.
    conn.commit()
    conn.close()
    return count


def fetch_and_store(playlist_id: str = TARGET_PLAYLIST_ID) -> int:
    items = fetch_playlist_items(playlist_id)
    return save_tracks(items)
