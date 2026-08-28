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
        # A handful of artist entries have a null name (a real, confirmed
        # Spotify data gap) - filtered out here so join() doesn't crash
        # trying to join a None, and so primary_artist below picks the
        # first real name rather than a null one.
        artist_names = [a["name"] for a in track["artists"] if a.get("name")]
        conn.execute(
            """
            INSERT INTO songs (track_id, name, artist, primary_artist, album, release_date, duration_ms, isrc, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
                name = excluded.name,
                artist = excluded.artist,
                primary_artist = excluded.primary_artist,
                album = excluded.album,
                release_date = excluded.release_date,
                duration_ms = excluded.duration_ms,
                isrc = excluded.isrc,
                fetched_at = excluded.fetched_at
            """,
            (
                track["id"],
                track["name"],
                ", ".join(artist_names),
                # Taken directly from Spotify's own artist list, not by
                # splitting the joined "artist" string above on "," - that
                # split silently breaks for any artist whose own name
                # contains a comma (confirmed: "Tyler, The Creator" and
                # "Earth, Wind & Fire" both do), since there's no way to
                # tell "a comma separating two artists" apart from "a comma
                # inside one artist's name" once they're already joined
                # into a single string.
                artist_names[0] if artist_names else None,
                track["album"]["name"],
                track["album"].get("release_date"),
                track["duration_ms"],
                track.get("external_ids", {}).get("isrc"),
                now,
            ),
        )
        # ON CONFLICT DO UPDATE (not INSERT OR REPLACE): if this track_id
        # already exists, update only these specific metadata columns -
        # confirmed via testing that INSERT OR REPLACE deletes the whole
        # existing row and reinserts it, silently wiping every column not
        # named in the statement (audio_features, lyrics, description,
        # embedding, genre_bucket) back to NULL, even for tracks that
        # hadn't changed at all. This preserves all that computed work
        # when the playlist is re-fetched later - only genuinely new
        # tracks end up with NULL downstream fields, which every other
        # pipeline stage's "only process what's still NULL" logic already
        # picks up correctly on its own.
        count += 1
    # SQLite doesn't persist changes to disk until commit() is called.
    conn.commit()
    conn.close()
    return count


def fetch_and_store(playlist_id: str = TARGET_PLAYLIST_ID) -> int:
    items = fetch_playlist_items(playlist_id)
    return save_tracks(items)
