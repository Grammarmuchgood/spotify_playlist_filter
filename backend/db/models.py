from __future__ import annotations

from db.database import get_connection

# One row per track. track_id is Spotify's own ID used directly as the
# primary key - no separate auto-increment ID needed, and it naturally
# prevents duplicate rows for the same song.
#
# audio_features / lyrics / description / embedding are all TEXT and
# start out NULL - reserved for the pipeline steps that fill them in
# later (Librosa features, fetched lyrics, the LLM-generated description,
# and its embedding vector). SQLite has no native JSON type, so structured
# data like audio_features gets stored as a JSON string in a TEXT column.
CREATE_SONGS_TABLE = """
CREATE TABLE IF NOT EXISTS songs (
    track_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    artist TEXT NOT NULL,
    primary_artist TEXT,
    album TEXT,
    release_date TEXT,
    duration_ms INTEGER,
    isrc TEXT,
    audio_features TEXT,
    lyrics TEXT,
    description TEXT,
    embedding TEXT,
    genre_bucket TEXT,
    fetched_at TEXT NOT NULL
)
"""

# One row, in each user's own database file - account/processing state
# for that user, separate from their song data. Lives alongside `songs`
# in the same per-user file rather than a shared "users" table, so a
# user's whole world (their own songs AND their own account state) stays
# in one portable file - the same reason per-user files were chosen for
# `songs` at all (see db.database.get_connection). Doesn't exist in the
# legacy single-user database, since there's exactly one user there and
# nothing to track per-user.
CREATE_USER_META_TABLE = """
CREATE TABLE IF NOT EXISTS user_meta (
    spotify_user_id TEXT PRIMARY KEY,
    display_name TEXT,
    selected_playlist_id TEXT,
    selected_playlist_name TEXT,
    song_count INTEGER,
    processing_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    last_active_at TEXT
)
"""


def init_db(user_id: str | None = None) -> None:
    conn = get_connection(user_id)
    # IF NOT EXISTS makes this safe to call every time (it's called at
    # the top of fetch_playlist.py's flow) - does nothing if the table's
    # already there. Note: this does NOT alter an existing table if the
    # schema changes later - that requires rebuilding the .db file.
    conn.execute(CREATE_SONGS_TABLE)
    if user_id is not None:
        conn.execute(CREATE_USER_META_TABLE)
    conn.commit()
    conn.close()
