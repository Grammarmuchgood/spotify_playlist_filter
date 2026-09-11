from __future__ import annotations

from datetime import datetime, timezone

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

# One row, in each user's own database file - pure account identity,
# nothing playlist-specific. Lives alongside `songs` in the same per-user
# file rather than a shared "users" table, so a user's whole world stays
# in one portable file - the same reason per-user files were chosen for
# `songs` at all (see db.database.get_connection). Doesn't exist in the
# legacy single-user database, since there's exactly one user there and
# nothing to track per-user.
#
# Used to also carry selected_playlist_id/selected_playlist_name/
# song_count/processing_status - dropped once multi-user playlists (see
# CREATE_PLAYLISTS_TABLE below) made "the one selected playlist" stop
# being a meaningful idea: a user can have several playlists processed at
# once, each in its own state, so that state belongs on a row per
# playlist, not smeared across the one row per user.
CREATE_USER_META_TABLE = """
CREATE TABLE IF NOT EXISTS user_meta (
    spotify_user_id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at TEXT NOT NULL,
    last_active_at TEXT
)
"""

# One row per playlist a user has ever picked - processing state and
# progress live here, not on user_meta, precisely because a user can have
# several playlists in different states at once (one fully processed, one
# mid-run, one untouched). processed_count is updated at each pipeline
# stage boundary as a checkpoint, not smoothly per song - the underlying
# stage functions (fetch_and_store_audio_features etc.) only return a
# final count when they finish, not a running one, so true per-song
# progress would mean changing five already-tested functions' internals
# for a cosmetic improvement. A stage name plus "how many songs have
# cleared this stage so far" is real, honest progress - just not smooth.
CREATE_PLAYLISTS_TABLE = """
CREATE TABLE IF NOT EXISTS playlists (
    playlist_id TEXT PRIMARY KEY,
    name TEXT,
    total_tracks INTEGER,
    processed_count INTEGER NOT NULL DEFAULT 0,
    processing_status TEXT NOT NULL DEFAULT 'not_started',
    last_processed_at TEXT
)
"""

# The many-to-many junction between playlists and songs - see the
# conversation that settled this design: a song's own processed data
# (audio_features/lyrics/description/embedding/genre_bucket, all on
# `songs`) is computed exactly once per track_id regardless of how many
# playlists it's in; this table only ever records membership, never a
# copy of that data. playlist_id and track_id are each a foreign key on
# their own, but neither is unique alone (the same playlist_id repeats
# once per song it contains; the same track_id repeats once per playlist
# it's in) - PRIMARY KEY (playlist_id, track_id) is the pair that's
# actually unique, and SQLite enforces that itself: inserting the same
# pair twice is rejected, not silently duplicated.
CREATE_PLAYLIST_SONGS_TABLE = """
CREATE TABLE IF NOT EXISTS playlist_songs (
    playlist_id TEXT NOT NULL REFERENCES playlists(playlist_id),
    track_id TEXT NOT NULL REFERENCES songs(track_id),
    PRIMARY KEY (playlist_id, track_id)
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
        conn.execute(CREATE_PLAYLISTS_TABLE)
        conn.execute(CREATE_PLAYLIST_SONGS_TABLE)
    conn.commit()
    conn.close()


def upsert_user_meta(user_id: str, **fields) -> None:
    """Creates a user's own account-state row on their first login
    (called from main.py's /callback), or updates specific fields on
    later calls - e.g. display_name if it changes on Spotify's side.
    last_active_at is always refreshed, even when called with no other
    fields.

    **fields becomes column names interpolated directly into the SQL
    below (values stay parameterized, as everywhere else in this
    codebase) - safe here specifically because every call site in this
    codebase passes fields as literal, hand-written keyword arguments
    (e.g. upsert_user_meta(user_id, processing_status="complete")),
    never raw values taken from a request. This function should never be
    called with a **fields dict built from user-controlled input."""
    conn = get_connection(user_id)
    conn.execute(CREATE_USER_META_TABLE)
    now = datetime.now(timezone.utc).isoformat()

    existing = conn.execute("SELECT 1 FROM user_meta WHERE spotify_user_id = ?", (user_id,)).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO user_meta (spotify_user_id, created_at, last_active_at) VALUES (?, ?, ?)",
            (user_id, now, now),
        )

    if fields:
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        conn.execute(
            f"UPDATE user_meta SET {set_clause}, last_active_at = ? WHERE spotify_user_id = ?",
            (*fields.values(), now, user_id),
        )
    else:
        conn.execute("UPDATE user_meta SET last_active_at = ? WHERE spotify_user_id = ?", (now, user_id))

    conn.commit()
    conn.close()


def upsert_playlist(playlist_id: str, user_id: str | None, **fields) -> None:
    """Same insert-if-missing-then-update shape as upsert_user_meta, one
    level down - see that function's docstring for why **fields as column
    names is safe here (identical reasoning: every call site passes
    literal kwargs, never request-controlled data). Called once up front
    when processing starts (name/total_tracks/processing_status), then
    again at each pipeline stage boundary to update processed_count -
    see CREATE_PLAYLISTS_TABLE for why that's checkpoint, not per-song,
    granularity."""
    conn = get_connection(user_id)
    conn.execute(CREATE_PLAYLISTS_TABLE)

    existing = conn.execute("SELECT 1 FROM playlists WHERE playlist_id = ?", (playlist_id,)).fetchone()
    if existing is None:
        conn.execute("INSERT INTO playlists (playlist_id) VALUES (?)", (playlist_id,))

    if fields:
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        conn.execute(
            f"UPDATE playlists SET {set_clause} WHERE playlist_id = ?",
            (*fields.values(), playlist_id),
        )

    conn.commit()
    conn.close()


def link_tracks_to_playlist(track_ids: list[str], playlist_id: str, user_id: str | None = None) -> None:
    """Records playlist membership only - never touches songs' own
    processed data. INSERT OR IGNORE (not a plain INSERT) because this is
    called on every (re-)process of a playlist, including tracks already
    linked from a previous run - the composite primary key on
    playlist_songs would otherwise reject the second run outright rather
    than just no-op on the rows that are already correct."""
    conn = get_connection(user_id)
    conn.execute(CREATE_PLAYLIST_SONGS_TABLE)
    conn.executemany(
        "INSERT OR IGNORE INTO playlist_songs (playlist_id, track_id) VALUES (?, ?)",
        [(playlist_id, track_id) for track_id in track_ids],
    )
    conn.commit()
    conn.close()
