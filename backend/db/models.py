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


def init_db() -> None:
    conn = get_connection()
    # IF NOT EXISTS makes this safe to call every time (it's called at
    # the top of fetch_playlist.py's flow) - does nothing if the table's
    # already there. Note: this does NOT alter an existing table if the
    # schema changes later - that requires rebuilding the .db file.
    conn.execute(CREATE_SONGS_TABLE)
    conn.commit()
    conn.close()
