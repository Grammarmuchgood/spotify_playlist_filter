from db.database import get_connection

CREATE_SONGS_TABLE = """
CREATE TABLE IF NOT EXISTS songs (
    track_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    release_date TEXT,
    duration_ms INTEGER,
    isrc TEXT,
    audio_features TEXT,
    lyrics TEXT,
    description TEXT,
    embedding TEXT,
    fetched_at TEXT NOT NULL
)
"""


def init_db() -> None:
    conn = get_connection()
    conn.execute(CREATE_SONGS_TABLE)
    conn.commit()
    conn.close()
