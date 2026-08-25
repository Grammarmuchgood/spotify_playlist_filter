import sqlite3
from pathlib import Path

from config import get_settings


def get_connection() -> sqlite3.Connection:
    # DATABASE_URL looks like "sqlite:///./backend/data/vibe_filter.db" -
    # that "sqlite:///" prefix is a URL-style convention SQLite itself
    # doesn't need, so strip it down to a plain file path.
    db_path = get_settings().database_url.removeprefix("sqlite:///")
    # SQLite won't create missing parent directories itself, so make sure
    # backend/data/ exists before connecting.
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # By default, query results come back as plain tuples (row[0],
    # row[1], ...), which means remembering column order. sqlite3.Row
    # makes rows behave like dicts too (row["name"]) - what every other
    # file in this project relies on for readability.
    conn.row_factory = sqlite3.Row
    return conn
