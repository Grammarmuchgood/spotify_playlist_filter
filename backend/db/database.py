import sqlite3
from pathlib import Path

from config import get_settings

# backend/db/database.py -> parent.parent.parent is the project root,
# regardless of what directory a script importing this happens to be run
# from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_connection() -> sqlite3.Connection:
    # DATABASE_URL looks like "sqlite:///./backend/data/vibe_filter.db" -
    # that "sqlite:///" prefix is a URL-style convention SQLite itself
    # doesn't need, so strip it down to a plain file path.
    db_path = Path(get_settings().database_url.removeprefix("sqlite:///"))
    # A relative path here is meant to be relative to the project root,
    # not to whatever directory a script happens to be run from - without
    # this, running anything outside the project root (e.g.
    # backend/tools/) would silently create/connect to a brand-new EMPTY
    # database at the wrong path instead of erroring, which is a worse
    # failure mode than just not finding the file.
    if not db_path.is_absolute():
        db_path = _PROJECT_ROOT / db_path
    # SQLite won't create missing parent directories itself, so make sure
    # backend/data/ exists before connecting.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # By default, query results come back as plain tuples (row[0],
    # row[1], ...), which means remembering column order. sqlite3.Row
    # makes rows behave like dicts too (row["name"]) - what every other
    # file in this project relies on for readability.
    conn.row_factory = sqlite3.Row
    return conn
