from __future__ import annotations

import sqlite3
from pathlib import Path

from config import get_settings

# Going up three levels from backend/db/database.py gets the project
# root, no matter what directory a script is actually run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_connection(user_id: str | None = None) -> sqlite3.Connection:
    """Leaving user_id as None connects to the original single-user
    database, so every script and test written before adding multi-user
    support still works unchanged. A real user_id gets its own,
    completely separate database file instead:
    {user_data_dir}/{user_id}/vibe_filter.db. One file per user instead
    of a shared "songs" table with a user_id column, on purpose - if a
    query ever forgets to filter by user (an easy mistake with the
    shared-table approach), there's no shared table left for it to leak
    across."""
    if user_id is None:
        # DATABASE_URL looks like "sqlite:///./backend/data/vibe_filter.db" -
        # SQLite doesn't need that "sqlite:///" prefix, so it gets
        # stripped down to a plain file path.
        db_path = Path(get_settings().database_url.removeprefix("sqlite:///"))
    else:
        db_path = Path(get_settings().user_data_dir) / user_id / "vibe_filter.db"
    # A relative path here is meant to be relative to the project root,
    # not wherever a script happens to be run from - otherwise running
    # something from outside the root (e.g. backend/tools/) would
    # silently connect to a brand-new empty database in the wrong place
    # instead of just failing loudly, which is a worse outcome.
    if not db_path.is_absolute():
        db_path = _PROJECT_ROOT / db_path
    # SQLite won't create missing parent directories on its own - this
    # makes sure backend/data/ (or backend/data/users/{user_id}/) exists
    # before connecting.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Plain sqlite3 rows come back as tuples, which means tracking
    # column order by hand. sqlite3.Row instead allows reading columns
    # by name (row["name"]) everywhere else in this project.
    conn.row_factory = sqlite3.Row
    return conn
