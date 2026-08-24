import sqlite3
from pathlib import Path

from config import get_settings


def get_connection() -> sqlite3.Connection:
    db_path = get_settings().database_url.removeprefix("sqlite:///")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
