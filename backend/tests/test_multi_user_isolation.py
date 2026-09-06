from __future__ import annotations

from config import get_settings
from db.database import get_connection
from db.models import init_db


def test_per_user_databases_are_fully_isolated(tmp_path, monkeypatch):
    # Points user_data_dir at a throwaway pytest tmp_path for this test only
    # (monkeypatch reverts it automatically afterward) - never touches the
    # real backend/data/users/ directory.
    monkeypatch.setattr(get_settings(), "user_data_dir", str(tmp_path))

    init_db("user_a")
    init_db("user_b")

    conn_a = get_connection("user_a")
    conn_a.execute(
        "INSERT INTO songs (track_id, name, artist, fetched_at) VALUES (?, ?, ?, ?)",
        ("track1", "Only In A's Library", "Some Artist", "2026-01-01"),
    )
    conn_a.commit()
    conn_a.close()

    # user_b's database must not see user_a's song - not because a query
    # filtered it out, but because it's a genuinely separate file on disk.
    conn_b = get_connection("user_b")
    rows_b = conn_b.execute("SELECT * FROM songs").fetchall()
    conn_b.close()
    assert len(rows_b) == 0

    conn_a2 = get_connection("user_a")
    rows_a = conn_a2.execute("SELECT * FROM songs").fetchall()
    conn_a2.close()
    assert len(rows_a) == 1
    assert rows_a[0]["name"] == "Only In A's Library"

    # each user's file genuinely exists at its own separate path
    assert (tmp_path / "user_a" / "vibe_filter.db").exists()
    assert (tmp_path / "user_b" / "vibe_filter.db").exists()


def test_legacy_path_unaffected_by_user_data_dir(tmp_path, monkeypatch):
    # user_id=None must keep reading DATABASE_URL, completely independent
    # of user_data_dir - the legacy single-user path and the multi-user
    # path are unrelated settings, so changing one must not affect the other.
    monkeypatch.setattr(get_settings(), "user_data_dir", str(tmp_path))
    conn = get_connection()  # no user_id -> legacy path
    row = conn.execute("SELECT COUNT(*) as n FROM songs").fetchone()
    conn.close()
    assert row["n"] == 649  # the real, existing single-user corpus, untouched by the override above


def test_user_meta_table_only_created_for_real_users(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "user_data_dir", str(tmp_path))
    init_db("user_c")
    conn = get_connection("user_c")
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"songs", "user_meta"} <= tables


def test_migrated_account_holder_data_is_reachable_via_real_user_id():
    # Confirms the actual migration this account went through: the
    # original 649-track corpus is reachable through the real Spotify
    # user_id, not just the legacy no-user_id path. user_meta itself only
    # carries account identity now - processing_status moved to the
    # per-playlist `playlists` table once multi-user playlists existed,
    # since a user can have several playlists in different states at
    # once (see test_playlist_processing.py for that table's own coverage).
    conn = get_connection("peter.dinning0507")
    songs_count = conn.execute("SELECT COUNT(*) as n FROM songs").fetchone()["n"]
    meta = conn.execute("SELECT * FROM user_meta WHERE spotify_user_id = ?", ("peter.dinning0507",)).fetchone()
    # Scoped to the specific playlist this test is actually about ("When"),
    # not an unscoped LIMIT 1 - that was only ever safe while it was the
    # sole row in this table. Real usage since has added more real
    # playlists, so an unordered LIMIT 1 stopped reliably returning this
    # one at all.
    playlist = conn.execute(
        "SELECT * FROM playlists WHERE playlist_id = ?", ("4Jlag9nPT6xEKjNa515hUB",)
    ).fetchone()
    conn.close()
    # >=, not == - this test's job is confirming the migrated corpus is
    # still reachable, not that it's frozen at its original size. It's
    # genuinely grown since (real playlists processed during later
    # live testing add real new songs) - an exact equality here would
    # just go stale again the next time that happens, the same fragility
    # just fixed above for the playlists table.
    assert songs_count >= 649
    assert meta is not None
    assert playlist is not None
    assert playlist["processing_status"] == "complete"
    assert playlist["processed_count"] == 649
