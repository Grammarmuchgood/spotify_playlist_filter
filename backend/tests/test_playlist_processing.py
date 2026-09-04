from __future__ import annotations

import json
import sqlite3

import pytest

from config import get_settings
from db.database import get_connection
from db.models import init_db, link_tracks_to_playlist, upsert_playlist
import pipeline.process_playlist as process_playlist_module
from pipeline.process_playlist import TERMINAL_STATUSES, process_playlist
from search.hybrid import _fetch_songs, hybrid_search


@pytest.fixture(autouse=True)
def _isolated_user_data(tmp_path, monkeypatch):
    # Every test in this file uses a throwaway per-test directory, never
    # the real backend/data/users/ - same isolation pattern as
    # test_multi_user_isolation.py.
    monkeypatch.setattr(get_settings(), "user_data_dir", str(tmp_path))


# Qwen3-Embedding-0.6B (see pipeline.embed.get_model) produces 1024-dim
# vectors - confirmed directly (get_model().encode(["x"]).shape) rather
# than guessed, since hybrid_search's cosine_similarity crashes outright
# on a shape mismatch rather than silently producing a wrong answer. The
# actual numbers are meaningless filler; only the dimensionality matters
# for these tests, which is why this is a plain literal, not a real
# model call - loading the real model here would also break this file's
# fast, no-ML pattern (matching conftest.py's separation between fast
# detection tests and the slower reranker-backed full-pipeline ones).
_FAKE_EMBEDDING = json.dumps([0.1] * 1024)


# A complete, valid audio_features blob - status="ok" plus all four
# numeric fields audio_features.describe_features actually reads.
# generate_descriptions (unlike every other stage) re-describes every
# "ok" row on every run, not just NULL ones - corpus-wide thresholds
# shift as more songs get added, so leaving old rows undescribed would
# make old and new descriptions inconsistent (see its own docstring) -
# so this needs to survive a REAL call to that function, not just carry
# enough to look plausible.
_FAKE_AUDIO_FEATURES = json.dumps({
    "status": "ok", "tempo_bpm": 120.0, "energy_rms": 0.1,
    "spectral_centroid_hz": 2000.0, "harmonic_ratio": 0.5,
})


def _insert_fully_processed_song(user_id: str, track_id: str, name: str = "Existing Song") -> None:
    """A song that's already been through every pipeline stage - used to
    test that reprocessing reuses it, without needing a real embedding
    model or LLM call to produce realistic-looking data."""
    conn = get_connection(user_id)
    init_db(user_id)
    conn.execute(
        """
        INSERT INTO songs (track_id, name, artist, primary_artist, album, release_date,
                            duration_ms, isrc, audio_features, lyrics, description, embedding,
                            genre_bucket, fetched_at)
        VALUES (?, ?, 'Some Artist', 'Some Artist', 'Some Album', '2020-01-01',
                200000, 'ISRC123', ?, 'la la la',
                '{"description":"a test song"}', ?, 'Pop', '2026-01-01')
        """,
        (track_id, name, _FAKE_AUDIO_FEATURES, _FAKE_EMBEDDING),
    )
    conn.commit()
    conn.close()


def _fake_track_item(track_id: str, name: str = "New Song") -> dict:
    return {
        "item": {
            "id": track_id,
            "name": name,
            "artists": [{"name": "Some Artist"}],
            "album": {"name": "Some Album", "release_date": "2020-01-01"},
            "duration_ms": 200000,
            "external_ids": {"isrc": "ISRC999"},
        }
    }


# ============================================================
# Schema: the composite primary key is a real, enforced constraint,
# not just a convention followed by the application code
# ============================================================

def test_playlist_songs_rejects_duplicate_link_via_raw_insert():
    init_db("user_a")
    conn = get_connection("user_a")
    conn.execute("INSERT INTO playlists (playlist_id) VALUES ('p1')")
    conn.execute(
        "INSERT INTO songs (track_id, name, artist, fetched_at) VALUES ('t1', 'Song', 'Artist', '2026-01-01')"
    )
    conn.execute("INSERT INTO playlist_songs (playlist_id, track_id) VALUES ('p1', 't1')")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO playlist_songs (playlist_id, track_id) VALUES ('p1', 't1')")
    conn.close()


def test_link_tracks_to_playlist_is_idempotent():
    init_db("user_a")
    conn = get_connection("user_a")
    conn.execute(
        "INSERT INTO songs (track_id, name, artist, fetched_at) VALUES ('t1', 'Song', 'Artist', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    link_tracks_to_playlist(["t1"], "p1", "user_a")
    link_tracks_to_playlist(["t1"], "p1", "user_a")  # re-running must not crash or duplicate

    conn = get_connection("user_a")
    rows = conn.execute("SELECT * FROM playlist_songs").fetchall()
    conn.close()
    assert len(rows) == 1


# ============================================================
# upsert_playlist - the same insert-if-missing-then-update shape as
# upsert_user_meta, one level down
# ============================================================

def test_upsert_playlist_creates_then_updates():
    upsert_playlist("p1", "user_a", name="My Playlist", processing_status="fetching")
    conn = get_connection("user_a")
    row = dict(conn.execute("SELECT * FROM playlists WHERE playlist_id = 'p1'").fetchone())
    conn.close()
    assert row["name"] == "My Playlist"
    assert row["processing_status"] == "fetching"
    assert row["processed_count"] == 0  # schema default, untouched by this call

    upsert_playlist("p1", "user_a", processing_status="complete", processed_count=5)
    conn = get_connection("user_a")
    row = dict(conn.execute("SELECT * FROM playlists WHERE playlist_id = 'p1'").fetchone())
    conn.close()
    assert row["name"] == "My Playlist"  # untouched by the second call, not wiped
    assert row["processing_status"] == "complete"
    assert row["processed_count"] == 5


# ============================================================
# The core multi-playlist requirement: a song already fully processed
# under one playlist is reused, not reprocessed, when it shows up in
# another
# ============================================================

def test_song_already_processed_is_reused_across_playlists(monkeypatch):
    """The orchestrator calls all six stage functions unconditionally
    every time - it never asks "has this song been seen before" itself.
    The reuse behavior comes entirely from each stage's own WHERE-column-
    IS-NULL filter finding nothing left to do (confirmed by reading every
    stage's query before writing process_playlist - see its docstring).
    So this test deliberately does NOT mock the six stages away - it lets
    the real ones run, and proves the externally-sourced fields (lyrics,
    the LLM description, the embedding, the genre bucket - everything
    that would mean a real iTunes/lyrics.ovh/Anthropic/model call to
    redo) come back byte-for-byte unchanged. audio_features' own
    "description" sub-phrase is the one deliberate exception -
    generate_descriptions re-derives it for every "ok" row on every run,
    by design, since it's cheap/local and corpus-wide thresholds shift as
    more songs are added (see that function's own docstring) - that's
    not a violation of "reuse the expensive stuff," it's a different,
    intentional contract."""
    _insert_fully_processed_song("user_a", "shared_track", name="Shared Song")
    link_tracks_to_playlist(["shared_track"], "playlist_a", "user_a")
    upsert_playlist("playlist_a", "user_a", name="Playlist A", total_tracks=1, processing_status="complete")
    song_before = dict(get_connection("user_a").execute("SELECT * FROM songs WHERE track_id = 'shared_track'").fetchone())

    class _FakeSpotify:
        def playlist(self, playlist_id, fields=None):
            return {"name": "Playlist B", "items": {"total": 1}}

    monkeypatch.setattr(process_playlist_module, "get_spotify_client", lambda user_id: _FakeSpotify())
    monkeypatch.setattr(
        process_playlist_module, "fetch_playlist_items", lambda playlist_id, user_id: [_fake_track_item("shared_track")]
    )

    process_playlist("user_a", "playlist_b")

    conn = get_connection("user_a")
    links = {tuple(r) for r in conn.execute("SELECT playlist_id, track_id FROM playlist_songs").fetchall()}
    song = dict(conn.execute("SELECT * FROM songs WHERE track_id = 'shared_track'").fetchone())
    playlist_b = dict(conn.execute("SELECT * FROM playlists WHERE playlist_id = 'playlist_b'").fetchone())
    conn.close()

    assert ("playlist_a", "shared_track") in links
    assert ("playlist_b", "shared_track") in links  # now linked to BOTH, not moved
    # The expensive, externally-sourced fields - exactly what "reuse
    # instead of reprocess" is actually protecting - are untouched.
    assert song["lyrics"] == song_before["lyrics"]
    assert song["description"] == song_before["description"]
    assert song["embedding"] == song_before["embedding"]
    assert song["genre_bucket"] == song_before["genre_bucket"]
    # audio_features' JSON as a whole may differ (its "description"
    # sub-phrase is deliberately re-derived every run), but the real
    # signal-processing measurements inside it never change.
    assert json.loads(song["audio_features"])["tempo_bpm"] == json.loads(song_before["audio_features"])["tempo_bpm"]
    assert playlist_b["processing_status"] == "complete"
    assert playlist_b["processed_count"] == 1


# ============================================================
# Orchestration for a genuinely new song - correct stage order, correct
# checkpoint updates. The six stage functions are mocked here
# deliberately: they're pre-existing, already-proven code (it's what
# built the real 649-song corpus over this project's history) - what's
# actually new and needs testing here is the sequencing around them.
# ============================================================

def test_process_playlist_runs_stages_in_dependency_order(monkeypatch):
    class _FakeSpotify:
        def playlist(self, playlist_id, fields=None):
            return {"name": "New Playlist", "items": {"total": 1}}

    monkeypatch.setattr(process_playlist_module, "get_spotify_client", lambda user_id: _FakeSpotify())
    monkeypatch.setattr(
        process_playlist_module, "fetch_playlist_items", lambda playlist_id, user_id: [_fake_track_item("new_track")]
    )

    call_order = []
    status_snapshots = []

    def _record(label):
        def _fn(user_id=None):
            call_order.append(label)
            # Snapshot processing_status at the moment each stage runs -
            # confirms upsert_playlist(status=...) happens BEFORE the
            # stage itself, not after, so a status poll mid-run reflects
            # what's actually happening rather than what just finished.
            conn = get_connection(user_id)
            row = conn.execute("SELECT processing_status FROM playlists WHERE playlist_id = 'new_playlist'").fetchone()
            conn.close()
            status_snapshots.append((label, row["processing_status"]))
            return 0
        return _fn

    for name in (
        "fetch_and_store_audio_features", "generate_descriptions", "fetch_and_store_lyrics",
        "fetch_and_store_descriptions", "embed_and_store", "assign_genre_buckets",
    ):
        monkeypatch.setattr(process_playlist_module, name, _record(name))

    process_playlist("user_a", "new_playlist")

    assert call_order == [
        "fetch_and_store_audio_features", "generate_descriptions", "fetch_and_store_lyrics",
        "fetch_and_store_descriptions", "embed_and_store", "assign_genre_buckets",
    ]
    # generate_descriptions must run strictly after audio_features (it
    # summarizes that data) and strictly before fetch_and_store_descriptions
    # (the LLM stage reads its output) - see process_playlist's own
    # docstring for why this ordering is a real dependency, not a choice.
    assert status_snapshots[0] == ("fetch_and_store_audio_features", "audio_features")
    assert status_snapshots[1] == ("generate_descriptions", "generating_descriptions")
    assert status_snapshots[3] == ("fetch_and_store_descriptions", "describing")

    conn = get_connection("user_a")
    playlist = dict(conn.execute("SELECT * FROM playlists WHERE playlist_id = 'new_playlist'").fetchone())
    linked = conn.execute("SELECT * FROM playlist_songs WHERE playlist_id = 'new_playlist'").fetchall()
    conn.close()
    assert playlist["processing_status"] == "complete"
    assert playlist["name"] == "New Playlist"
    assert playlist["total_tracks"] == 1
    assert len(linked) == 1


def test_process_playlist_skips_local_files_and_missing_tracks(monkeypatch):
    """A playlist item with no resolvable track (a local file, or a track
    removed from Spotify's catalog since being added) must not be linked
    or crash the run - confirmed against a real account's actual
    playlist, which genuinely has both cases."""
    class _FakeSpotify:
        def playlist(self, playlist_id, fields=None):
            return {"name": "Messy Playlist", "items": {"total": 3}}

    fake_items = [
        {"item": None},  # a track removed from Spotify's catalog
        {"item": {"id": None, "name": "local.mp3", "is_local": True}},  # a local file
        _fake_track_item("real_track"),
    ]
    monkeypatch.setattr(process_playlist_module, "get_spotify_client", lambda user_id: _FakeSpotify())
    monkeypatch.setattr(process_playlist_module, "fetch_playlist_items", lambda playlist_id, user_id: fake_items)
    for name in (
        "fetch_and_store_audio_features", "generate_descriptions", "fetch_and_store_lyrics",
        "fetch_and_store_descriptions", "embed_and_store", "assign_genre_buckets",
    ):
        monkeypatch.setattr(process_playlist_module, name, lambda user_id=None: 0)

    process_playlist("user_a", "messy_playlist")

    conn = get_connection("user_a")
    linked = conn.execute("SELECT track_id FROM playlist_songs WHERE playlist_id = 'messy_playlist'").fetchall()
    songs = conn.execute("SELECT track_id FROM songs").fetchall()
    conn.close()
    assert [r["track_id"] for r in linked] == ["real_track"]
    assert [r["track_id"] for r in songs] == ["real_track"]


# ============================================================
# Failure handling: a stage that throws marks the playlist 'failed'
# (not stuck in 'processing' forever) and still surfaces the real error
# ============================================================

def test_stage_failure_marks_playlist_failed_and_reraises(monkeypatch):
    class _FakeSpotify:
        def playlist(self, playlist_id, fields=None):
            return {"name": "Doomed Playlist", "items": {"total": 1}}

    monkeypatch.setattr(process_playlist_module, "get_spotify_client", lambda user_id: _FakeSpotify())
    monkeypatch.setattr(
        process_playlist_module, "fetch_playlist_items", lambda playlist_id, user_id: [_fake_track_item("t1")]
    )

    def _boom(user_id=None):
        raise RuntimeError("simulated iTunes outage")

    monkeypatch.setattr(process_playlist_module, "fetch_and_store_audio_features", _boom)

    with pytest.raises(RuntimeError, match="simulated iTunes outage"):
        process_playlist("user_a", "doomed_playlist")

    conn = get_connection("user_a")
    status = conn.execute("SELECT processing_status FROM playlists WHERE playlist_id = 'doomed_playlist'").fetchone()
    conn.close()
    assert status["processing_status"] == "failed"


def test_terminal_statuses_are_exactly_the_non_in_progress_states():
    assert TERMINAL_STATUSES == {"not_started", "complete", "failed"}
    in_progress = {
        "fetching", "audio_features", "generating_descriptions", "lyrics", "describing",
        "embedding", "genre_bucketing",
    }
    assert TERMINAL_STATUSES.isdisjoint(in_progress)


# ============================================================
# Search actually scopes to one playlist when asked
# ============================================================

def test_search_playlist_scoping_excludes_other_playlists_songs():
    _insert_fully_processed_song("user_a", "in_playlist_1", name="Only In One")
    _insert_fully_processed_song("user_a", "in_playlist_2", name="Only In Two")
    link_tracks_to_playlist(["in_playlist_1"], "playlist_1", "user_a")
    link_tracks_to_playlist(["in_playlist_2"], "playlist_2", "user_a")

    songs_1, _ = _fetch_songs("user_a", playlist_id="playlist_1")
    songs_2, _ = _fetch_songs("user_a", playlist_id="playlist_2")
    songs_all, _ = _fetch_songs("user_a", playlist_id=None)

    assert [s["track_id"] for s in songs_1] == ["in_playlist_1"]
    assert [s["track_id"] for s in songs_2] == ["in_playlist_2"]
    assert {s["track_id"] for s in songs_all} == {"in_playlist_1", "in_playlist_2"}


def test_hybrid_search_respects_playlist_id():
    _insert_fully_processed_song("user_a", "rock_track", name="Rock Song")
    conn = get_connection("user_a")
    conn.execute("UPDATE songs SET genre_bucket = 'Rock' WHERE track_id = 'rock_track'")
    conn.commit()
    conn.close()
    link_tracks_to_playlist(["rock_track"], "only_playlist", "user_a")

    result_scoped = hybrid_search("rock songs", top_n=5, user_id="user_a", playlist_id="only_playlist")
    result_other = hybrid_search("rock songs", top_n=5, user_id="user_a", playlist_id="some_other_playlist")

    assert len(result_scoped["results"]) == 1
    assert result_scoped["results"][0]["track_id"] == "rock_track"
    assert len(result_other["results"]) == 0  # nothing linked to this playlist_id at all
