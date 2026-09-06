from __future__ import annotations

import base64
import json

import itsdangerous
import pytest
from fastapi.testclient import TestClient

import main
from config import get_settings
from db.database import get_connection
from db.models import init_db, upsert_playlist


def _signed_session_cookie(data: dict) -> str:
    """Same technique as test_auth.py - a genuinely valid session cookie,
    signed with the app's real secret, without a live OAuth handshake."""
    signer = itsdangerous.TimestampSigner(str(get_settings().session_secret_key))
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def isolated_user_data(tmp_path, monkeypatch):
    # NOT autouse - one test below (test_search_with_playlist_id_scopes_
    # to_the_real_migrated_playlist) deliberately reads the real account
    # holder's production data, same as test_auth.py's existing tests,
    # and must NOT have user_data_dir redirected out from under it.
    monkeypatch.setattr(get_settings(), "user_data_dir", str(tmp_path))


@pytest.fixture
def logged_in_client(client, isolated_user_data):
    client.cookies.set("session", _signed_session_cookie({"user_id": "user_a"}))
    return client


# ============================================================
# GET /playlists - the front page's picker data
# ============================================================

def test_list_playlists_requires_a_session(client):
    response = client.get("/playlists")
    assert response.status_code == 401


def test_list_playlists_returns_spotify_playlists_with_post_migration_field_names(logged_in_client, monkeypatch):
    class _FakeSpotify:
        def current_user_playlists(self):
            return {
                "items": [
                    {"id": "p1", "name": "Workout", "items": {"total": 12}, "images": [{"url": "http://img/1"}]},
                    {"id": "p2", "name": "No Cover Art", "items": {"total": 3}, "images": None},
                ],
                "next": None,
            }

    monkeypatch.setattr(main, "get_spotify_client", lambda user_id: _FakeSpotify())
    response = logged_in_client.get("/playlists")
    assert response.status_code == 200
    playlists = response.json()["playlists"]
    assert playlists == [
        {"id": "p1", "name": "Workout", "track_count": 12, "image_url": "http://img/1"},
        {"id": "p2", "name": "No Cover Art", "track_count": 3, "image_url": None},
    ]


def test_list_playlists_follows_pagination(logged_in_client, monkeypatch):
    page_1 = {"items": [{"id": "p1", "name": "First", "items": {"total": 1}, "images": []}], "next": "page2"}
    page_2 = {"items": [{"id": "p2", "name": "Second", "items": {"total": 1}, "images": []}], "next": None}

    class _FakeSpotify:
        def current_user_playlists(self):
            return page_1

        def next(self, results):
            return page_2

    monkeypatch.setattr(main, "get_spotify_client", lambda user_id: _FakeSpotify())
    response = logged_in_client.get("/playlists")
    ids = [p["id"] for p in response.json()["playlists"]]
    assert ids == ["p1", "p2"]


# ============================================================
# POST /playlists/{id}/process - the double-trigger guard
# ============================================================

def test_start_processing_requires_a_session(client):
    response = client.post("/playlists/abc/process")
    assert response.status_code == 401


def test_start_processing_accepts_when_never_processed_before(logged_in_client, monkeypatch):
    started_with = []
    monkeypatch.setattr(main, "process_playlist", lambda user_id, playlist_id: started_with.append((user_id, playlist_id)))
    response = logged_in_client.post("/playlists/new_playlist/process")
    assert response.status_code == 200
    assert started_with == [("user_a", "new_playlist")]


def test_start_processing_rejects_when_already_in_progress(logged_in_client, monkeypatch):
    init_db("user_a")
    upsert_playlist("mid_run", "user_a", processing_status="describing")
    monkeypatch.setattr(main, "process_playlist", lambda user_id, playlist_id: pytest.fail("should not have started a second run"))

    response = logged_in_client.post("/playlists/mid_run/process")
    assert response.status_code == 409


@pytest.mark.parametrize("status", ["not_started", "complete", "failed"])
def test_start_processing_accepts_when_status_is_terminal(logged_in_client, monkeypatch, status):
    init_db("user_a")
    upsert_playlist("retry_me", "user_a", processing_status=status)
    started = []
    monkeypatch.setattr(main, "process_playlist", lambda user_id, playlist_id: started.append(playlist_id))
    response = logged_in_client.post("/playlists/retry_me/process")
    assert response.status_code == 200
    assert started == ["retry_me"]


# ============================================================
# GET /playlists/{id}/status - what the frontend polls
# ============================================================

def test_playlist_status_404_before_ever_processed(logged_in_client):
    response = logged_in_client.get("/playlists/never_touched/status")
    assert response.status_code == 404


def test_playlist_status_reports_real_progress(logged_in_client):
    init_db("user_a")
    upsert_playlist(
        "in_progress", "user_a",
        name="My Playlist", total_tracks=100, processed_count=42, processing_status="embedding",
    )
    response = logged_in_client.get("/playlists/in_progress/status")
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "embedding"
    assert body["processed_count"] == 42
    assert body["total_tracks"] == 100


# ============================================================
# /search's playlist_id parameter, against the real migrated account
# ============================================================

def test_search_with_playlist_id_scopes_to_the_real_migrated_playlist(client):
    client.cookies.set("session", _signed_session_cookie({"user_id": "peter.dinning0507"}))
    # Deliberately NOT using the tmp_path isolation fixture's user_id here
    # - this reads the account holder's real, migrated production data on
    # purpose, the same way test_auth.py's existing tests do.
    # "When" specifically, by ID - not an unscoped LIMIT 1, which stopped
    # reliably returning this one once real usage added more playlists
    # (Metal, Big artists big songs) as genuine additional rows in the
    # same table. This test needs a playlist with real Rock-bucketed
    # songs, which "When" specifically has - not just any processed one.
    conn = get_connection("peter.dinning0507")
    real_playlist_id = conn.execute(
        "SELECT playlist_id FROM playlists WHERE playlist_id = ?", ("4Jlag9nPT6xEKjNa515hUB",)
    ).fetchone()["playlist_id"]
    conn.close()

    response = client.get("/search", params={"q": "rock songs", "top_n": 3, "playlist_id": real_playlist_id})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["detected"]["genre"] == "Rock"

    response_empty = client.get("/search", params={"q": "rock songs", "top_n": 3, "playlist_id": "not_a_real_playlist"})
    assert response_empty.json()["results"] == []
