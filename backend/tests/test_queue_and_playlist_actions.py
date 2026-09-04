from __future__ import annotations

import base64
import json

import itsdangerous
import pytest
from fastapi.testclient import TestClient
from spotipy import SpotifyException

import main
from config import get_settings


def _signed_session_cookie(data: dict) -> str:
    signer = itsdangerous.TimestampSigner(str(get_settings().session_secret_key))
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def isolated_user_data(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "user_data_dir", str(tmp_path))


@pytest.fixture
def logged_in_client(client, isolated_user_data):
    client.cookies.set("session", _signed_session_cookie({"user_id": "user_a"}))
    return client


# ============================================================
# POST /queue - success and the expected failure (no active device)
# ============================================================

def test_queue_requires_a_session(client):
    response = client.post("/queue", params={"track_id": "abc"})
    assert response.status_code == 401


def test_queue_succeeds(logged_in_client, monkeypatch):
    calls = []
    monkeypatch.setattr("main.queue_track", lambda uri, user_id: calls.append((uri, user_id)))
    response = logged_in_client.post("/queue", params={"track_id": "abc123"})
    assert response.status_code == 200
    assert response.json() == {"queued": True}
    assert calls == [("spotify:track:abc123", "user_a")]


def test_queue_failure_returns_a_clear_503_not_a_500(logged_in_client, monkeypatch):
    # The real, expected failure mode confirmed while building this
    # feature: no active Spotify Connect device. The endpoint must turn
    # this into a clean, documented error - not let it become an
    # unhandled crash the way an unguarded Spotify exception once did for
    # the songs-listing endpoint (see main.py's /playlists/{id}/songs).
    def fake_queue_track(uri, user_id):
        raise SpotifyException(404, -1, "NO_ACTIVE_DEVICE")

    monkeypatch.setattr("main.queue_track", fake_queue_track)
    response = logged_in_client.post("/queue", params={"track_id": "abc123"})
    assert response.status_code == 503
    assert "nothing seems to be actively playing" in response.json()["detail"]


# ============================================================
# POST /add-to-playlist - add to a specific, already-existing playlist
# the user picked, not an auto-created fixed one
# ============================================================

def test_add_to_playlist_requires_a_session(client):
    response = client.post("/add-to-playlist", params={"track_id": "abc", "playlist_id": "p1"})
    assert response.status_code == 401


def test_add_to_playlist_adds_to_the_chosen_playlist(logged_in_client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "main.add_tracks_to_playlist",
        lambda playlist_id, track_uris, user_id=None: calls.append((playlist_id, track_uris, user_id)),
    )
    response = logged_in_client.post("/add-to-playlist", params={"track_id": "abc123", "playlist_id": "my-playlist"})
    assert response.status_code == 200
    assert response.json() == {"status": "added"}
    assert calls == [("my-playlist", ["spotify:track:abc123"], "user_a")]


# ============================================================
# POST /playlists/new - create a playlist and add a song to it in one step
# ============================================================

def test_create_new_playlist_requires_a_session(client):
    response = client.post("/playlists/new", params={"name": "My Playlist", "track_id": "abc"})
    assert response.status_code == 401


def test_create_new_playlist_creates_then_adds_the_track(logged_in_client, monkeypatch):
    create_calls = []
    add_calls = []
    monkeypatch.setattr(
        "main.create_playlist",
        lambda name, user_id=None: create_calls.append((name, user_id)) or {"id": "new-id", "name": name},
    )
    monkeypatch.setattr(
        "main.add_tracks_to_playlist",
        lambda playlist_id, track_uris, user_id=None: add_calls.append((playlist_id, track_uris, user_id)),
    )

    response = logged_in_client.post("/playlists/new", params={"name": "Road Trip", "track_id": "abc123"})
    assert response.status_code == 200
    assert response.json() == {"id": "new-id", "name": "Road Trip"}
    assert create_calls == [("Road Trip", "user_a")]
    assert add_calls == [("new-id", ["spotify:track:abc123"], "user_a")]
