from __future__ import annotations

import base64
import json

import itsdangerous
import pytest
from fastapi.testclient import TestClient

import main
from auth.spotify_oauth import _token_cache_path
from config import get_settings


def _signed_session_cookie(data: dict) -> str:
    """Builds a valid session cookie the same way SessionMiddleware does
    internally (itsdangerous.TimestampSigner, same secret), so tests can
    simulate an already-logged-in request without a live Spotify OAuth
    handshake - that needs a real browser and genuinely can't be
    automated here."""
    signer = itsdangerous.TimestampSigner(str(get_settings().session_secret_key))
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


@pytest.fixture
def client():
    return TestClient(main.app)


# ============================================================
# Token cache path resolution - no live Spotify calls needed
# ============================================================

def test_legacy_token_cache_path_when_no_user_id():
    path = _token_cache_path(None)
    assert path.name == ".spotify_token_cache"
    assert "users" not in path.parts


def test_per_user_token_cache_path_is_isolated():
    path_a = _token_cache_path("user_a")
    path_b = _token_cache_path("user_b")
    assert path_a != path_b
    assert "user_a" in path_a.parts
    assert "user_b" in path_b.parts
    assert path_a.name == ".token_cache"


# ============================================================
# The actual security property Step 2 exists to deliver: no session,
# no data - not even accidentally falling back to a default.
# ============================================================

def test_search_rejects_request_with_no_session(client):
    response = client.get("/search", params={"q": "rock songs"})
    assert response.status_code == 401


def test_search_rejects_a_forged_cookie(client):
    # Well-formed-looking but not actually signed with the real secret -
    # simulates someone trying to hand-craft a session rather than log in.
    client.cookies.set("session", "not-a-real-signed-value")
    response = client.get("/search", params={"q": "rock songs"})
    assert response.status_code == 401


def test_me_rejects_request_with_no_session(client):
    response = client.get("/me")
    assert response.status_code == 401


# ============================================================
# A genuinely valid, signed session works end to end
# ============================================================

def test_search_works_with_a_valid_session(client):
    client.cookies.set("session", _signed_session_cookie({"user_id": "peter.dinning0507"}))
    response = client.get("/search", params={"q": "rock songs", "top_n": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["detected"]["genre"] == "Rock"


def test_logout_clears_the_session(client):
    client.cookies.set("session", _signed_session_cookie({"user_id": "peter.dinning0507"}))
    response = client.get("/logout", follow_redirects=False)
    set_cookie = response.headers.get("set-cookie", "")
    # SessionMiddleware sets an expiry date in the past to tell the
    # browser to delete the cookie immediately - see sessions.py's
    # source for the exact mechanism this is asserting on.
    assert "1970" in set_cookie
