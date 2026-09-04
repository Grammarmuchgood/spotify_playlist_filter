from __future__ import annotations

from spotipy import SpotifyException

from auth.spotify_oauth import get_spotify_client


def queue_track(track_uri: str, user_id: str | None = None) -> None:
    """Raises SpotifyException on failure - most commonly a 404 with
    reason NO_ACTIVE_DEVICE, Spotify's real, expected failure mode when
    nothing is actively playing anywhere on the account (genuinely
    common, not an edge case - queueing needs a live Spotify Connect
    session, and just having the app open isn't enough). No error
    handling here on purpose - the caller (main.py) decides what
    "failed" should mean to the user; this function's only job is the
    API call itself."""
    sp = get_spotify_client(user_id)
    sp.add_to_queue(track_uri)
