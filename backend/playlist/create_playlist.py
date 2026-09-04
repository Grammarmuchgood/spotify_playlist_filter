from __future__ import annotations

from auth.spotify_oauth import get_spotify_client


def create_playlist(name: str, user_id: str | None = None, description: str = "", public: bool = False) -> dict:
    sp = get_spotify_client(user_id)
    # spotipy's named helper user_playlist_create() still targets
    # /users/{id}/playlists, which Spotify removed for Development Mode
    # apps in the Feb 2026 API migration (a real 403 on test). sp._post()
    # is spotipy's low-level "call any endpoint directly" method - used
    # here to skip the outdated wrapper and hit the current replacement
    # endpoint, POST /me/playlists, directly.
    payload = {"name": name, "description": description, "public": public}
    return sp._post("me/playlists", payload=payload)


def add_tracks_to_playlist(playlist_id: str, track_uris: list[str], user_id: str | None = None) -> dict:
    # Confirmed directly against the real API: unlike playlist *creation*
    # above, adding items to an existing playlist by ID was NOT affected
    # by the Feb 2026 migration - spotipy's own playlist_add_items()
    # works normally here.
    sp = get_spotify_client(user_id)
    return sp.playlist_add_items(playlist_id, track_uris)
