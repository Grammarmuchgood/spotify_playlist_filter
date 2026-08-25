from auth.spotify_oauth import get_spotify_client


def create_playlist(name: str, description: str = "", public: bool = False) -> dict:
    sp = get_spotify_client()
    # spotipy's named helper user_playlist_create() still targets
    # /users/{id}/playlists, which Spotify removed for Development Mode
    # apps in the Feb 2026 API migration (confirmed via a real 403 when
    # tested). sp._post() is spotipy's low-level "call any endpoint
    # myself" method - used here to bypass the outdated wrapper and hit
    # the current replacement endpoint, POST /me/playlists, directly.
    payload = {"name": name, "description": description, "public": public}
    return sp._post("me/playlists", payload=payload)
