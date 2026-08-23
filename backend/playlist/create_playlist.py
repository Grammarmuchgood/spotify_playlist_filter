from auth.spotify_oauth import get_spotify_client


def create_playlist(name: str, description: str = "", public: bool = False) -> dict:
    sp = get_spotify_client()
    # spotipy's user_playlist_create() still targets /users/{id}/playlists,
    # which Spotify removed for Development Mode apps in the Feb 2026 API
    # migration. The replacement is POST /me/playlists.
    payload = {"name": name, "description": description, "public": public}
    return sp._post("me/playlists", payload=payload)
