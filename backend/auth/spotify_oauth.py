import spotipy
from spotipy.oauth2 import SpotifyOAuth

from config import get_settings

SCOPES = " ".join([
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-public",
    "playlist-modify-private",
])


def get_oauth_manager() -> SpotifyOAuth:
    settings = get_settings()
    return SpotifyOAuth(
        client_id=settings.spotify_client_id,
        client_secret=settings.spotify_client_secret,
        redirect_uri=settings.spotify_redirect_uri,
        scope=SCOPES,
        cache_path="backend/data/.spotify_token_cache",
    )


def get_spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=get_oauth_manager())
