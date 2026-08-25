import spotipy
from spotipy.oauth2 import SpotifyOAuth

from config import get_settings

# Space-separated string of permissions we're asking the user to grant -
# this is the exact format Spotify's API expects. Read scopes for pulling
# playlist contents, write scopes for creating/filling the output playlist.
SCOPES = " ".join([
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-public",
    "playlist-modify-private",
])


def get_oauth_manager() -> SpotifyOAuth:
    settings = get_settings()
    # SpotifyOAuth (from the spotipy library) implements Spotify's entire
    # OAuth "Authorization Code" flow: building the login URL, exchanging
    # the code Spotify sends back for a real token, and auto-refreshing
    # that token when it expires - none of that is written by hand here.
    return SpotifyOAuth(
        client_id=settings.spotify_client_id,
        client_secret=settings.spotify_client_secret,
        redirect_uri=settings.spotify_redirect_uri,
        scope=SCOPES,
        # Where the access/refresh token gets saved on disk after login,
        # so the app doesn't need to re-login on every restart.
        cache_path="backend/data/.spotify_token_cache",
    )


def get_spotify_client() -> spotipy.Spotify:
    # The actual client other files call API methods on (e.g.
    # sp.current_user()). Handing it the OAuth manager means every request
    # it makes automatically carries a valid, auto-refreshed token.
    return spotipy.Spotify(auth_manager=get_oauth_manager())
