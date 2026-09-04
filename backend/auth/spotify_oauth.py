from __future__ import annotations

from pathlib import Path

import spotipy
from spotipy.cache_handler import CacheFileHandler, MemoryCacheHandler
from spotipy.oauth2 import SpotifyOAuth

from config import get_settings

# Space-separated string of permissions being requested from the user -
# this is the exact format Spotify's API expects. Read scopes for pulling
# playlist contents, write scopes for creating/filling the output
# playlist, playback scope for queueing (confirmed necessary directly:
# calling the queue endpoint without it returns 401 "Permissions
# missing", not a device-related error). Adding a new scope to an app
# that users have already authorized doesn't cover their existing grant -
# Spotify requires re-consent, so anyone already logged in needs to log
# out and back in once for a new scope to take effect.
SCOPES = " ".join([
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-modify-playback-state",
])

# backend/auth/spotify_oauth.py -> parent.parent.parent is the project
# root, same pattern used in db/database.py and config.py - anchoring
# every path to a fixed file location instead of the caller's current
# working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _token_cache_path(user_id: str | None) -> Path:
    """None (the legacy default) keeps using the one original, shared
    cache file - unchanged, so nothing that already worked breaks. A real
    user_id gets its own separate cache file inside that user's own data
    directory. Without this, every new login would overwrite the
    previous user's saved token in the same shared file - the same bug
    class per-user database files were built to prevent, just showing up
    in a different part of the app."""
    if user_id is None:
        return _resolve("backend/data/.spotify_token_cache")
    return _resolve(get_settings().user_data_dir) / user_id / ".token_cache"


def get_oauth_manager(user_id: str | None = None) -> SpotifyOAuth:
    settings = get_settings()
    cache_path = _token_cache_path(user_id)
    # spotipy won't create missing parent directories itself.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # SpotifyOAuth (from the spotipy library) implements Spotify's entire
    # OAuth "Authorization Code" flow: building the login URL, exchanging
    # the code Spotify sends back for a real token, and auto-refreshing
    # that token when it expires - none of that is written by hand here.
    return SpotifyOAuth(
        client_id=settings.spotify_client_id,
        client_secret=settings.spotify_client_secret,
        redirect_uri=settings.spotify_redirect_uri,
        scope=SCOPES,
        cache_handler=CacheFileHandler(cache_path=str(cache_path)),
    )


def get_spotify_client(user_id: str | None = None) -> spotipy.Spotify:
    # The actual client other files call API methods on (e.g.
    # sp.current_user()). Handing it the OAuth manager means every request
    # it makes automatically carries a valid, auto-refreshed token - for
    # that user specifically, once user_id is real.
    return spotipy.Spotify(auth_manager=get_oauth_manager(user_id))


def exchange_code_for_token(code: str) -> dict:
    """First half of a new login: trade the one-time code Spotify sent
    back for a real access/refresh token pair. Deliberately doesn't know
    or try to guess a user_id here - that's the actual bootstrapping
    problem this function exists to solve correctly: at this point in
    the flow, the user's identity isn't known yet, since finding it out
    requires USING this token to ask Spotify (see main.py's /callback,
    which does exactly that immediately after calling this).

    Uses MemoryCacheHandler, not a file, on purpose - leaving cache_path
    unset would make spotipy default to a plain ".cache" file wherever
    the process happens to be running from (confirmed by reading
    spotipy's own source), the exact same cwd-dependent landmine already
    fixed twice elsewhere in this project. This exchange only needs the
    returned token dict, not a persistent cache - the real, permanent
    save happens in save_token_for_user() once the real user_id is
    known."""
    settings = get_settings()
    oauth = SpotifyOAuth(
        client_id=settings.spotify_client_id,
        client_secret=settings.spotify_client_secret,
        redirect_uri=settings.spotify_redirect_uri,
        scope=SCOPES,
        cache_handler=MemoryCacheHandler(),
    )
    return oauth.get_access_token(code, as_dict=True)


def client_from_access_token(access_token: str) -> spotipy.Spotify:
    """A client built directly from an already-known access token, not
    through an OAuth manager/cache at all - used exactly once per login,
    in the brief window between having a fresh token and knowing whose
    it is (see exchange_code_for_token)."""
    return spotipy.Spotify(auth=access_token)


def save_token_for_user(user_id: str, token_info: dict) -> None:
    """Second half of a new login, called once the real user_id is known
    - persists the token this user actually needs reused (so they aren't
    asked to log in again every time their hour-long access token
    expires) at their own, isolated path."""
    cache_path = _token_cache_path(user_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    CacheFileHandler(cache_path=str(cache_path)).save_token_to_cache(token_info)
