from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from auth.spotify_oauth import get_oauth_manager, get_spotify_client

# The actual web server application - every route below attaches to this.
app = FastAPI()


# @app.get(...) is a decorator: "when an HTTP GET request hits this URL,
# call this function and send its return value back as the response."
# FastAPI calls these functions itself; they're never called directly.
@app.get("/login")
def login():
    oauth = get_oauth_manager()
    # Builds the real Spotify login URL (client ID, scopes, redirect URI
    # baked in as query params), then tells the browser to go there
    # instead (an HTTP redirect) - this is how the user reaches Spotify's
    # actual login/consent screen.
    return RedirectResponse(oauth.get_authorize_url())


@app.get("/callback")
def callback(request: Request):
    oauth = get_oauth_manager()
    # Spotify redirects back here after the user approves access, with
    # ?code=... appended to the URL - this reads that value out.
    code = request.query_params.get("code")
    # Trades the one-time code for a real access token; spotipy saves it
    # to cache_path automatically because of how get_oauth_manager() built
    # this object.
    oauth.get_access_token(code)
    # FastAPI auto-converts a returned dict into a JSON HTTP response.
    return {"status": "authenticated"}


@app.get("/me")
def me():
    sp = get_spotify_client()
    # Hits Spotify's /me endpoint and returns the logged-in user's own
    # profile - used as an end-to-end proof that login actually worked.
    return sp.current_user()
