from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from auth.spotify_oauth import get_oauth_manager, get_spotify_client

app = FastAPI()


@app.get("/login")
def login():
    oauth = get_oauth_manager()
    return RedirectResponse(oauth.get_authorize_url())


@app.get("/callback")
def callback(request: Request):
    oauth = get_oauth_manager()
    code = request.query_params.get("code")
    oauth.get_access_token(code)
    return {"status": "authenticated"}


@app.get("/me")
def me():
    sp = get_spotify_client()
    return sp.current_user()
