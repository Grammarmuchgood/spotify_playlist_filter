from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from auth.spotify_oauth import get_oauth_manager, get_spotify_client
from pipeline.embed import get_model
from pipeline.genre_buckets import get_bucket_embeddings
from search.hybrid import get_reranker, hybrid_search

# The actual web server application - every route below attaches to this.
app = FastAPI()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.on_event("startup")
def warm_up_search_models() -> None:
    # get_model/get_reranker/get_bucket_embeddings are lazy singletons -
    # loading them takes ~60s the first time (downloading/initializing
    # the embedding and reranker models). Without this, that ~60s would
    # land on whichever request happens to be the first real search
    # after the server starts, making it look hung. Doing it once here
    # at startup instead means every actual search request only pays the
    # real per-query cost (~4s, almost entirely the reranker).
    get_model()
    get_reranker()
    get_bucket_embeddings()


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


@app.get("/search")
def search(q: str, top_n: int = 20):
    # Search reads from the local, already-indexed SQLite database, not
    # live Spotify - no login needed to use it, only to run the fetch/
    # ingest pipeline that populates that database in the first place.
    # hybrid_search returns {"results", "detected", "exact_match_count"} -
    # spread alongside "query" for a flat top-level response the frontend
    # can use to tell "no exact matches" apart from a fully-satisfied one.
    return {"query": q, **hybrid_search(q, top_n=top_n)}


# Mounted last and deliberately last - Starlette matches routes in
# registration order, and this mount is a catch-all for "/" that would
# otherwise shadow every route defined above it. html=True serves
# frontend/index.html for "/" itself, not just exact file paths.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
