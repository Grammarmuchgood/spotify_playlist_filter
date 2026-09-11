from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from auth.spotify_oauth import (
    client_from_access_token,
    exchange_code_for_token,
    get_oauth_manager,
    get_spotify_client,
    save_token_for_user,
)
from config import get_settings
from db.database import get_connection
from db.models import init_db, upsert_user_meta
from pipeline.embed import get_model
from pipeline.genre_buckets import get_bucket_embeddings
from pipeline.process_playlist import TERMINAL_STATUSES, process_playlist
from search.hybrid import get_reranker, hybrid_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # get_model/get_reranker/get_bucket_embeddings are lazy singletons -
    # loading them takes ~60s the first time (downloading/initializing
    # the embedding and reranker models). Without this, that ~60s would
    # land on whichever request happens to be the first real search
    # after the server starts, making it look hung. Doing it once here
    # at startup instead means every actual search request only pays the
    # real per-query cost (~4s, almost entirely the reranker). The
    # `yield` is where the app actually runs - anything after it would
    # run on shutdown, which nothing here needs.
    get_model()
    get_reranker()
    get_bucket_embeddings()
    yield


# The actual web server application - every route below attaches to this.
# lifespan= is the current FastAPI way of running startup/shutdown code -
# @app.on_event("startup") (the old way) still works but is deprecated,
# and this is the first test in the whole suite to actually import and
# exercise this app object, which is why the warning only surfaced now.
app = FastAPI(lifespan=lifespan)

# Runs on every request, not just these routes - reads the incoming
# signed cookie into request.session, and re-signs/re-sends whatever's
# in request.session on the way out. session_cookie_secure/
# session_secret_key come from Settings (see config.py) rather than
# being hardcoded, so a real secret only ever lives in .env, never in
# code, and the Secure flag can differ between local dev (plain http)
# and production (https-only) without a code change.
app.add_middleware(
    SessionMiddleware,
    secret_key=get_settings().session_secret_key,
    https_only=get_settings().session_cookie_secure,
    same_site="lax",
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def get_current_user_id(request: Request) -> str:
    """A FastAPI dependency - any route that declares
    `user_id: str = Depends(get_current_user_id)` gets this run first,
    and never executes its own body at all if it raises. Reads the
    identity out of the signed session cookie (see SessionMiddleware
    above); raises 401 if there isn't one, rather than letting a route
    silently fall back to any default. This is the one place "is this
    request actually logged in" gets decided - every protected route
    reuses it instead of repeating the check itself, so it can't be
    forgotten at one call site the way a hand-copied check could be."""
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user_id


# @app.get(...) is a decorator: "when an HTTP GET request hits this URL,
# call this function and send its return value back as the response."
# FastAPI calls these functions itself; they're never called directly.
@app.get("/login")
def login():
    # user_id isn't known yet at this point - this is the very start of
    # the flow, before Spotify has told us anything. get_oauth_manager()
    # with no argument builds against the legacy cache path, but that's
    # harmless here: get_authorize_url() below never touches the cache
    # at all, only get_access_token() (in /callback) does.
    oauth = get_oauth_manager()
    # Builds the real Spotify login URL (client ID, scopes, redirect URI
    # baked in as query params), then tells the browser to go there
    # instead (an HTTP redirect) - this is how the user reaches Spotify's
    # actual login/consent screen.
    return RedirectResponse(oauth.get_authorize_url())


@app.get("/callback")
def callback(request: Request):
    # Spotify redirects back here after the user approves access, with
    # ?code=... appended to the URL - this reads that value out.
    code = request.query_params.get("code")
    if code is None:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    # Step 1: trade the one-time code for a real token. Still don't know
    # who this is yet - see exchange_code_for_token's own docstring for
    # why that's unavoidable at this point in the flow.
    token_info = exchange_code_for_token(code)

    # Step 2: use that fresh token directly (not cached anywhere yet) to
    # find out who actually just logged in.
    sp = client_from_access_token(token_info["access_token"])
    profile = sp.current_user()
    user_id = profile["id"]

    # Step 3: now that the real identity is known, set up everything
    # that depends on it. init_db creates this user's own database file
    # (and its user_meta table) if this is their first login - see
    # db/database.py for why this is a separate file per user, not a
    # shared table. upsert_user_meta records who they are; running it
    # again on a later login just refreshes last_active_at. Finally,
    # the token gets saved to THIS user's own cache file, not the shared
    # legacy one - see save_token_for_user's docstring for why that
    # matters (every new login would otherwise overwrite the previous
    # user's saved token).
    init_db(user_id)
    upsert_user_meta(user_id, display_name=profile.get("display_name"))
    save_token_for_user(user_id, token_info)

    # Step 4: the signed session cookie is how every later request
    # proves who it's from - see get_current_user_id above and
    # SessionMiddleware's setup at the top of this file.
    request.session["user_id"] = user_id

    return RedirectResponse("/")


@app.get("/logout")
def logout(request: Request):
    # Clearing the session makes SessionMiddleware send back a cookie
    # with an expiry date in the past, which tells the browser to delete
    # it immediately - not "stop sending it eventually," gone on the
    # very next page load.
    request.session.clear()
    return RedirectResponse("/")


@app.get("/me")
def me(user_id: str = Depends(get_current_user_id)):
    sp = get_spotify_client(user_id)
    # Hits Spotify's /me endpoint and returns the logged-in user's own
    # profile - used as an end-to-end proof that login actually worked,
    # for this specific logged-in user rather than whoever happens to be
    # in the legacy cache.
    return sp.current_user()


@app.get("/search")
def search(
    q: str, top_n: int = 20, playlist_id: str | None = None, user_id: str = Depends(get_current_user_id)
):
    # Depends(get_current_user_id) means this line never runs at all for
    # a request with no valid session - there is no path left here that
    # reaches anyone's data without a real, verified login. hybrid_search
    # returns {"results", "detected", "exact_match_count"} - spread
    # alongside "query" for a flat top-level response the frontend can
    # use to tell "no exact matches" apart from a fully-satisfied one.
    # playlist_id=None searches everything this user has ever processed,
    # across every playlist at once - a real playlist_id scopes results
    # to just that one (see search.hybrid._fetch_songs).
    return {"query": q, **hybrid_search(q, top_n=top_n, user_id=user_id, playlist_id=playlist_id)}


@app.get("/playlists")
def list_playlists(user_id: str = Depends(get_current_user_id)):
    # The front page's picker data - every real playlist this Spotify
    # account has, not just ones already processed, so a user can pick
    # a brand new one too. Paginated the same way fetch_playlist_items
    # pages through a single playlist's tracks: follow results["next"]
    # until Spotify stops returning one.
    sp = get_spotify_client(user_id)
    playlists = []
    results = sp.current_user_playlists()
    while True:
        for p in results["items"]:
            playlists.append({
                "id": p["id"],
                "name": p["name"],
                # Post-Feb-2026-migration shape, confirmed directly
                # against the live API - see process_playlist's matching
                # comment on the same rename.
                "track_count": p["items"]["total"],
                "image_url": p["images"][0]["url"] if p["images"] else None,
            })
        if not results["next"]:
            break
        results = sp.next(results)
    return {"playlists": playlists}


@app.post("/playlists/{playlist_id}/process")
def start_processing(playlist_id: str, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)):
    # Guards against a double-click or a page refresh starting a second,
    # overlapping background run against the same playlist - checked
    # BEFORE scheduling anything, not left to the background task itself
    # to sort out, since two concurrent runs writing to the same rows has
    # no well-defined outcome. init_db is called here (not just relied on
    # from /callback at login) so this route works correctly even if
    # somehow reached before a normal login ever ran it - IF NOT EXISTS
    # makes the call itself free on every other request.
    init_db(user_id)
    conn = get_connection(user_id)
    row = conn.execute("SELECT processing_status FROM playlists WHERE playlist_id = ?", (playlist_id,)).fetchone()
    conn.close()
    if row is not None and row["processing_status"] not in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="This playlist is already being processed")

    # add_task runs process_playlist AFTER this response is already sent
    # back to the browser - a real playlist can take several minutes
    # (audio downloads, lyrics lookups, an LLM call per song), which would
    # otherwise leave the request hanging well past any sane timeout.
    background_tasks.add_task(process_playlist, user_id, playlist_id)
    return {"status": "started"}


@app.get("/playlists/{playlist_id}/status")
def playlist_status(playlist_id: str, user_id: str = Depends(get_current_user_id)):
    # What the frontend polls every few seconds while a background run is
    # in flight, to show "342 of 649" and know when search is ready.
    init_db(user_id)
    conn = get_connection(user_id)
    row = conn.execute("SELECT * FROM playlists WHERE playlist_id = ?", (playlist_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="This playlist hasn't been processed yet")
    return dict(row)


# Mounted last and deliberately last - Starlette matches routes in
# registration order, and this mount is a catch-all for "/" that would
# otherwise shadow every route defined above it. html=True serves
# frontend/index.html for "/" itself, not just exact file paths.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
