import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from spotipy import SpotifyException
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
from pipeline.fetch_playlist import fetch_playlist_items
from pipeline.genre_buckets import get_bucket_embeddings
from playlist.create_playlist import add_tracks_to_playlist, create_playlist
from playlist.queue import queue_track
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
# @app.on_event("startup") (the old way) still works but is deprecated.
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

# The React app's BUILT output (npm run build's dist/), not its source -
# this only gets served in production. In local dev, the browser only
# ever talks to Vite's dev server (127.0.0.1:5173), which proxies API
# paths back to this backend - this mount is never actually reached
# during dev, only once there's no separate dev server anymore. The old
# frontend/ (vanilla JS) directory is no longer mounted at all - the
# React app is now the real, complete frontend it was built to replace,
# not left running alongside it.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend-react" / "dist"


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

    # NOT a relative "/" - Spotify redirects the browser here directly,
    # to whatever host:port SPOTIFY_REDIRECT_URI says, completely
    # independent of which port a frontend dev server happens to be
    # running on (this request never passes through Vite's dev proxy at
    # all, unlike every other route in this file). A relative redirect
    # would strand the browser on this backend's own static files
    # instead of sending it back to the frontend actually being used.
    # frontend_url defaults to "/" for production, where the built
    # frontend and this backend share one origin and there's nothing to
    # redirect across.
    return RedirectResponse(get_settings().frontend_url)


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


@app.get("/playlists/{playlist_id}/songs")
def list_playlist_songs(playlist_id: str, user_id: str = Depends(get_current_user_id)):
    # Browsable regardless of processing state - a playlist that's never
    # been touched by the pipeline has zero rows in playlist_songs at all,
    # so there's nothing local to read yet. In that case, fall back to
    # asking Spotify directly for the same raw track list the pipeline
    # itself would fetch - same function process_playlist uses, called
    # here purely to read, not to start any processing.
    conn = get_connection(user_id)
    rows = conn.execute(
        """
        SELECT songs.track_id, songs.name, songs.artist, songs.genre_bucket, songs.description
        FROM songs
        JOIN playlist_songs ON songs.track_id = playlist_songs.track_id
        WHERE playlist_songs.playlist_id = ?
        ORDER BY songs.name COLLATE NOCASE
        """,
        (playlist_id,),
    ).fetchall()
    conn.close()

    if rows:
        songs = [
            {
                "track_id": row["track_id"],
                "name": row["name"],
                "artist": row["artist"],
                "genre_bucket": row["genre_bucket"],
                # description is stored as a JSON blob ({"description":
                # ..., "mood": ..., ...}, see pipeline/describe.py) - NULL
                # for a song this playlist has linked but the pipeline
                # hasn't reached yet (e.g. mid-run), not just for a
                # playlist that's never been processed at all.
                "description": json.loads(row["description"])["description"] if row["description"] else None,
            }
            for row in rows
        ]
        return {"songs": songs}

    try:
        items = fetch_playlist_items(playlist_id, user_id)
    except SpotifyException as exc:
        # Confirmed real, not a bug in this endpoint: reproduced the same
        # 403 directly against spotipy's own official playlist_items()
        # helper, unmodified, for a specific real playlist (a label-owned
        # public playlist, "ATMA Classique") - Spotify itself blocks
        # third-party item-level access to some playlists' contents even
        # though the playlist's own metadata (name, owner, public status)
        # is readable. process_playlist already handles this gracefully
        # via its own broad try/except (marks the playlist "failed") -
        # this endpoint had no equivalent handling at all before this,
        # so the same real failure surfaced as an unhandled 500 instead
        # of a clear answer.
        if exc.http_status == 403:
            raise HTTPException(
                status_code=403,
                detail="Spotify won't allow this playlist's songs to be read via the API, even though it's visible in your library.",
            ) from exc
        raise HTTPException(status_code=502, detail="Spotify's API returned an error fetching this playlist.") from exc

    songs = []
    for entry in items:
        track = entry.get("item")
        if track is None or track.get("id") is None:
            continue
        artist_names = [a["name"] for a in track["artists"] if a.get("name")]
        songs.append({
            "track_id": track["id"],
            "name": track["name"],
            "artist": ", ".join(artist_names),
            "genre_bucket": None,
            "description": None,
        })
    return {"songs": songs}


@app.post("/queue")
def add_to_queue(track_id: str, user_id: str = Depends(get_current_user_id)):
    # Deliberately doesn't catch failures into a "soft" success here - if
    # this returns anything but 200, the frontend is meant to actually
    # try the fallback (POST /add-to-playlist), not silently swallow the
    # problem. The most common real failure, confirmed directly while
    # building this: a 404 with reason NO_ACTIVE_DEVICE - queueing needs
    # a live Spotify Connect session somewhere (phone, desktop app, web
    # player), not just this app being open.
    try:
        queue_track(f"spotify:track:{track_id}", user_id)
    except SpotifyException as exc:
        raise HTTPException(status_code=503, detail="Couldn't queue - nothing seems to be actively playing right now.") from exc
    return {"queued": True}


@app.post("/add-to-playlist")
def add_to_playlist(track_id: str, playlist_id: str, user_id: str = Depends(get_current_user_id)):
    # The fallback path when queueing doesn't work, revised: rather than
    # one fixed auto-created playlist, the frontend shows a real picker
    # (backed by the same GET /playlists this app already has) and the
    # user chooses which of their own existing playlists to use.
    add_tracks_to_playlist(playlist_id, [f"spotify:track:{track_id}"], user_id=user_id)
    return {"status": "added"}


@app.post("/playlists/new")
def create_new_playlist(name: str, track_id: str, user_id: str = Depends(get_current_user_id)):
    # The other half of the picker: create a brand new playlist on the
    # spot and add this song to it in one action, rather than making the
    # user create it on Spotify first and come back.
    playlist = create_playlist(name, user_id=user_id)
    add_tracks_to_playlist(playlist["id"], [f"spotify:track:{track_id}"], user_id=user_id)
    return {"id": playlist["id"], "name": playlist["name"]}


# Mounted last and deliberately last - Starlette matches routes in
# registration order, and this mount is a catch-all for "/" that would
# otherwise shadow every route defined above it. html=True serves
# FRONTEND_DIR/index.html for "/" itself, not just exact file paths.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
