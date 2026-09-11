from __future__ import annotations

from datetime import datetime, timezone

from auth.spotify_oauth import get_spotify_client
from db.database import get_connection
from db.models import init_db, link_tracks_to_playlist, upsert_playlist
from pipeline.audio_features import fetch_and_store_audio_features, generate_descriptions
from pipeline.describe import fetch_and_store_descriptions
from pipeline.embed import embed_and_store
from pipeline.fetch_playlist import fetch_playlist_items, save_tracks
from pipeline.genre_buckets import assign_genre_buckets
from pipeline.lyrics import fetch_and_store_lyrics

# A playlist's processing_status is one of: the states here, plus one
# in-progress stage name per entry in process_playlist's own `stages`
# list below (fetching/audio_features/generating_descriptions/lyrics/
# describing/embedding/genre_bucketing). TERMINAL_STATUSES is what
# main.py's POST /playlists/{id}/process checks before starting a new
# background run - anything NOT in this set means a run is already
# in flight for that playlist, so a second click/request is rejected
# rather than starting an overlapping second pass against the same
# database file.
TERMINAL_STATUSES = {"not_started", "complete", "failed"}


def _completed_count(user_id: str | None, playlist_id: str, column: str) -> int:
    """How many of this playlist's tracks have cleared a given pipeline
    stage so far - the join through playlist_songs is what scopes a
    corpus-wide column (songs.{column}) down to just this one playlist,
    since songs itself has no idea which playlist(s) it belongs to."""
    conn = get_connection(user_id)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS n FROM songs
        JOIN playlist_songs ON songs.track_id = playlist_songs.track_id
        WHERE playlist_songs.playlist_id = ? AND songs.{column} IS NOT NULL
        """,  # noqa: S608 - column is one of a fixed set of literal names below, never user input
        (playlist_id,),
    ).fetchone()
    conn.close()
    return row["n"]


def process_playlist(user_id: str, playlist_id: str) -> None:
    """The orchestration entry point that didn't exist anywhere before
    Step 3 - none of the six stage functions below call each other, and
    scripts/run_pipeline.py (the obvious place for this) has been empty
    since the project's first commit. Runs as a FastAPI BackgroundTask
    (see main.py's POST /playlists/{id}/process) - not awaited by the
    request that triggers it, so a real playlist genuinely taking several
    minutes never blocks that request.

    Stage order is a real dependency chain, confirmed by reading each
    stage's source before writing this, not assumed from the function
    names alone: audio_features must run before generate_descriptions
    (which summarizes it into a plain-language phrase); that phrase and
    lyrics must both be ready before fetch_and_store_descriptions (the
    LLM stage), which reads them out of the audio_features JSON and the
    lyrics column respectively; embed_and_store and assign_genre_buckets
    both then need that LLM description to exist.

    Every stage function already only processes rows still NULL in its
    own column - confirmed by reading each one's WHERE clause, not
    assumed. So a song already fully processed from an earlier playlist
    is automatically skipped by every stage below, with no special-case
    "have I seen this before" code needed here at all - the only
    genuinely new work per playlist is recording membership in
    playlist_songs and checkpointing progress in `playlists`.

    A stage function throwing an unhandled exception marks the whole
    playlist 'failed' rather than leaving it stuck in 'processing'
    forever, then re-raises so the real traceback still surfaces (in
    uvicorn's logs, since there's no request left to attach a response
    to by the time a background task runs). This is deliberately the
    only failure handling done here: every stage already has its own
    internal per-song resilience (e.g. fetch_and_store_audio_features
    catches and skips a single song's network error without aborting the
    rest of its own run) - re-implementing that per-song logic here would
    duplicate what each stage function is already tested to do itself.
    Re-running process_playlist after a failure is always safe - the
    NULL-only filtering means it picks up exactly where it left off.
    """
    sp = get_spotify_client(user_id)
    # Spotify's Feb 2026 migration (see fetch_playlist_items's own
    # comment on the /tracks -> /items endpoint rename) renamed this
    # metadata field too - confirmed directly against the live API
    # before writing this, not assumed from the old shape: playlist
    # objects now nest track count under "items.total", not "tracks.total".
    playlist_meta = sp.playlist(playlist_id, fields="name,items.total")

    init_db(user_id)
    upsert_playlist(
        playlist_id,
        user_id,
        name=playlist_meta["name"],
        total_tracks=playlist_meta["items"]["total"],
        processing_status="fetching",
    )

    try:
        items = fetch_playlist_items(playlist_id, user_id)
        # Same None/local-file/missing-id filtering save_tracks itself
        # applies - track_ids collected here must match exactly what
        # save_tracks actually wrote, or linking would reference a
        # track_id that was never saved to `songs` at all.
        track_ids = [
            entry["item"]["id"]
            for entry in items
            if entry.get("item") is not None and entry["item"].get("id") is not None
        ]
        save_tracks(items, user_id)
        link_tracks_to_playlist(track_ids, playlist_id, user_id)

        # (status label, the call itself, the column that measures this
        # stage's progress - None where "populated" isn't a meaningful
        # completion signal, e.g. lyrics is legitimately NULL forever for
        # a genuinely instrumental track, not a sign anything is stuck).
        stages = [
            ("audio_features", lambda: fetch_and_store_audio_features(user_id=user_id), "audio_features"),
            ("generating_descriptions", lambda: generate_descriptions(user_id=user_id), None),
            ("lyrics", lambda: fetch_and_store_lyrics(user_id=user_id), None),
            ("describing", lambda: fetch_and_store_descriptions(user_id=user_id), "description"),
            ("embedding", lambda: embed_and_store(user_id=user_id), "embedding"),
            ("genre_bucketing", lambda: assign_genre_buckets(user_id=user_id), "genre_bucket"),
        ]
        for status_label, run_stage, progress_column in stages:
            upsert_playlist(playlist_id, user_id, processing_status=status_label)
            run_stage()
            if progress_column is not None:
                upsert_playlist(
                    playlist_id, user_id,
                    processed_count=_completed_count(user_id, playlist_id, progress_column),
                )

        upsert_playlist(
            playlist_id,
            user_id,
            processing_status="complete",
            # embedding, not genre_bucket, is the true "this song is
            # actually searchable" gate - it's exactly what
            # search.hybrid._fetch_songs filters on.
            processed_count=_completed_count(user_id, playlist_id, "embedding"),
            last_processed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        upsert_playlist(playlist_id, user_id, processing_status="failed")
        raise
