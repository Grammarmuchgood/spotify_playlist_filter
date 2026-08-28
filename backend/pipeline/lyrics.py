from __future__ import annotations

import time
from urllib.parse import quote

import requests

from db.database import get_connection
from pipeline.http_utils import get_with_retry

LYRICS_OVH_URL = "https://api.lyrics.ovh/v1"
LYRICS_MIN_INTERVAL_SECONDS = 1.0  # no published rate limit, but be a good citizen of a free volunteer-run service

_last_lyrics_call = 0.0


def _throttle_lyrics() -> None:
    global _last_lyrics_call
    elapsed = time.monotonic() - _last_lyrics_call
    if elapsed < LYRICS_MIN_INTERVAL_SECONDS:
        time.sleep(LYRICS_MIN_INTERVAL_SECONDS - elapsed)
    _last_lyrics_call = time.monotonic()


def fetch_lyrics(track_name: str, artist: str) -> str | None:
    primary_artist = artist.split(",")[0].strip()
    url = f"{LYRICS_OVH_URL}/{quote(primary_artist)}/{quote(track_name)}"
    resp = get_with_retry(_throttle_lyrics, url)
    if resp.status_code == 404:
        # A real, meaningful answer from the API ("no lyrics found" for
        # this title/artist pair) - not a failure to retry or raise on.
        return None
    resp.raise_for_status()
    return resp.json().get("lyrics")


def fetch_and_store_lyrics(limit: int | None = None) -> dict:
    conn = get_connection()
    # Only rows never attempted - '' (set below for "confirmed no lyrics
    # found") is NOT NULL, so it's correctly skipped on future runs rather
    # than being re-queried forever.
    query = "SELECT track_id, name, artist FROM songs WHERE lyrics IS NULL"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    counts = {"found": 0, "not_found": 0, "skipped_network_error": 0}
    for row in rows:
        try:
            lyrics = fetch_lyrics(row["name"], row["artist"])
        except requests.exceptions.RequestException:
            # Retries in get_with_retry are already exhausted by this
            # point - leave lyrics as NULL (don't write anything) so a
            # future run retries this row, rather than wrongly recording
            # a network failure as "confirmed no lyrics."
            counts["skipped_network_error"] += 1
            continue
        counts["found" if lyrics else "not_found"] += 1
        conn.execute(
            "UPDATE songs SET lyrics = ? WHERE track_id = ?",
            (lyrics if lyrics else "", row["track_id"]),
        )
        conn.commit()

    conn.close()
    return counts
