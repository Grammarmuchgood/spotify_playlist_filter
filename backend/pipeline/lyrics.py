from __future__ import annotations

import re
import time
from urllib.parse import quote

import requests

from db.database import get_connection
from pipeline.http_utils import get_with_retry

LYRICS_OVH_URL = "https://api.lyrics.ovh/v1"
LYRICS_MIN_INTERVAL_SECONDS = 1.0  # no published rate limit, but be a good citizen of a free volunteer-run service

# Confirmed bug: a trailing " - Remastered 2015" / " - Radio Edit" / etc.
# suffix makes the search return 404 even though the song is indexed under
# its base title (verified directly: "Hey Jude - Remastered 2015" -> 404,
# "Hey Jude" -> 200). Strips everything from the first " - " onward.
SUFFIX_PATTERN = re.compile(r"\s+-\s+.+$")

# Two more confirmed bugs, verified directly the same way: punctuation in
# the title breaks the search ("Ain't No Sunshine" -> 404, "Aint No
# Sunshine" -> 200; "N.Y. State of Mind" -> 404, "NY State of Mind" -> 200),
# and a leading "The " in the artist name breaks it too ("The Temptations"
# -> 404, "Temptations" -> 200).
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
LEADING_THE_PATTERN = re.compile(r"^the\s+", re.IGNORECASE)

_last_lyrics_call = 0.0


def _throttle_lyrics() -> None:
    global _last_lyrics_call
    elapsed = time.monotonic() - _last_lyrics_call
    if elapsed < LYRICS_MIN_INTERVAL_SECONDS:
        time.sleep(LYRICS_MIN_INTERVAL_SECONDS - elapsed)
    _last_lyrics_call = time.monotonic()


def _fetch_lyrics_raw(track_name: str, primary_artist: str) -> str | None:
    url = f"{LYRICS_OVH_URL}/{quote(primary_artist)}/{quote(track_name)}"
    resp = get_with_retry(_throttle_lyrics, url)
    if resp.status_code == 404:
        # A real, meaningful answer from the API ("no lyrics found" for
        # this title/artist pair) - not a failure to retry or raise on.
        return None
    resp.raise_for_status()
    return resp.json().get("lyrics")


def fetch_lyrics(track_name: str, primary_artist: str) -> str | None:
    # Takes the primary artist directly (stored at fetch time from
    # Spotify's own artist list) rather than deriving it here by splitting
    # a joined artist string - that split silently breaks for any artist
    # whose own name contains a comma (e.g. "Tyler, The Creator"), since
    # there's no way to tell "a comma separating two artists" apart from
    # "a comma inside one artist's name" after they've already been joined.

    # Progressively more aggressive attempts, each only added when it would
    # actually change something - stops at the first one that succeeds.
    # Ordered from least to most aggressive so a title that already works
    # exactly as-is never gets an unnecessary extra request.
    title = track_name
    candidates = [(title, primary_artist)]

    no_suffix = SUFFIX_PATTERN.sub("", title)
    if no_suffix != title:
        candidates.append((no_suffix, primary_artist))
        title = no_suffix

    no_punctuation = PUNCTUATION_PATTERN.sub("", title)
    no_the = LEADING_THE_PATTERN.sub("", primary_artist)
    if no_punctuation != title or no_the != primary_artist:
        candidates.append((no_punctuation, no_the))

    for candidate_title, candidate_artist in candidates:
        lyrics = _fetch_lyrics_raw(candidate_title, candidate_artist)
        if lyrics:
            return lyrics
    return None


def fetch_and_store_lyrics(limit: int | None = None) -> dict:
    conn = get_connection()
    # Only rows never attempted - '' (set below for "confirmed no lyrics
    # found") is NOT NULL, so it's correctly skipped on future runs rather
    # than being re-queried forever.
    query = "SELECT track_id, name, artist, primary_artist FROM songs WHERE lyrics IS NULL"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    counts = {"found": 0, "not_found": 0, "skipped_network_error": 0}
    for row in rows:
        try:
            lyrics = fetch_lyrics(row["name"], row["primary_artist"] or row["artist"].split(",")[0].strip())
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
