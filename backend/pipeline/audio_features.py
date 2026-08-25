# Deferred type-hint evaluation - needed because this file uses "str |
# None" union syntax, which the installed Python 3.9 can't evaluate at
# runtime (that syntax needs 3.10+); this future-import defers evaluating
# annotations so they're just treated as strings and never crash.
from __future__ import annotations

import json
import re
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import requests

from db.database import get_connection

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_MIN_INTERVAL_SECONDS = 3.5  # stay under the 20 req/min rate limit
DURATION_TOLERANCE_MS = 3000  # how close a candidate's length must be to Spotify's duration_ms to count as the same recording
MIN_NAME_SIMILARITY = 0.5  # reject a match even if duration matches, if the title/artist text is too dissimilar

# Module-level variable - persists across function calls for as long as
# the program runs, which is how _throttle_itunes remembers "when did I
# last call iTunes" across many calls in a loop.
_last_itunes_call = 0.0


def _throttle_itunes() -> None:
    # "global" is required to modify a module-level variable from inside
    # a function - without it, this would create a new local variable
    # instead of updating the shared one.
    global _last_itunes_call
    # time.monotonic() only ever moves forward (unaffected by system
    # clock changes), which matters for measuring elapsed durations.
    elapsed = time.monotonic() - _last_itunes_call
    if elapsed < ITUNES_MIN_INTERVAL_SECONDS:
        # Sleep just long enough to reach the minimum spacing between
        # calls - keeps us under iTunes' 20 requests/minute limit
        # (60s / 20 = 3s minimum; 3.5s adds a small safety margin).
        time.sleep(ITUNES_MIN_INTERVAL_SECONDS - elapsed)
    _last_itunes_call = time.monotonic()


def _normalize(text: str) -> str:
    text = text.lower()
    # Strip "(Live)", "[Remix]"-style suffixes; the *? makes the regex
    # non-greedy so it stops at the first closing bracket, not the last.
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)
    # Strip "feat. Someone" and everything after it.
    text = re.sub(r"feat\.?.*", "", text)
    # Strip remaining punctuation, keeping only letters/digits/spaces.
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    # difflib.SequenceMatcher is part of Python's standard library (no
    # extra dependency). .ratio() returns 0.0-1.0 for how similar two
    # strings are, based on the length of matching subsequences -
    # tolerant of small differences in spacing/capitalization/wording
    # after normalization.
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def find_itunes_match(track_name: str, artist: str, duration_ms: int) -> dict | None:
    # Search on the primary artist only - "Kanye West, Pusha T" as a
    # literal search term matches worse than "Kanye West" alone.
    primary_artist = artist.split(",")[0].strip()
    _throttle_itunes()
    resp = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": f"{primary_artist} {track_name}", "entity": "song", "limit": 5},
        timeout=10,
    )
    resp.raise_for_status()
    candidates = resp.json().get("results", [])

    # "Track the best candidate seen so far" loop: score every candidate,
    # keep whichever scores highest.
    best = None
    best_score = 0.0
    for c in candidates:
        # Skip anything with no preview to analyze, or no duration to
        # compare against.
        if not c.get("previewUrl") or not c.get("trackTimeMillis"):
            continue
        # Hard filter: reject candidates whose length differs too much -
        # this is what distinguishes the correct version of a song from
        # an acoustic version, radio edit, or remix with the same title.
        if abs(c["trackTimeMillis"] - duration_ms) > DURATION_TOLERANCE_MS:
            continue
        # Soft score: average of title-similarity and artist-similarity,
        # used to pick the best among candidates that already passed the
        # duration filter.
        score = (_similarity(c.get("trackName", ""), track_name) + _similarity(c.get("artistName", ""), primary_artist)) / 2
        if score > best_score:
            best_score = score
            best = c

    # Final sanity check: even the "best" candidate gets rejected if it's
    # still a weak match, rather than accepting a bad guess.
    if best is None or best_score < MIN_NAME_SIMILARITY:
        return None
    return best


def extract_features(preview_url: str) -> dict:
    audio_bytes = requests.get(preview_url, timeout=15).content
    # librosa.load can't decode this AAC/m4a audio from an in-memory
    # buffer without ffmpeg installed - it needs a real file on disk to
    # hand to the system's native decoder. NamedTemporaryFile creates one;
    # delete=False keeps it from being auto-deleted when the "with" block
    # exits, since librosa still needs to open it afterward.
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        # y = the audio waveform itself, as a NumPy array of amplitude
        # samples. sr = the sample rate (samples per second). sr=None
        # means "use the file's native rate" rather than resampling.
        y, sr = librosa.load(tmp_path, sr=None)
    finally:
        # try/finally guarantees the temp file is deleted whether loading
        # succeeds or raises an exception. missing_ok avoids an error if
        # it's already gone somehow.
        Path(tmp_path).unlink(missing_ok=True)

    # Genuine signal-processing, not a lookup: analyzes the waveform for
    # periodic onset patterns to estimate tempo in beats per minute.
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # RMS ("root mean square") is a standard measure of average
    # loudness/intensity, computed in short windows across the track,
    # then averaged into one number.
    rms = float(np.mean(librosa.feature.rms(y=y)))
    # The frequency spectrum's "center of mass" - a track dominated by
    # high frequencies (cymbals, bright synths) scores higher than one
    # dominated by bass/low mids. A genuine proxy for perceived brightness.
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Harmonic-percussive source separation: splits the waveform into two
    # new waveforms - one mostly tonal/sustained content (melody, chords),
    # one mostly transient/percussive content (drum hits).
    harmonic, percussive = librosa.effects.hpss(y)
    # Summing squared amplitude gives an energy measure for each part;
    # the ratio says which one dominates the mix. The tiny +1e-9 avoids a
    # divide-by-zero crash on a silent track.
    harmonic_energy = float(np.sum(harmonic**2))
    percussive_energy = float(np.sum(percussive**2))
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-9)

    return {
        "tempo_bpm": round(float(tempo), 1),
        "energy_rms": round(rms, 4),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "harmonic_ratio": round(harmonic_ratio, 3),
    }


def process_song(track_name: str, artist: str, duration_ms: int) -> dict:
    match = find_itunes_match(track_name, artist, duration_ms)
    if match is None:
        return {"status": "no_itunes_match"}
    try:
        features = extract_features(match["previewUrl"])
    except Exception as exc:
        # Catch-all so one bad/corrupt preview doesn't crash the whole
        # batch run - the failure gets recorded per-song instead.
        return {"status": "extraction_failed", "error": str(exc)}
    features["status"] = "ok"
    # primaryGenreName comes free in the same iTunes response used for
    # matching above - real genre metadata at zero extra API cost.
    features["itunes_genre"] = match.get("primaryGenreName")
    return features


def fetch_and_store_audio_features(limit: int | None = None) -> dict:
    conn = get_connection()
    # Only process rows that haven't been done yet - makes this safely
    # re-runnable/resumable if it's interrupted partway through.
    query = "SELECT track_id, name, artist, duration_ms FROM songs WHERE audio_features IS NULL"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    counts = {"ok": 0, "no_itunes_match": 0, "extraction_failed": 0}
    for row in rows:
        result = process_song(row["name"], row["artist"], row["duration_ms"])
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        # SQLite has no native JSON column type, so the result dict is
        # serialized to a JSON string and stored in the TEXT column -
        # json.loads() reverses this whenever the data is read back.
        conn.execute(
            "UPDATE songs SET audio_features = ? WHERE track_id = ?",
            (json.dumps(result), row["track_id"]),
        )
        # Commit after every row (not just at the end) so progress
        # survives if the process is killed partway through a long run.
        conn.commit()

    conn.close()
    return counts


def backfill_genre() -> int:
    """For rows processed before itunes_genre was captured: re-run just the cheap
    iTunes search (no audio download/Librosa) to fill in the missing field."""
    conn = get_connection()
    rows = conn.execute("SELECT track_id, name, artist, duration_ms, audio_features FROM songs WHERE audio_features IS NOT NULL").fetchall()

    count = 0
    for row in rows:
        data = json.loads(row["audio_features"])
        # Skip rows that failed/had no match, and rows that already have
        # genre from a run after this field was added.
        if data.get("status") != "ok" or "itunes_genre" in data:
            continue
        match = find_itunes_match(row["name"], row["artist"], row["duration_ms"])
        data["itunes_genre"] = match.get("primaryGenreName") if match else None
        conn.execute("UPDATE songs SET audio_features = ? WHERE track_id = ?", (json.dumps(data), row["track_id"]))
        conn.commit()
        count += 1

    conn.close()
    return count


def compute_thresholds(conn) -> dict:
    """Percentile cutoffs (tertiles) computed from the actual corpus, so 'high energy'
    means high relative to this playlist rather than an arbitrarily guessed number."""
    rows = conn.execute("SELECT audio_features FROM songs WHERE audio_features IS NOT NULL").fetchall()
    parsed = [json.loads(r["audio_features"]) for r in rows]
    ok = [f for f in parsed if f.get("status") == "ok"]

    def tertiles(key: str) -> tuple[float, float]:
        values = np.array([f[key] for f in ok])
        # The value below which 33%/66% of the corpus falls - bucketing
        # each song against these (rather than numbers guessed up front)
        # is what makes "high energy" a real, discriminating signal
        # instead of a label nearly every track gets.
        return float(np.percentile(values, 33)), float(np.percentile(values, 66))

    return {
        "tempo_bpm": tertiles("tempo_bpm"),
        "energy_rms": tertiles("energy_rms"),
        "spectral_centroid_hz": tertiles("spectral_centroid_hz"),
        "harmonic_ratio": tertiles("harmonic_ratio"),
    }


def describe_features(features: dict, thresholds: dict) -> str:
    # Shared bucketing logic for all four features: below the lower
    # cutoff = "low" word, between cutoffs = "mid" word, above = "high".
    def bucket(value: float, cutoffs: tuple[float, float], low: str, mid: str, high: str) -> str:
        lo, hi = cutoffs
        return low if value < lo else mid if value < hi else high

    tempo_word = bucket(features["tempo_bpm"], thresholds["tempo_bpm"], "slow tempo", "moderate tempo", "fast tempo")
    energy_word = bucket(features["energy_rms"], thresholds["energy_rms"], "low energy", "medium energy", "high energy")
    timbre_word = bucket(
        features["spectral_centroid_hz"], thresholds["spectral_centroid_hz"],
        "warm, mellow timbre", "balanced timbre", "bright, sharp timbre",
    )
    texture_word = bucket(
        features["harmonic_ratio"], thresholds["harmonic_ratio"],
        "rhythm/percussion-dominant mix", "balanced mix of tone and rhythm", "tonal/melodic-dominant mix",
    )

    return f"{tempo_word}, {energy_word}, {timbre_word}, {texture_word}"


def generate_descriptions() -> int:
    conn = get_connection()
    thresholds = compute_thresholds(conn)
    rows = conn.execute("SELECT track_id, audio_features FROM songs WHERE audio_features IS NOT NULL").fetchall()

    count = 0
    for row in rows:
        data = json.loads(row["audio_features"])
        if data.get("status") != "ok":
            continue
        # Re-describes every "ok" row using corpus-wide thresholds, not
        # just newly-added ones - needed because thresholds themselves
        # change as more songs get processed, so old descriptions could
        # otherwise go stale relative to the full corpus.
        data["description"] = describe_features(data, thresholds)
        conn.execute("UPDATE songs SET audio_features = ? WHERE track_id = ?", (json.dumps(data), row["track_id"]))
        count += 1

    conn.commit()
    conn.close()
    return count
