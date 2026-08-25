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
DURATION_TOLERANCE_MS = 3000
MIN_NAME_SIMILARITY = 0.5

_last_itunes_call = 0.0


def _throttle_itunes() -> None:
    global _last_itunes_call
    elapsed = time.monotonic() - _last_itunes_call
    if elapsed < ITUNES_MIN_INTERVAL_SECONDS:
        time.sleep(ITUNES_MIN_INTERVAL_SECONDS - elapsed)
    _last_itunes_call = time.monotonic()


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)
    text = re.sub(r"feat\.?.*", "", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def find_itunes_match(track_name: str, artist: str, duration_ms: int) -> dict | None:
    primary_artist = artist.split(",")[0].strip()
    _throttle_itunes()
    resp = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": f"{primary_artist} {track_name}", "entity": "song", "limit": 5},
        timeout=10,
    )
    resp.raise_for_status()
    candidates = resp.json().get("results", [])

    best = None
    best_score = 0.0
    for c in candidates:
        if not c.get("previewUrl") or not c.get("trackTimeMillis"):
            continue
        if abs(c["trackTimeMillis"] - duration_ms) > DURATION_TOLERANCE_MS:
            continue
        score = (_similarity(c.get("trackName", ""), track_name) + _similarity(c.get("artistName", ""), primary_artist)) / 2
        if score > best_score:
            best_score = score
            best = c

    if best is None or best_score < MIN_NAME_SIMILARITY:
        return None
    return best


def extract_features(preview_url: str) -> dict:
    audio_bytes = requests.get(preview_url, timeout=15).content
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    harmonic, percussive = librosa.effects.hpss(y)
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
        return {"status": "extraction_failed", "error": str(exc)}
    features["status"] = "ok"
    features["itunes_genre"] = match.get("primaryGenreName")
    return features


def fetch_and_store_audio_features(limit: int | None = None) -> dict:
    conn = get_connection()
    query = "SELECT track_id, name, artist, duration_ms FROM songs WHERE audio_features IS NULL"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    counts = {"ok": 0, "no_itunes_match": 0, "extraction_failed": 0}
    for row in rows:
        result = process_song(row["name"], row["artist"], row["duration_ms"])
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        conn.execute(
            "UPDATE songs SET audio_features = ? WHERE track_id = ?",
            (json.dumps(result), row["track_id"]),
        )
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
        return float(np.percentile(values, 33)), float(np.percentile(values, 66))

    return {
        "tempo_bpm": tertiles("tempo_bpm"),
        "energy_rms": tertiles("energy_rms"),
        "spectral_centroid_hz": tertiles("spectral_centroid_hz"),
        "harmonic_ratio": tertiles("harmonic_ratio"),
    }


def describe_features(features: dict, thresholds: dict) -> str:
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
        data["description"] = describe_features(data, thresholds)
        conn.execute("UPDATE songs SET audio_features = ? WHERE track_id = ?", (json.dumps(data), row["track_id"]))
        count += 1

    conn.commit()
    conn.close()
    return count
