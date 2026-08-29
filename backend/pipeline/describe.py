from __future__ import annotations

import json
from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel

from config import get_settings
from db.database import get_connection
from pipeline.audio_features import _energy_era_cohort, bucket_tertile, compute_thresholds

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are analyzing songs for a personal music search tool that lets someone find tracks in their own playlist by describing a vibe in plain language (e.g. "chill rainy drive", "gym music", "sad but upbeat"). For each song you're given whatever data is available - audio characteristics, genre, release era, and lyrics - and you produce a short vibe description plus a few structured tags. The description becomes a search embedding, so it needs to read like a natural, focused description of a vibe - not a technical summary, and not a lyrical plot recap."""

RULES = """Rules:
- If the production/audio feel and the lyrical mood pull in different directions (e.g. upbeat instrumental over sad lyrics), say so explicitly rather than averaging into a vague middle.
- Do not mention the artist name or track title in your output.
- If lyrics are unavailable, say so plainly (e.g. note it's likely instrumental) and base the description on audio + genre only.
- If audio characteristics are unavailable, base the description on lyrics + genre + era only.
- If very little information is available at all, keep the description general and hedged rather than inventing specific detail you don't have."""


class SongDescription(BaseModel):
    mood: str
    context_tags: list[str]
    energy: Literal["low", "medium", "high"]
    description: str


def _audio_context(
    audio_features_json: str | None, energy_thresholds_by_cohort: dict[str, tuple[float, float]], year: int | None
) -> tuple[str | None, str | None, str | None]:
    """Returns (genre, audio_description, code_computed_energy). code_computed_energy
    is None when there's no real audio data - the LLM is asked to estimate it itself
    only in that case, since a real signal-processing measurement is always preferred
    over an LLM guess when one exists."""
    if not audio_features_json:
        return None, None, None
    data = json.loads(audio_features_json)
    if data.get("status") == "ok":
        genre = data.get("itunes_genre")
        audio_description = data.get("description")
        # Bucketed against this song's own release-era cohort, not the
        # pooled corpus - see audio_features.compute_thresholds for why
        # (energy_rms tracks mastering-loudness era far more than it
        # tracks actual musical energy).
        cutoffs = energy_thresholds_by_cohort[_energy_era_cohort(year)]
        code_energy = bucket_tertile(data["energy_rms"], cutoffs)
        return genre, audio_description, code_energy
    return data.get("musicbrainz_genre"), None, None


def build_prompt(track_name: str, artist: str, genre: str | None, year: str | None, audio_description: str | None, lyrics: str | None) -> str:
    audio_block = audio_description or "no audio data available for this track"
    lyrics_block = lyrics or "No lyrics available - likely instrumental, or not found."
    return f"""Song: "{track_name}"
Artist: {artist}   [for your context only - do not name the artist or track title in your output]
Genre: {genre or "unknown"}
Era: {year or "unknown"}
Audio characteristics (already measured): {audio_block}
Lyrics: {lyrics_block}

{RULES}"""


def describe_song(client: Anthropic, track_name: str, artist: str, genre: str | None, year: str | None, audio_description: str | None, lyrics: str | None) -> SongDescription:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(track_name, artist, genre, year, audio_description, lyrics)}],
        output_format=SongDescription,
    )
    return response.parsed_output


def fetch_and_store_descriptions(limit: int | None = None) -> dict:
    conn = get_connection()
    client = Anthropic(api_key=get_settings().anthropic_api_key)
    energy_thresholds_by_cohort = compute_thresholds(conn)["energy_rms"]

    query = "SELECT track_id, name, artist, release_date, audio_features, lyrics FROM songs WHERE description IS NULL"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    counts = {"ok": 0, "error": 0}
    for row in rows:
        year_int = int(row["release_date"][:4]) if row["release_date"] else None
        genre, audio_description, code_energy = _audio_context(row["audio_features"], energy_thresholds_by_cohort, year_int)
        year = row["release_date"][:4] if row["release_date"] else None
        # '' (from lyrics.py's "confirmed no lyrics found") is treated the
        # same as never having fetched lyrics at all - both mean "nothing
        # to give the model here."
        lyrics = row["lyrics"] if row["lyrics"] else None

        try:
            parsed = describe_song(client, row["name"], row["artist"], genre, year, audio_description, lyrics)
        except Exception:
            # Covers both real API failures (after the SDK's own built-in
            # retries on 429/5xx are exhausted) and a response that fails
            # schema validation - leave description as NULL so a future
            # run retries this row.
            counts["error"] += 1
            continue

        # Prefer the real, signal-processing-measured energy over the
        # LLM's self-reported guess whenever one exists.
        final_energy = code_energy or parsed.energy
        result = {
            "status": "ok",
            "mood": parsed.mood,
            "context_tags": parsed.context_tags,
            "energy": final_energy,
            "description": parsed.description,
        }
        conn.execute("UPDATE songs SET description = ? WHERE track_id = ?", (json.dumps(result), row["track_id"]))
        conn.commit()
        counts["ok"] += 1

    conn.close()
    return counts
