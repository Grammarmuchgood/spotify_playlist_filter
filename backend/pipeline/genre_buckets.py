from __future__ import annotations

import json

import numpy as np
from anthropic import Anthropic

from config import get_settings
from db.database import get_connection
from pipeline.embed import get_model

RECLASSIFY_MODEL = "claude-haiku-4-5"

# A small, fixed taxonomy - not a keyword-matching dictionary. The actual
# matching between messy real genre strings ("Hip-Hop/Rap", "trap", "east
# coast hip hop", "drill"...) and these buckets happens via embedding
# similarity, not hardcoded rules. Derived from the distinct genre labels
# actually present in this library (37 iTunes + 26 MusicBrainz labels).
CANONICAL_GENRES = [
    "Hip-Hop/Rap", "Pop", "R&B/Soul", "Rock", "Alternative/Indie",
    "Electronic/Dance", "K-Pop", "Jazz", "Metal", "Reggae", "Folk",
    "Latin", "Classical", "Country", "Afrobeats", "Soundtrack",
    "Singer-Songwriter",
    # Confirmed via direct testing: these four are short, common-English-word
    # genre terms that get embedding-mismatched to unrelated buckets when
    # forced to bridge to a differently-worded parent bucket - "jerk" alone
    # scored higher against "Jazz" (0.569) than "Hip-Hop/Rap" (0.526), since
    # an isolated ambiguous word has no context to disambiguate its niche
    # musical sense. Adding them as their own buckets fixes this because a
    # term matching itself is a far more reliable comparison than bridging
    # to an unrelated name - confirmed "trap" then scores 0.932 against its
    # own bucket vs. 0.581 against the wrong one (Rock) it used to hit.
    "Trap", "Drill", "Grime", "Jerk",
]

_bucket_embeddings: np.ndarray | None = None


def get_bucket_embeddings() -> np.ndarray:
    global _bucket_embeddings
    if _bucket_embeddings is None:
        _bucket_embeddings = get_model().encode(CANONICAL_GENRES)
    return _bucket_embeddings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _genre_text_for(audio_features_json: str | None, description_json: str | None) -> str:
    # Prefer the real structured genre (iTunes, then MusicBrainz) when one
    # exists. Only for the ~32 songs with neither does this fall back to
    # the song's own generated description text - a weaker signal, but
    # not zero signal.
    if audio_features_json:
        data = json.loads(audio_features_json)
        genre = data.get("itunes_genre") or data.get("musicbrainz_genre")
        if genre:
            return genre
    if description_json:
        return json.loads(description_json)["description"]
    return "unknown"


def reclassify_with_llm(track_ids: list[str]) -> int:
    """Targeted, cheap fix for specific known-mismatched tracks - not a full
    corpus pass. Uses the song's own generated description (built from that
    track's real lyrics/audio) as context, not the raw genre string, so it's
    unaffected by upstream problems like a wrong-artist MusicBrainz match or
    an artist-level tag that doesn't describe this specific song."""
    from pydantic import BaseModel

    class GenreClassification(BaseModel):
        genre: str

    conn = get_connection()
    client = Anthropic(api_key=get_settings().anthropic_api_key)
    options = ", ".join(CANONICAL_GENRES)

    count = 0
    for track_id in track_ids:
        row = conn.execute("SELECT name, artist, description FROM songs WHERE track_id = ?", (track_id,)).fetchone()
        if row is None or row["description"] is None:
            continue
        desc = json.loads(row["description"])["description"]

        response = client.messages.parse(
            model=RECLASSIFY_MODEL,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Song vibe description: {desc}\n\n"
                    f"Pick the single best-fitting genre for this song from exactly this list: {options}\n"
                    "Respond with only the genre name, spelled exactly as given in the list."
                ),
            }],
            output_format=GenreClassification,
        )
        genre = response.parsed_output.genre
        if genre not in CANONICAL_GENRES:
            # Model didn't return an exact match from the list - skip
            # rather than write an invalid bucket value.
            continue
        conn.execute("UPDATE songs SET genre_bucket = ? WHERE track_id = ?", (genre, track_id))
        conn.commit()
        count += 1

    conn.close()
    return count


def assign_genre_buckets() -> int:
    conn = get_connection()
    rows = conn.execute(
        "SELECT track_id, audio_features, description FROM songs WHERE genre_bucket IS NULL"
    ).fetchall()
    if not rows:
        conn.close()
        return 0

    bucket_names = CANONICAL_GENRES
    bucket_vecs = get_bucket_embeddings()

    track_ids = [row["track_id"] for row in rows]
    texts = [_genre_text_for(row["audio_features"], row["description"]) for row in rows]

    model = get_model()
    text_vecs = model.encode(texts)

    for track_id, vec in zip(track_ids, text_vecs):
        similarities = [_cosine_similarity(vec, bv) for bv in bucket_vecs]
        best_bucket = bucket_names[int(np.argmax(similarities))]
        conn.execute("UPDATE songs SET genre_bucket = ? WHERE track_id = ?", (best_bucket, track_id))

    conn.commit()
    conn.close()
    return len(rows)
