from __future__ import annotations

import json
import re

import numpy as np
from anthropic import Anthropic

from config import get_settings
from db.database import get_connection
from pipeline.embed import cosine_similarity, get_model

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


# Common ways people actually type each canonical genre into a search box -
# not exhaustive, but covers the obvious spellings. Each alias is stored
# pre-tokenized (see detect_genre_mention) so "k-pop", "k pop", and "kpop"
# all match the same way regardless of punctuation/spacing.
GENRE_ALIASES: dict[str, list[tuple[str, ...]]] = {
    "Hip-Hop/Rap": [("hip", "hop"), ("hiphop",), ("rap",)],
    "Pop": [("pop",)],
    "R&B/Soul": [("r", "b"), ("rnb",), ("soul",)],
    "Rock": [("rock",)],
    "Alternative/Indie": [("alternative",), ("indie",)],
    "Electronic/Dance": [("electronic",), ("edm",), ("dance",)],
    "K-Pop": [("k", "pop"), ("kpop",), ("korean", "pop")],
    "Jazz": [("jazz",)],
    "Metal": [("metal",)],
    "Reggae": [("reggae",)],
    "Folk": [("folk",)],
    "Latin": [("latin",)],
    "Classical": [("classical",)],
    "Country": [("country",)],
    "Afrobeats": [("afrobeats",), ("afrobeat",)],
    "Soundtrack": [("soundtrack",)],
    "Singer-Songwriter": [("singer", "songwriter"), ("singersongwriter",)],
    "Trap": [("trap",)],
    "Drill": [("drill",)],
    "Grime": [("grime",)],
    "Jerk": [("jerk",)],
}

# A known word/phrase buried in a longer sentence only counts as a real
# request if it sits next to a word that signals "give me music" - "rock
# songs" clearly means the genre; "rock" alone deep in an unrelated
# sentence might not. Confirmed necessary: "songs similar to michael
# jackson" scored Jazz as its best-matching genre bucket by pure
# embedding similarity (0.436, a 0.040 margin over the runner-up) despite
# never mentioning a genre at all - a false positive an embedding-
# similarity threshold alone can't rule out, since there's no way to
# distinguish "this text resembles jazz" from "this text names jazz"
# using similarity scores by themselves. Matching literal words against a
# small, known vocabulary sidesteps that entirely. Shared by artist
# detection too (see search.hybrid.detect_artist_mention) - the same
# reasoning and the same failure modes apply to "does this query name one
# specific known artist," just against a different, larger vocabulary.
MENTION_ANCHOR_WORDS = {
    "song", "songs", "music", "track", "tracks", "tune", "tunes",
    "vibe", "vibes", "anthem", "anthems", "banger", "bangers",
    "hit", "hits", "jam", "jams", "playlist", "genre", "genres",
}
MENTION_ANCHOR_WINDOW = 2  # words to either side of a mention that still count as "adjacent"
MENTION_SHORT_QUERY_WORDS = 3  # a query this short or shorter needs no anchor word at all


def match_known_phrase(query: str, vocabulary: dict[tuple[str, ...], str]) -> str | None:
    """Generic literal-phrase-in-query matcher - not by semantic
    similarity, which is the wrong tool for a question with a discrete
    right answer ("is this specific known word/phrase present") and is
    vulnerable to both false negatives (the target's own signal diluted
    by other similarly-flavored words nearby, e.g. "gentle" pulling
    toward Folk/R&B even in "gentle rock songs") and false positives (see
    the "songs similar to michael jackson" example above).

    vocabulary maps each candidate's own pre-tokenized form to whatever
    should be returned on a match (a canonical genre name, an artist's
    display name, ...) - callers build this from their own source (a
    fixed alias table for genre, the corpus's actual artist list for
    artist), this function only does the matching.

    A short query ("rock", "some rock") is name enough on its own - this
    is a music search tool, so a bare mention has nowhere else to mean. A
    mention inside a longer sentence only counts if it sits near a
    request word like "songs" or "music", so a real word buried in
    unrelated prose - or a homograph like "trap"/"jerk" used in its
    everyday, non-musical sense - doesn't fire by accident. Returns the
    leftmost match in the query if more than one candidate matches."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    short_query = len(words) <= MENTION_SHORT_QUERY_WORDS

    for i in range(len(words)):
        for tokens, value in vocabulary.items():
            n = len(tokens)
            if tuple(words[i:i + n]) != tokens:
                continue
            if short_query:
                return value
            window = words[max(0, i - MENTION_ANCHOR_WINDOW):i] + words[i + n:i + n + MENTION_ANCHOR_WINDOW]
            if any(w in MENTION_ANCHOR_WORDS for w in window):
                return value
    return None


def detect_genre_mention(query: str) -> str | None:
    """Returns a canonical genre name if the query literally names one -
    see match_known_phrase for the matching rules."""
    vocabulary = {alias: canonical for canonical, aliases in GENRE_ALIASES.items() for alias in aliases}
    return match_known_phrase(query, vocabulary)


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
        similarities = [cosine_similarity(vec, bv) for bv in bucket_vecs]
        best_bucket = bucket_names[int(np.argmax(similarities))]
        conn.execute("UPDATE songs SET genre_bucket = ? WHERE track_id = ?", (best_bucket, track_id))

    conn.commit()
    conn.close()
    return len(rows)
