from __future__ import annotations

import json
import re

import numpy as np
from sentence_transformers import CrossEncoder

from db.database import get_connection
from pipeline.embed import cosine_similarity, get_model
from pipeline.genre_buckets import CANONICAL_GENRES, detect_genre_mention, get_bucket_embeddings, match_known_phrase
from pipeline.mood import contradicts_mood, detect_mood_preference

RRF_K = 60  # standard IR-literature default; tunable
SHORTLIST_SIZE = 50  # how many candidates a ranking stage hands to the reranker

# Confirmed unsafe for automatic detection: "fun." tokenizes to the
# single word "fun" after punctuation stripping, and "fun songs" / "fun
# music" is completely ordinary language that has nothing to do with the
# band - the anchor-word check that resolves every other common-word
# collision found in this library (Ghost, Future, Train, Oasis, Player,
# Silver, Smiley, "A-Wall" -> "a wall", ...) can't help here, because
# "songs" sitting right next to "fun" is exactly the anchor pattern, not
# a red flag that rules anything out. Add other artist names here only
# after finding a real, confirmed false positive the same way - most
# artist names are distinctive enough that anchor-word adjacency alone
# handles them fine.
ARTIST_DETECTION_BLOCKLIST = {"fun."}

# The official Qwen/Qwen3-Reranker-0.6B repo isn't set up for
# sentence-transformers' CrossEncoder out of the box - loading it left its
# actual scoring layer randomly initialized (confirmed: a
# "newly initialized: ['score.weight']" warning, meaning it would produce
# meaningless scores). This is a community-converted checkpoint
# specifically fixed for CrossEncoder compatibility - verified directly
# that this one loads with real trained weights and produces sensible,
# well-separated scores.
RERANKER_MODEL_NAME = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def _fetch_songs() -> tuple[list[dict], dict[str, np.ndarray]]:
    """Splits metadata from embeddings on purpose - result dicts get
    handed straight back to callers, and a raw embedding vector has no
    business showing up in search output."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT track_id, name, artist, primary_artist, embedding, genre_bucket, description FROM songs WHERE embedding IS NOT NULL"
    ).fetchall()
    conn.close()

    songs = []
    embeddings = {}
    for row in rows:
        songs.append({
            "track_id": row["track_id"],
            "name": row["name"],
            "artist": row["artist"],
            "primary_artist": row["primary_artist"],
            "genre_bucket": row["genre_bucket"],
            "description": json.loads(row["description"])["description"],
        })
        embeddings[row["track_id"]] = np.array(json.loads(row["embedding"]))
    return songs, embeddings


def detect_artist_mention(query: str, known_artists: set[str]) -> str | None:
    """Returns a primary_artist name if the query literally names one of
    the artists actually in this library, checked by the same
    anchor-word-adjacency word match as genre_buckets.detect_genre_mention
    (see match_known_phrase) - reused rather than duplicated, since the
    underlying question ("does this text mention one specific known
    thing") and its failure modes are identical to genre's.

    Unlike genre, this vocabulary isn't hand-curated - it's built fresh
    from whichever artists are actually in the corpus for this search, so
    it grows and shrinks with the playlist automatically, no maintenance
    needed. ARTIST_DETECTION_BLOCKLIST excludes the one confirmed case
    where that automatic growth isn't safe."""
    vocabulary = {}
    for artist in known_artists:
        if artist in ARTIST_DETECTION_BLOCKLIST:
            continue
        tokens = tuple(re.findall(r"[a-z0-9]+", artist.lower()))
        if tokens:
            vocabulary[tokens] = artist
    return match_known_phrase(query, vocabulary)


def _bucket_scores(query_vector: np.ndarray) -> dict[str, float]:
    bucket_vecs = get_bucket_embeddings()
    return {name: cosine_similarity(query_vector, vec) for name, vec in zip(CANONICAL_GENRES, bucket_vecs)}


def _to_ranks(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {track_id: rank for rank, (track_id, _) in enumerate(ordered, start=1)}


def _with_vibe_scores(songs: list[dict], embeddings: dict[str, np.ndarray], query_vector: np.ndarray) -> list[dict]:
    for song in songs:
        song["vibe_score"] = round(cosine_similarity(query_vector, embeddings[song["track_id"]]), 4)
    return songs


def _filter_by_genre(songs: list[dict], genre: str | None) -> list[dict]:
    if genre is None:
        return songs
    return [s for s in songs if s["genre_bucket"] == genre]


def _filter_by_artist(songs: list[dict], artist: str | None) -> list[dict]:
    if artist is None:
        return songs
    return [s for s in songs if s["primary_artist"] == artist]


def _filter_by_mood(songs: list[dict], mood_preference: str | None) -> list[dict]:
    if mood_preference is None:
        return songs
    return [s for s in songs if not contradicts_mood(s["description"], mood_preference)]


def _rrf_rank(songs: list[dict], bucket_scores: dict[str, float], rrf_k: int) -> list[dict]:
    """Fuses vibe similarity and genre-bucket similarity by rank position
    (Reciprocal Rank Fusion), not raw score - avoids having to calibrate
    an arbitrary weight between two differently-scaled signals. Expects
    songs to already carry a vibe_score (see _with_vibe_scores)."""
    for song in songs:
        song["genre_score"] = round(bucket_scores.get(song["genre_bucket"], 0.0), 4)

    vibe_ranks = _to_ranks({s["track_id"]: s["vibe_score"] for s in songs})
    genre_ranks = _to_ranks({s["track_id"]: s["genre_score"] for s in songs})

    for song in songs:
        song["vibe_rank"] = vibe_ranks[song["track_id"]]
        song["genre_rank"] = genre_ranks[song["track_id"]]
        song["rrf_score"] = round(1 / (rrf_k + song["vibe_rank"]) + 1 / (rrf_k + song["genre_rank"]), 6)

    songs.sort(key=lambda s: s["rrf_score"], reverse=True)
    return songs


def _rerank(query: str, candidates: list[dict], match_type: str) -> list[dict]:
    """Stage 2: a cross-encoder reads the query and each candidate's actual
    description text jointly - not a bucket label, not a precomputed
    vector - for the final ordering. Slower than stage 1, so it only ever
    runs on a shortlist, never the full corpus. match_type records which
    ranking path a result came from (rrf / genre_locked / backfill) -
    cheap to attach, and it's what makes "why is this song here" answerable
    from the result alone rather than by re-deriving it after the fact."""
    if not candidates:
        return candidates
    reranker = get_reranker()
    pairs = [(query, c["description"]) for c in candidates]
    for candidate, score in zip(candidates, reranker.predict(pairs)):
        candidate["rerank_score"] = round(float(score), 4)
        candidate["match_type"] = match_type
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates


def hybrid_search(query: str, top_n: int = 20, rrf_k: int = RRF_K, shortlist_size: int = SHORTLIST_SIZE) -> list[dict]:
    """Vibe-only queries are ranked by RRF-fused vibe + genre similarity
    (stage 1, cheap, full corpus), narrowed to a shortlist, then reordered
    by the cross-encoder reranker (stage 2, slower, shortlist only).

    A query that literally names one specific genre (see
    genre_buckets.detect_genre_mention) or one specific artist actually in
    this library (see detect_artist_mention) skips the blend entirely:
    it's filtered to just that genre and/or artist first, so "rock songs"
    can't lose its top-20 slots to a hip-hop track that merely has a
    stronger vibe-embedding match, and "Drake songs" surfaces his actual
    tracks first rather than merely tracks that happen to sound like him.
    Genre and artist compose (both can apply at once) rather than one
    overriding the other. A query that literally expresses a mood/
    intensity preference (see pipeline.mood.detect_mood_preference)
    additionally excludes any candidate whose own description
    contradicts it - confirmed necessary: songs whose description
    explicitly says "pure aggression" or "cathartic aggression" still
    ranked in the top 20 for "calm rock songs" and "gentle rock songs",
    because the reranker reads that language without weighting it
    strongly enough against the query. All three checks are literal word
    matches, not embedding similarity - similarity was tried first for
    genre and dropped: it both missed real mentions (a qualifier like
    "gentle" pulling the embedding toward Folk/R&B diluted "rock"'s own
    signal in "gentle rock songs") and fired on non-mentions ("songs
    similar to michael jackson" scored Jazz as its top bucket by pure
    chance, with no genre named at all). A query's literal words are a
    fixed, known-in-advance fact; matching against them directly
    sidesteps both failure modes at once.

    If genre/artist/mood filtering leaves too few candidates to fill
    top_n, the remaining slots backfill from the normal cross-genre,
    cross-artist ranking (still respecting the mood preference, if any) -
    in practice this fires on nearly every artist mention, since most
    artists have far fewer songs in the library than a genre bucket does;
    that's what actually produces "this artist's songs first, then
    everything else" rather than a strict, often-too-small filter."""
    songs, embeddings = _fetch_songs()
    query_vector = get_model().encode([query], prompt_name="query")[0]
    bucket_scores = _bucket_scores(query_vector)
    _with_vibe_scores(songs, embeddings, query_vector)

    locked_genre = detect_genre_mention(query)
    known_artists = {s["primary_artist"] for s in songs if s["primary_artist"]}
    locked_artist = detect_artist_mention(query, known_artists)
    mood_preference = detect_mood_preference(query)
    eligible = _filter_by_mood(
        _filter_by_artist(_filter_by_genre(songs, locked_genre), locked_artist),
        mood_preference,
    )

    if locked_genre is None and locked_artist is None:
        shortlist = _rrf_rank(eligible, bucket_scores, rrf_k)[:shortlist_size]
        results = _rerank(query, shortlist, match_type="rrf")[:top_n]
    else:
        eligible.sort(key=lambda s: s["vibe_score"], reverse=True)
        locked_parts = [name for name, locked in (("genre", locked_genre), ("artist", locked_artist)) if locked is not None]
        results = _rerank(query, eligible[:shortlist_size], match_type="+".join(locked_parts) + "_locked")[:top_n]

    if len(results) < top_n:
        used_ids = {r["track_id"] for r in results}
        other_songs = _filter_by_mood([s for s in songs if s["track_id"] not in used_ids], mood_preference)
        fallback_shortlist = _rrf_rank(other_songs, bucket_scores, rrf_k)[:shortlist_size]
        fallback = _rerank(query, fallback_shortlist, match_type="backfill")
        results += fallback[: top_n - len(results)]

    return results
