from __future__ import annotations

import json

import numpy as np
from sentence_transformers import CrossEncoder

from db.database import get_connection
from pipeline.embed import cosine_similarity, get_model
from pipeline.genre_buckets import CANONICAL_GENRES, get_bucket_embeddings

RRF_K = 60  # standard IR-literature default; tunable
SHORTLIST_SIZE = 50  # how many candidates a ranking stage hands to the reranker

# How confidently a query must name one specific genre before it's used as
# a hard filter instead of a soft RRF signal. Calibrated against real
# queries: genre-naming queries ("chill rap", "reggae vibes", "trap
# music") separated cleanly from vibe-only queries ("calm relaxing
# songs", "gym workout hype music") at a runner-up margin around 0.03 -
# every vibe-only query tested topped out at 0.025, every genre query
# started at 0.030. GENRE_FILTER_MIN_SCORE is a low sanity floor, not the
# real discriminator - absolute bucket-similarity score turned out not to
# separate the two cases on its own (a vibe-only query can score as high
# as a genre one); the margin over the runner-up is what actually means
# "the query names a genre" rather than "the query's wording just happens
# to lean toward one bucket a little more than the others."
GENRE_FILTER_MIN_SCORE = 0.35
GENRE_FILTER_MIN_MARGIN = 0.028

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
        "SELECT track_id, name, artist, embedding, genre_bucket, description FROM songs WHERE embedding IS NOT NULL"
    ).fetchall()
    conn.close()

    songs = []
    embeddings = {}
    for row in rows:
        songs.append({
            "track_id": row["track_id"],
            "name": row["name"],
            "artist": row["artist"],
            "genre_bucket": row["genre_bucket"],
            "description": json.loads(row["description"])["description"],
        })
        embeddings[row["track_id"]] = np.array(json.loads(row["embedding"]))
    return songs, embeddings


def _bucket_scores(query_vector: np.ndarray) -> dict[str, float]:
    bucket_vecs = get_bucket_embeddings()
    return {name: cosine_similarity(query_vector, vec) for name, vec in zip(CANONICAL_GENRES, bucket_vecs)}


def _detect_genre_lock(bucket_scores: dict[str, float]) -> str | None:
    """Returns a canonical genre name if the query confidently names one
    specific genre, so hybrid_search can filter to just that genre
    instead of treating it as one soft signal among several - otherwise
    None. Same confidence-margin pattern used for MusicBrainz tags
    elsewhere in this pipeline: a score floor plus a clear margin over
    the runner-up, so a query doesn't get hard-locked to a genre it
    merely resembles a little more than every other bucket."""
    ordered = sorted(bucket_scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ordered[0]
    runner_up_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if top_score < GENRE_FILTER_MIN_SCORE:
        return None
    if top_score - runner_up_score < GENRE_FILTER_MIN_MARGIN:
        return None
    return top_name


def _to_ranks(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {track_id: rank for rank, (track_id, _) in enumerate(ordered, start=1)}


def _with_vibe_scores(songs: list[dict], embeddings: dict[str, np.ndarray], query_vector: np.ndarray) -> list[dict]:
    for song in songs:
        song["vibe_score"] = round(cosine_similarity(query_vector, embeddings[song["track_id"]]), 4)
    return songs


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

    A query that confidently names one specific genre (see
    _detect_genre_lock) skips the blend entirely: it's filtered to just
    that genre first, so "rock songs" can't lose its top-20 slots to a
    hip-hop track that merely has a stronger vibe-embedding match. If that
    genre doesn't have enough songs to fill top_n, the remaining slots
    backfill from the normal cross-genre ranking."""
    songs, embeddings = _fetch_songs()
    query_vector = get_model().encode([query], prompt_name="query")[0]
    bucket_scores = _bucket_scores(query_vector)
    _with_vibe_scores(songs, embeddings, query_vector)

    locked_genre = _detect_genre_lock(bucket_scores)
    if locked_genre is None:
        shortlist = _rrf_rank(songs, bucket_scores, rrf_k)[:shortlist_size]
        return _rerank(query, shortlist, match_type="rrf")[:top_n]

    genre_songs = [s for s in songs if s["genre_bucket"] == locked_genre]
    genre_songs.sort(key=lambda s: s["vibe_score"], reverse=True)
    results = _rerank(query, genre_songs[:shortlist_size], match_type="genre_locked")[:top_n]

    if len(results) < top_n:
        used_ids = {r["track_id"] for r in results}
        other_songs = [s for s in songs if s["track_id"] not in used_ids]
        fallback_shortlist = _rrf_rank(other_songs, bucket_scores, rrf_k)[:shortlist_size]
        fallback = _rerank(query, fallback_shortlist, match_type="backfill")
        results += fallback[: top_n - len(results)]

    return results
