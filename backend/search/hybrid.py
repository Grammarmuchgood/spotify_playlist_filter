from __future__ import annotations

import json

import numpy as np
from sentence_transformers import CrossEncoder

from db.database import get_connection
from pipeline.embed import get_model
from pipeline.genre_buckets import CANONICAL_GENRES, get_bucket_embeddings

RRF_K = 60  # standard IR-literature default; tunable
SHORTLIST_SIZE = 50  # how many candidates RRF hands to the reranker

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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _rrf_rank(query: str, rrf_k: int) -> list[dict]:
    """Stage 1: cheap, runs across the full corpus. Returns every song
    ranked by RRF-fused vibe + genre rank, richest-first, with description
    text included so stage 2 can rerank without a second DB round-trip."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT track_id, name, artist, embedding, genre_bucket, description FROM songs WHERE embedding IS NOT NULL"
    ).fetchall()
    conn.close()

    model = get_model()
    query_vector = model.encode([query], prompt_name="query")[0]

    vibe_scores = {row["track_id"]: _cosine_similarity(query_vector, np.array(json.loads(row["embedding"]))) for row in rows}

    bucket_vecs = get_bucket_embeddings()
    bucket_scores = {name: _cosine_similarity(query_vector, vec) for name, vec in zip(CANONICAL_GENRES, bucket_vecs)}
    genre_scores = {row["track_id"]: bucket_scores.get(row["genre_bucket"], 0.0) for row in rows}

    def to_ranks(scores: dict) -> dict:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {track_id: rank for rank, (track_id, _) in enumerate(ordered, start=1)}

    vibe_ranks = to_ranks(vibe_scores)
    genre_ranks = to_ranks(genre_scores)

    results = []
    for row in rows:
        track_id = row["track_id"]
        rrf_score = 1 / (rrf_k + vibe_ranks[track_id]) + 1 / (rrf_k + genre_ranks[track_id])
        results.append({
            "track_id": track_id,
            "name": row["name"],
            "artist": row["artist"],
            "genre_bucket": row["genre_bucket"],
            "description": json.loads(row["description"])["description"],
            "vibe_score": round(vibe_scores[track_id], 4),
            "vibe_rank": vibe_ranks[track_id],
            "genre_score": round(genre_scores[track_id], 4),
            "genre_rank": genre_ranks[track_id],
            "rrf_score": round(rrf_score, 6),
        })

    results.sort(key=lambda r: r["rrf_score"], reverse=True)
    return results


def hybrid_search(query: str, top_n: int = 20, rrf_k: int = RRF_K, shortlist_size: int = SHORTLIST_SIZE) -> list[dict]:
    """Stage 1 (RRF, cheap, full corpus) narrows to shortlist_size candidates.
    Stage 2 (cross-encoder reranker, slower, shortlist only) reads the query
    and each candidate's actual description text jointly - not the bucket
    label, not a precomputed vector - for the final top_n ordering."""
    shortlist = _rrf_rank(query, rrf_k)[:shortlist_size]

    reranker = get_reranker()
    pairs = [(query, r["description"]) for r in shortlist]
    rerank_scores = reranker.predict(pairs)

    for r, score in zip(shortlist, rerank_scores):
        r["rerank_score"] = round(float(score), 4)

    shortlist.sort(key=lambda r: r["rerank_score"], reverse=True)
    return shortlist[:top_n]
