from __future__ import annotations

import json

import numpy as np

from db.database import get_connection
from pipeline.embed import get_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query: str, top_n: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT track_id, name, artist, embedding FROM songs WHERE embedding IS NOT NULL").fetchall()
    conn.close()

    model = get_model()
    # Qwen3-Embedding-0.6B is trained for asymmetric retrieval: queries need
    # a task-instruction prefix (built into the model as prompts["query"]),
    # documents (song descriptions) don't - embed.py already encodes those
    # plain, which is correct.
    query_vector = model.encode([query], prompt_name="query")[0]

    results = []
    for row in rows:
        song_vector = np.array(json.loads(row["embedding"]))
        score = _cosine_similarity(query_vector, song_vector)
        results.append({
            "track_id": row["track_id"],
            "name": row["name"],
            "artist": row["artist"],
            "score": score,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]
