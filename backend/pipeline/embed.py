from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from db.database import get_connection

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Loading the model takes ~60s (first run downloads ~1.5GB of weights,
# cached locally after that) - a module-level singleton means it's only
# loaded once per process, not once per function call.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_and_store() -> int:
    conn = get_connection()
    # Only rows with a description but no embedding yet - safely
    # re-runnable/resumable, same pattern as every other pipeline stage.
    rows = conn.execute(
        "SELECT track_id, description FROM songs WHERE description IS NOT NULL AND embedding IS NULL"
    ).fetchall()
    if not rows:
        conn.close()
        return 0

    track_ids = []
    texts = []
    for row in rows:
        # description is stored as the structured JSON blob from step 7
        # ({"mood": ..., "context_tags": [...], "energy": ..., "description": ...}) -
        # only the "description" field is what actually gets embedded.
        data = json.loads(row["description"])
        track_ids.append(row["track_id"])
        texts.append(data["description"])

    # No external rate limit here (this runs entirely on local compute), so
    # the whole batch goes in one encode() call rather than a throttled loop -
    # sentence-transformers batches internally far more efficiently than
    # encoding one text at a time.
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=True)

    for track_id, vector in zip(track_ids, vectors):
        conn.execute(
            "UPDATE songs SET embedding = ? WHERE track_id = ?",
            (json.dumps(vector.tolist()), track_id),
        )
    conn.commit()
    conn.close()
    return len(rows)
