from __future__ import annotations

import csv
import json
import os

import numpy as np

from db.database import get_connection
from pipeline.embed import cosine_similarity, get_model

EVAL_LABELS_PATH = os.path.join(os.path.dirname(__file__), "eval_labels.csv")
TOP_K = 10

# Representative queries spanning the vibes actually present across the
# 50 hand-labeled songs, including "sad but upbeat" as a deliberate test
# of the compound-mood cases the pipeline was specifically designed to
# catch (e.g. Save Your Tears, Redbone) rather than average away.
TEST_QUERIES = [
    "gym workout hype music",
    "chill rainy drive",
    "heartbreak and sadness",
    "party dance anthem",
    "study focus background music",
    "romantic love song",
    "nostalgic throwback",
    "sad but upbeat",
]


def load_eval_set() -> list[dict]:
    with open(EVAL_LABELS_PATH, newline="") as f:
        return list(csv.DictReader(f))


def evaluate() -> dict:
    eval_rows = load_eval_set()
    conn = get_connection()

    # Two independent pieces of text per song: what you wrote by hand
    # (mood/context_tags/notes) vs. what the pipeline generated
    # (description) - comparing rankings built from these two separately
    # tests real agreement, not the pipeline agreeing with itself.
    human_texts = []
    pipeline_vectors = []
    names = []
    for row in eval_rows:
        human_texts.append(f"{row['mood']}. {row['context_tags']}. {row['notes']}")
        db_row = conn.execute("SELECT name, artist, description, embedding FROM songs WHERE track_id = ?", (row["track_id"],)).fetchone()
        names.append(f"{db_row['name']} — {db_row['artist']}")
        pipeline_vectors.append(np.array(json.loads(db_row["embedding"])))
    conn.close()

    model = get_model()
    # human_texts and pipeline description embeddings are both the
    # "document" side being ranked - no prompt, matching how embed.py
    # encodes descriptions. Only the query itself needs Qwen3-Embedding's
    # built-in retrieval instruction prefix.
    human_vectors = model.encode(human_texts)

    per_query = []
    for query in TEST_QUERIES:
        query_vector = model.encode([query], prompt_name="query")[0]

        human_scores = [(names[i], cosine_similarity(query_vector, human_vectors[i])) for i in range(len(eval_rows))]
        pipeline_scores = [(names[i], cosine_similarity(query_vector, pipeline_vectors[i])) for i in range(len(eval_rows))]

        human_top = {name for name, _ in sorted(human_scores, key=lambda x: x[1], reverse=True)[:TOP_K]}
        pipeline_top = {name for name, _ in sorted(pipeline_scores, key=lambda x: x[1], reverse=True)[:TOP_K]}

        overlap = human_top & pipeline_top
        agreement = len(overlap) / TOP_K

        per_query.append({
            "query": query,
            "agreement": agreement,
            "human_top": sorted(human_top),
            "pipeline_top": sorted(pipeline_top),
            "agreed_on": sorted(overlap),
            "human_only": sorted(human_top - pipeline_top),
            "pipeline_only": sorted(pipeline_top - human_top),
        })

    overall_agreement = sum(q["agreement"] for q in per_query) / len(per_query)
    return {"overall_agreement": overall_agreement, "per_query": per_query}


def print_report(result: dict) -> None:
    print(f"Overall agreement: {result['overall_agreement']:.0%}\n")
    for q in result["per_query"]:
        print(f"--- {q['query']!r}: {q['agreement']:.0%} agreement ---")
        print(f"  Agreed on: {q['agreed_on']}")
        print(f"  Human-only (pipeline missed): {q['human_only']}")
        print(f"  Pipeline-only (pipeline added): {q['pipeline_only']}")
        print()


if __name__ == "__main__":
    print_report(evaluate())
