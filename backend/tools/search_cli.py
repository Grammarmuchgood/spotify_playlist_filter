"""Interactive REPL for manually testing hybrid_search.

Run from the project root:
    HF_HUB_OFFLINE=1 .venv/bin/python3 backend/tools/search_cli.py

Loads the embedding/reranker models once at startup (~15-30s), then
loops - every query after that only pays the real per-query cost
(~3-5s, almost entirely the reranker), so you can fire off many test
queries back to back without waiting for a fresh model load each time.

Commands:
  <any text>   run it as a search query
  /top N       change how many results are shown (default 10)
  quit / exit  leave
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search.hybrid import hybrid_search  # noqa: E402


def print_results(data: dict, top_n: int) -> None:
    detected = data["detected"]
    parts = [f"{k}={v}" for k, v in detected.items() if v]
    print(f"\ndetected: {', '.join(parts) if parts else '(nothing locked - plain vibe search)'}")
    print(f"exact matches: {data['exact_match_count']} / {len(data['results'])}\n")
    for i, r in enumerate(data["results"][:top_n], start=1):
        print(
            f"{i:2}. {r['name']:<40.40} {r['artist']:<30.30} "
            f"[{r['genre_bucket']:<18}] {r['match_type']:<22} score={r['rerank_score']}"
        )
    print()


def main() -> None:
    print("Loading models (one-time, ~15-30s)...")
    hybrid_search("warm up", top_n=1)  # pays the model-load cost now, not on your first real query
    print("Ready. Type a query and press enter. '/top N' to change result count. 'quit' to exit.\n")

    top_n = 10
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        if query.startswith("/top "):
            try:
                top_n = int(query.split()[1])
                print(f"top_n set to {top_n}\n")
            except (IndexError, ValueError):
                print("usage: /top 15\n")
            continue

        data = hybrid_search(query, top_n=top_n)
        print_results(data, top_n)


if __name__ == "__main__":
    main()
