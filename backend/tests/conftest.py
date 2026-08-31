from __future__ import annotations

import os

# Must be set before anything imports sentence_transformers/CrossEncoder -
# without it, model loading tries a network round-trip to check for
# updates even when the model is already fully cached locally, and a
# slow/flaky connection can hang for 10+ minutes with zero output rather
# than failing fast. Every model used here (Qwen3-Embedding-0.6B,
# tomaarsen/Qwen3-Reranker-0.6B-seq-cls) is already cached after the
# first real run, so offline mode costs nothing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest

from pipeline.artist_aliases import build_artist_aliases
from search.hybrid import _fetch_songs


@pytest.fixture(scope="session")
def songs() -> list[dict]:
    """The real, current 649-track library - these tests deliberately run
    against actual data rather than a mocked fixture DB, same as
    test_similarity.py already does. Session-scoped: read-only, and
    re-fetching per test would just slow the suite down for no benefit."""
    all_songs, _ = _fetch_songs()
    return all_songs


@pytest.fixture(scope="session")
def known_artists(songs: list[dict]) -> set[str]:
    return {s["primary_artist"] for s in songs if s["primary_artist"]}


@pytest.fixture(scope="session")
def artist_aliases(known_artists: set[str]) -> dict[str, list[str]]:
    return build_artist_aliases(known_artists)
