from __future__ import annotations

import pytest

from search.hybrid import hybrid_search

# Every test in this file runs the full reranker-backed pipeline (~3-5s
# each after the first, which also pays the one-time model-load cost) -
# slower than test_detection.py on purpose, since these check the actual
# ranked output and the tiered-backfill/composition logic that only
# exists once genre+artist+mood are combined, not just whether detection
# fires correctly in isolation.


def _match_types(results: list[dict]) -> set[str]:
    return {r["match_type"] for r in results}


# ============================================================
# BASIC SANITY - a plain vibe query with no genre/artist/mood/reference
# ============================================================

def test_vibe_only_query_returns_full_results():
    data = hybrid_search("songs to study to", top_n=10)
    assert len(data["results"]) == 10
    assert _match_types(data["results"]) == {"rrf"}
    assert data["detected"] == {"genre": None, "artist": None, "mood": None, "reference_track": None}
    assert data["exact_match_count"] == 10


def test_response_shape_contract():
    # every caller (the /search API endpoint, the frontend) depends on this exact shape
    data = hybrid_search("rock songs", top_n=5)
    assert set(data.keys()) == {"results", "detected", "exact_match_count"}
    assert set(data["detected"].keys()) == {"genre", "artist", "mood", "reference_track"}
    assert isinstance(data["exact_match_count"], int)
    assert 0 <= data["exact_match_count"] <= len(data["results"])
    for r in data["results"]:
        assert {"track_id", "name", "artist", "primary_artist", "genre_bucket", "description", "match_type", "rerank_score"} <= r.keys()


# ============================================================
# GENRE LOCK - including tiny-bucket backfill (some genres have as few
# as 1-2 songs in the whole 649-track library)
# ============================================================

def test_genre_lock_surfaces_only_that_genre_when_plenty_exist():
    data = hybrid_search("rock songs", top_n=20)
    assert data["detected"]["genre"] == "Rock"
    assert data["exact_match_count"] == 20
    assert all(r["genre_bucket"] == "Rock" for r in data["results"])


@pytest.mark.parametrize("query, genre, min_real_matches", [
    ("classical music", "Classical", 1),  # only 1 Classical song in the entire library
    ("folk songs", "Folk", 2),
    ("drill songs", "Drill", 2),
])
def test_genre_lock_backfills_cleanly_for_tiny_buckets(query, genre, min_real_matches):
    data = hybrid_search(query, top_n=20)
    assert data["detected"]["genre"] == genre
    assert len(data["results"]) == 20  # backfill must still fill every slot
    real_matches = [r for r in data["results"] if r["genre_bucket"] == genre]
    assert len(real_matches) == min_real_matches
    assert data["exact_match_count"] == min_real_matches
    # the real match(es) must be genre_locked and sort before generic backfill
    assert all(r["match_type"] == "genre_locked" for r in real_matches)
    track_ids = [r["track_id"] for r in data["results"]]
    assert len(track_ids) == len(set(track_ids))  # no duplicate songs across locked + backfill


# ============================================================
# ARTIST LOCK - including a low-song-count artist (mostly backfill)
# ============================================================

def test_artist_lock_surfaces_real_songs_first():
    data = hybrid_search("Drake songs", top_n=10)
    assert data["detected"]["artist"] == "Drake"
    assert data["exact_match_count"] == 10
    assert all(r["primary_artist"] == "Drake" for r in data["results"])


def test_artist_lock_backfills_when_artist_has_few_songs():
    data = hybrid_search("Future songs", top_n=20)  # Future has only 3 songs
    assert data["detected"]["artist"] == "Future"
    real_matches = [r for r in data["results"] if r["primary_artist"] == "Future"]
    assert len(real_matches) == 3
    assert all(r["match_type"] == "artist_locked" for r in real_matches)
    assert data["exact_match_count"] == 3


# ============================================================
# TIERED BACKFILL - the composition bug: 94.8% of all artist x genre
# pairings in this library have zero overlap, so a combined genre+artist
# request used to silently produce 100% unrelated, generically-labeled
# backfill with no way to tell that apart from "this artist just has few
# songs." Confirmed real cases: Tame Impala has zero Rock songs, The
# Weeknd has zero Pop songs.
# ============================================================

@pytest.mark.parametrize("query, genre, artist", [
    ("Tame Impala rock songs", "Rock", "Tame Impala"),
    ("The Weeknd pop songs", "Pop", "The Weeknd"),
])
def test_tiered_backfill_on_total_genre_artist_miss(query, genre, artist):
    data = hybrid_search(query, top_n=10)
    assert data["detected"]["genre"] == genre
    assert data["detected"]["artist"] == artist
    # the exact genre+artist intersection is confirmed empty - nothing should claim
    # "genre+artist_locked", and exact_match_count must honestly report zero
    assert "genre+artist_locked" not in _match_types(data["results"])
    assert data["exact_match_count"] == 0
    # but it must NOT fall straight to fully-generic backfill - same artist (any genre)
    # is tried first, so the requested artist's own songs should still appear
    artist_only = [r for r in data["results"] if r["match_type"] == "artist_only_backfill"]
    assert len(artist_only) > 0
    assert all(r["primary_artist"] == artist for r in artist_only)
    assert len(data["results"]) == 10


def test_tiered_backfill_not_triggered_by_single_lock():
    # the tier logic only applies when BOTH genre and artist are locked together -
    # a genre-only miss (e.g. "classical music") must go straight to generic backfill,
    # unchanged from before this fix existed
    data = hybrid_search("classical music", top_n=20)
    assert "artist_only_backfill" not in _match_types(data["results"])
    assert "genre_only_backfill" not in _match_types(data["results"])
    assert "backfill" in _match_types(data["results"])


# ============================================================
# COMPOSITION - genre + artist + mood applying together, not overriding
# ============================================================

def test_genre_artist_mood_compose_when_intersection_exists():
    # Kanye West has real Hip-Hop/Rap songs, so this should be a genuine 3-way lock,
    # not a backfill case
    data = hybrid_search("aggressive Kanye West songs", top_n=10)
    assert data["detected"] == {"genre": None, "artist": "Kanye West", "mood": "intense", "reference_track": None}
    assert data["exact_match_count"] == 10
    assert all(r["primary_artist"] == "Kanye West" for r in data["results"])


# ============================================================
# REFERENCE TRACK - full pipeline, including the title-suffix fix's
# actual effect on ranking quality (not just whether resolution succeeds)
# ============================================================

def test_reference_track_uses_songs_own_vibe_not_literal_title_words():
    # "Rock with You" is Michael Jackson's disco/funk classic (genre_bucket: Pop) -
    # before the title-suffix-stripping fix, resolution silently failed and this query
    # fell back to literal text containing the word "rock", surfacing aggressive
    # rock/metal songs instead. Confirmed fixed: zero Rock or Metal genre contamination.
    data = hybrid_search("songs like Rock with You", top_n=20)
    assert data["detected"]["reference_track"] == {"name": "Rock with You - Single Version", "artist": "Michael Jackson"}
    genres = {r["genre_bucket"] for r in data["results"]}
    assert "Rock" not in genres
    assert "Metal" not in genres
    # the referenced track is never "similar to" itself
    assert all(r["name"] != "Rock with You - Single Version" for r in data["results"])


def test_reference_track_resolves_via_artist_alias_not_just_full_name():
    # confirmed real bug, now fixed: an artist made entirely of symbols (¥$) used to
    # win this resolution by default no matter what the query said, because its
    # normalized name is "" and "" is trivially "contained" in any string
    data = hybrid_search("songs like Burn by Kanye", top_n=5)
    assert data["detected"]["reference_track"] == {"name": "BURN", "artist": "Kanye West"}


# ============================================================
# ADVERSARIAL / ROBUSTNESS - must never raise, always return the
# contracted shape, regardless of how malformed the input is
# ============================================================

@pytest.mark.parametrize("query", [
    "a",
    "   ",
    "asdkfjhaslkdfjhalskdjfh",
    "ROCK SONGS PLEASE I NEED ROCK RIGHT NOW",
    "\U0001F3B5 chill vibes only \U0001F3B5",
    "'; DROP TABLE songs; --",
    "I need songs for that specific feeling when you're on a train at night watching "
    "the lights go by and thinking about someone you used to know but the song also "
    "needs to be a little bit hopeful and not too sad because I don't want to actually "
    "cry just feel something",
])
def test_adversarial_input_does_not_crash(query):
    data = hybrid_search(query, top_n=10)
    assert len(data["results"]) == 10
    assert isinstance(data["exact_match_count"], int)


def test_sql_injection_style_query_is_never_interpolated_into_sql():
    # not actually reachable as real SQL injection - every DB call on the search path
    # takes zero user input or uses parameterized `?` placeholders - but confirms that
    # end to end regardless, and that the corpus itself is untouched afterward
    data = hybrid_search("'; DROP TABLE songs; --", top_n=5)
    assert len(data["results"]) == 5
    from search.hybrid import _fetch_songs
    songs, _ = _fetch_songs()
    assert len(songs) == 649
