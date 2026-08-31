from __future__ import annotations

import json
import re

import numpy as np
from sentence_transformers import CrossEncoder

from db.database import get_connection
from pipeline.artist_aliases import build_artist_aliases
from pipeline.embed import cosine_similarity, get_model
from pipeline.genre_buckets import CANONICAL_GENRES, detect_genre_mention, get_bucket_embeddings, match_known_phrase
from pipeline.mood import contradicts_mood, detect_mood_preference
from pipeline.reference_track import extract_reference_mention, resolve_reference_track

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


def detect_artist_mention(
    query: str, known_artists: set[str], artist_aliases: dict[str, list[str]] | None = None
) -> str | None:
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
    where that automatic growth isn't safe.

    artist_aliases (see pipeline.artist_aliases.build_artist_aliases)
    adds a few safe short forms on top of each artist's full name - e.g.
    "Kanye" alongside "Kanye West" - confirmed real gap: "Kanye songs"
    used to fall through to a plain vibe search with no lock at all,
    since almost nobody types an artist's complete stored name from
    memory."""
    vocabulary = {}
    for artist in known_artists:
        if artist in ARTIST_DETECTION_BLOCKLIST:
            continue
        tokens = tuple(re.findall(r"[a-z0-9]+", artist.lower()))
        if tokens:
            vocabulary[tokens] = artist
    for artist, aliases in (artist_aliases or {}).items():
        if artist in ARTIST_DETECTION_BLOCKLIST:
            continue
        for alias in aliases:
            vocabulary[tuple(alias.split())] = artist
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


def hybrid_search(query: str, top_n: int = 20, rrf_k: int = RRF_K, shortlist_size: int = SHORTLIST_SIZE) -> dict:
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
    everything else" rather than a strict, often-too-small filter.

    When BOTH genre and artist are locked together, backfill tries
    dropping just one of the two before giving up on both entirely -
    confirmed necessary by actually measuring it: 94.8% of every
    possible artist x genre pairing in this library has zero overlap
    (e.g. Tame Impala has no Rock-bucketed songs at all, nor does Drake;
    every one of the 20 most-prolific artists is missing at least one of
    the six biggest genres), so "genre+artist_locked" silently producing
    zero results and falling straight through to fully-generic backfill
    was the default outcome for a combined query, not an edge case - and
    every one of those 20 results carried the exact same "backfill" label
    a legitimate "this artist just has few songs" case gets, with no way
    to tell the two apart from the output. Artist is tried first
    ("artist_only_backfill") before genre ("genre_only_backfill") -
    naming a specific artist is a more deliberate, specific signal than
    a genre modifier sitting on top of it, so "Tame Impala, just not
    rock" honors the request more than "rock, but not by Tame Impala."

    Returns a dict, not a bare list - `results` is the ranked list as
    before, `detected` reports what genre/artist/mood/reference this
    query was actually understood as (so a caller can tell "no exact
    matches for Rock + Tame Impala" apart from "no genre or artist was
    even mentioned"), and `exact_match_count` is how many of `results`
    came from something other than a backfill tier - the number a caller
    needs to decide whether to show that message at all.

    A query containing a "songs like X" / "similar to X" construction
    (see pipeline.reference_track.extract_reference_mention) that
    resolves to an actual track in this library uses that track's own
    embedding as the vibe query instead of encoding the query text -
    "songs like In My Feelings" then means actual vibe-similarity to
    that song, not textual similarity to the phrase "In My Feelings".
    Genre/artist/mood detection still run, but only against the query
    text outside the resolved reference span, so a word inside the
    referenced title itself can't be mistaken for the user's own request.
    If extraction or resolution fails (no such phrase, or nothing in the
    library matches confidently), this falls back to a normal text query
    using the full original query - resolution is a bonus when it's
    confident, never a hard requirement."""
    songs, embeddings = _fetch_songs()
    known_artists = {s["primary_artist"] for s in songs if s["primary_artist"]}
    artist_aliases = build_artist_aliases(known_artists)

    reference = extract_reference_mention(query)
    resolved_track = resolve_reference_track(reference[0], songs, artist_aliases) if reference else None
    detection_text = reference[1] if resolved_track else query

    if resolved_track is not None:
        query_vector = embeddings[resolved_track["track_id"]]
        # The reranker judges relevance by reading real vibe language
        # jointly against each candidate's description - "In My Feelings"
        # and "Drake" are proper nouns it has no vibe content to compare,
        # but the resolved track's own description ("slow-burning trap
        # with a mellow, warm vibe...") is exactly the kind of text it
        # was built to judge. Any words outside the reference span (e.g.
        # "chill" in "chill songs similar to Easy") ride along too, so an
        # explicit modifier on top of the reference still counts.
        rerank_text = f"{detection_text}. {resolved_track['description']}" if detection_text else resolved_track["description"]
    else:
        query_vector = get_model().encode([query], prompt_name="query")[0]
        rerank_text = query
    bucket_scores = _bucket_scores(query_vector)
    _with_vibe_scores(songs, embeddings, query_vector)

    locked_genre = detect_genre_mention(detection_text)
    locked_artist = detect_artist_mention(detection_text, known_artists, artist_aliases)
    mood_preference = detect_mood_preference(detection_text)
    eligible = _filter_by_mood(
        _filter_by_artist(_filter_by_genre(songs, locked_genre), locked_artist),
        mood_preference,
    )
    if resolved_track is not None:
        # A song is never "similar to" itself.
        eligible = [s for s in eligible if s["track_id"] != resolved_track["track_id"]]

    if locked_genre is None and locked_artist is None:
        shortlist = _rrf_rank(eligible, bucket_scores, rrf_k)[:shortlist_size]
        results = _rerank(rerank_text, shortlist, match_type="rrf")[:top_n]
    else:
        eligible.sort(key=lambda s: s["vibe_score"], reverse=True)
        locked_parts = [name for name, locked in (("genre", locked_genre), ("artist", locked_artist)) if locked is not None]
        results = _rerank(rerank_text, eligible[:shortlist_size], match_type="+".join(locked_parts) + "_locked")[:top_n]

    if len(results) < top_n:
        used_ids = {r["track_id"] for r in results}
        if resolved_track is not None:
            used_ids.add(resolved_track["track_id"])

        if locked_genre is not None and locked_artist is not None:
            for tier_pool, tier_label in (
                (_filter_by_artist(songs, locked_artist), "artist_only_backfill"),
                (_filter_by_genre(songs, locked_genre), "genre_only_backfill"),
            ):
                if len(results) >= top_n:
                    break
                tier_candidates = _filter_by_mood(
                    [s for s in tier_pool if s["track_id"] not in used_ids], mood_preference
                )
                tier_shortlist = _rrf_rank(tier_candidates, bucket_scores, rrf_k)[:shortlist_size]
                tier_results = _rerank(rerank_text, tier_shortlist, match_type=tier_label)
                added = tier_results[: top_n - len(results)]
                results += added
                used_ids.update(r["track_id"] for r in added)

        other_songs = _filter_by_mood([s for s in songs if s["track_id"] not in used_ids], mood_preference)
        fallback_shortlist = _rrf_rank(other_songs, bucket_scores, rrf_k)[:shortlist_size]
        fallback = _rerank(rerank_text, fallback_shortlist, match_type="backfill")
        results += fallback[: top_n - len(results)]

    return {
        "results": results,
        "detected": {
            "genre": locked_genre,
            "artist": locked_artist,
            "mood": mood_preference,
            "reference_track": (
                {"name": resolved_track["name"], "artist": resolved_track["primary_artist"]}
                if resolved_track is not None
                else None
            ),
        },
        "exact_match_count": sum(1 for r in results if not r["match_type"].endswith("backfill")),
    }
