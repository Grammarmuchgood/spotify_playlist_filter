from __future__ import annotations

import re

from pipeline.audio_features import normalize_title

# Common ways people ask for "songs like this one" - not exhaustive, same
# open-ended-language caveat as pipeline.mood's word lists, grown the same
# evidence-driven way if a real phrasing gets missed.
REFERENCE_TRIGGER_PATTERNS = [
    re.compile(r"\bsongs?\s+like\s+", re.IGNORECASE),
    re.compile(r"\bsomething\s+like\s+", re.IGNORECASE),
    re.compile(r"\bsimilar\s+to\s+", re.IGNORECASE),
    re.compile(r"\bsounds?\s+like\s+", re.IGNORECASE),
    re.compile(r"\bin\s+the\s+style\s+of\s+", re.IGNORECASE),
    re.compile(r"\breminds?\s+me\s+of\s+", re.IGNORECASE),
]

# A 1-word title match ("Easy," "Ghost," "Poison" are all real titles in
# this library) is too easily coincidental to trust on its own - the same
# short-word risk already found for artist names. It's only trusted
# alongside a matching artist mention in the same text; a 2+ word title
# is distinctive enough to stand on its own, the same reasoning genre and
# artist phrase matching already use.
MIN_CONFIDENT_TITLE_WORDS = 2


def extract_reference_mention(query: str) -> tuple[str, str] | None:
    """Returns (reference_text, remaining_text) if the query contains a
    "songs like X" / "similar to X" construction - reference_text is
    everything after the trigger phrase (the song to resolve and match
    against), remaining_text is everything before it. Genre/mood/artist
    detection should run on remaining_text, not the full query or
    reference_text alone - otherwise a word inside the referenced title
    itself (e.g. "chill" in a hypothetical "songs like Chill Bill") could
    get mistaken for the user's own vibe request rather than part of the
    name being looked up. Returns None if no such construction is found."""
    for pattern in REFERENCE_TRIGGER_PATTERNS:
        match = pattern.search(query)
        if match:
            return query[match.end():].strip(), query[:match.start()].strip()
    return None


def resolve_reference_track(
    reference_text: str, songs: list[dict], artist_aliases: dict[str, list[str]] | None = None
) -> dict | None:
    """Finds the song in `songs` most likely being referenced by free
    text like "In My Feelings by Drake" or "In My Feelings by Drake
    mainly rap" - trailing words after the actual title/artist are
    normal in a real query and shouldn't prevent a match, which is why
    this checks normalized substring containment (does this song's title
    appear literally within the reference text?) rather than whole-string
    similarity - a similarity ratio gets diluted by exactly that kind of
    trailing noise, the same problem that ruled out embedding similarity
    for genre detection earlier.

    A short title (see MIN_CONFIDENT_TITLE_WORDS) only counts if the
    song's own artist also appears in the reference text - otherwise a
    title like "Easy" or "Ghost" would match almost anything, the same
    coincidental-short-word risk already found and guarded against for
    artist names. "Appears" includes a known short form from
    artist_aliases (see pipeline.artist_aliases.build_artist_aliases),
    not just the full stored name - confirmed real gap: "songs like
    Maneater by Daryl Hall" named the artist correctly, just not by
    their complete stored name "Daryl Hall & John Oates", and used to
    fail to confirm for exactly that reason. Among multiple candidates,
    prefers one with a confirmed artist, then the longest (most
    specific) title match. Returns None - falling back to a normal
    text-based vibe search - rather than guess when nothing clears that
    bar."""
    normalized_ref = normalize_title(reference_text)
    artist_aliases = artist_aliases or {}

    candidates = []
    for song in songs:
        normalized_name = normalize_title(song["name"])
        if not normalized_name or normalized_name not in normalized_ref:
            continue
        # An artist name made entirely of symbols (e.g. "¥$") normalizes to
        # "" - and "" is trivially "contained" in any string, which would
        # make artist_confirmed always true for that artist regardless of
        # what the query actually said. Requiring a non-empty normalized
        # name closes that off.
        normalized_artist = normalize_title(song["primary_artist"]) if song["primary_artist"] else ""
        artist_confirmed = bool(normalized_artist) and normalized_artist in normalized_ref
        if not artist_confirmed:
            artist_confirmed = any(
                alias in normalized_ref for alias in artist_aliases.get(song["primary_artist"], [])
            )
        if len(normalized_name.split()) < MIN_CONFIDENT_TITLE_WORDS and not artist_confirmed:
            continue
        candidates.append((artist_confirmed, len(normalized_name), song))

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]
