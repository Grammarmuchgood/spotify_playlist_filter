from __future__ import annotations

import re

from pipeline.genre_buckets import GENRE_ALIASES

# A single- or two-letter derived alias ("a", "j", "g" from A$AP Rocky,
# J. Cole, G-Eazy) is too short to trust even when it happens to be the
# only artist producing it in this corpus - a word that short shows up
# constantly in ordinary English ("a song for...") and would false-fire
# via the anchor-adjacency check almost by accident. Every real short
# name confirmed useful here (Kanye, Tyler, Weeknd, Daryl) is well above
# this, so nothing of value is lost by requiring it.
MIN_ALIAS_LENGTH = 3

_GENRE_ALIAS_TOKENS = {alias for aliases in GENRE_ALIASES.values() for alias in aliases}


def _tokenize(name: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", name.lower()))


def build_artist_aliases(known_artists: set[str]) -> dict[str, list[str]]:
    """Real people rarely type an artist's full stored name - "Kanye
    songs" not "Kanye West songs", "Weeknd songs" not "The Weeknd
    songs". This derives a small set of extra short forms per artist
    (dropping a leading "The", or the first word alone otherwise) and
    keeps only the ones safe to trust: unique to exactly one artist in
    this corpus, not already the complete name of some unrelated artist
    (confirmed real case: "Fun Guns" would derive "fun", which is
    already "fun."'s whole name), and not a genre word (confirmed real
    case: "Pop Smoke" would derive "pop", already meaning the Pop
    genre). Anything ambiguous - confirmed real case: "Michael" among
    Michael Jackson/Bublé/McDonald - is dropped entirely rather than
    guessed, same as the full-name matcher already does.

    Deliberately does NOT derive a last-word form ("West" from "Kanye
    West") - unlike a first name, a bare surname isn't how people
    actually refer to most artists here, so it would mostly add
    low-value coincidental aliases without fixing a real gap.

    Callers hand the result to both hybrid.detect_artist_mention (for
    the artist-lock feature) and reference_track.resolve_reference_track
    (for "songs like X" artist confirmation) - built fresh from the
    actual corpus every call, same as detect_artist_mention's own
    vocabulary, so it grows and shrinks with the playlist automatically
    with no hand-maintained list."""
    full_name_tokens: dict[tuple[str, ...], set[str]] = {}
    candidates: dict[tuple[str, ...], set[str]] = {}

    for artist in known_artists:
        tokens = _tokenize(artist)
        if not tokens:
            continue
        full_name_tokens.setdefault(tokens, set()).add(artist)
        if len(tokens) < 2:
            continue
        derived = tokens[1:] if tokens[0] == "the" else tokens[:1]
        if len("".join(derived)) < MIN_ALIAS_LENGTH:
            continue
        candidates.setdefault(derived, set()).add(artist)

    aliases: dict[str, list[str]] = {}
    for derived, artists in candidates.items():
        if len(artists) != 1:
            continue  # ambiguous between two+ artists in this corpus - don't guess
        if derived in _GENRE_ALIAS_TOKENS:
            continue  # collides with a genre word
        if derived in full_name_tokens and full_name_tokens[derived] != artists:
            continue  # collides with an unrelated artist's actual full name
        artist = next(iter(artists))
        aliases.setdefault(artist, []).append(" ".join(derived))
    return aliases
