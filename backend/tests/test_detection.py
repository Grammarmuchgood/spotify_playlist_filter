from __future__ import annotations

import pytest

from pipeline.genre_buckets import detect_genre_mention
from pipeline.mood import contradicts_mood, detect_mood_preference
from pipeline.reference_track import extract_reference_mention, resolve_reference_track
from search.hybrid import detect_artist_mention

# ============================================================
# GENRE DETECTION - pure string/regex logic, no ML. Fast enough to run
# on every change; each case here was found by deliberately trying to
# break detect_genre_mention, not just cases expected to pass.
# ============================================================

GENRE_CASES = [
    # straightforward + aliases
    ("rock songs", "Rock"), ("some pop music", "Pop"), ("hip hop songs", "Hip-Hop/Rap"),
    ("hiphop tracks", "Hip-Hop/Rap"), ("rap songs", "Hip-Hop/Rap"), ("jazz songs", "Jazz"),
    ("electronic music", "Electronic/Dance"), ("edm songs", "Electronic/Dance"),
    ("indie songs", "Alternative/Indie"), ("alternative songs", "Alternative/Indie"),
    ("r&b songs", "R&B/Soul"), ("rnb music", "R&B/Soul"), ("soul music", "R&B/Soul"),
    ("k-pop songs", "K-Pop"), ("kpop tracks", "K-Pop"), ("country songs", "Country"),
    ("reggae songs", "Reggae"), ("metal songs", "Metal"), ("folk songs", "Folk"),
    ("latin music", "Latin"), ("classical music", "Classical"), ("afrobeats songs", "Afrobeats"),
    ("soundtrack music", "Soundtrack"), ("singer songwriter music", "Singer-Songwriter"),
    # homograph true positives - genre words that are ALSO ordinary English words
    ("trap music", "Trap"), ("jerk songs", "Jerk"), ("drill songs", "Drill"), ("grime music", "Grime"),
    # homograph true negatives - same words, ordinary non-musical sense, must NOT lock
    ("don't fall into that trap", None), ("stop being such a jerk", None),
    ("we have a fire drill at work today", None), ("clean the grime off the floor", None),
    # adjective/mood composition - previously failed via the embedding-similarity approach
    # that was tried and dropped before literal matching (see genre_buckets.py docstring)
    ("gentle rock songs", "Rock"), ("aggressive electronic songs", "Electronic/Dance"),
    ("sad pop songs", "Pop"), ("calm jazz music", "Jazz"),
    # false-positive risk, confirmed one real miss out of the whole battery: "songs" sits
    # within the anchor window of "rock" even though "rock" is used as a verb here
    ("songs that rock my world", "Rock"),
    ("songs that make my heart race", None),
    ("need a soul song for tonight", "R&B/Soul"),
    ("I need my playlist to feel like a warm hug", None),
    # multiple genres - NOT leftmost-wins (see match_known_phrase docstring): whichever
    # genre sits closest to the anchor word wins, confirmed via these two
    ("rock and jazz songs", "Jazz"), ("pop or hip hop songs", "Hip-Hop/Rap"),
    # bare/short queries - a query this short needs no anchor word at all
    ("rock", "Rock"), ("jazz", "Jazz"),
]


@pytest.mark.parametrize("query, expected", GENRE_CASES)
def test_genre_detection(query, expected):
    assert detect_genre_mention(query) == expected


# ============================================================
# ARTIST DETECTION - pure string/regex logic (match_known_phrase), plus
# build_artist_aliases for real-world short forms ("Kanye" for "Kanye
# West"). Vocabulary is built fresh from the real corpus via fixtures.
# ============================================================

ARTIST_CASES = [
    ("songs by Kanye West", "Kanye West"), ("Drake songs", "Drake"),
    ("some Travis Scott tracks", "Travis Scott"), ("Kendrick Lamar music", "Kendrick Lamar"),
    ("NewJeans songs", "NewJeans"),
    # homograph artist names - real artists whose stored name is an ordinary English word
    ("Future songs", "Future"), ("Ghost songs", "Ghost"), ("Player songs", "Player"),
    ("Silver songs", "Silver"), ("Smiley songs", "Smiley"), ("Train songs", "Train"),
    ("Oasis songs", "Oasis"),
    # blocklist - "fun." tokenizes to the ordinary word "fun", confirmed unsafe to auto-detect
    ("fun songs", None), ("fun. songs", None), ("I just want some fun music tonight", None),
    ("Fun Guns songs", "Fun Guns"),  # a different, non-blocklisted artist - must still work
    # partial names - fixed by build_artist_aliases (each confirmed unique in this corpus,
    # not colliding with another artist, a genre word, or an unrelated artist's full name)
    ("Kanye songs", "Kanye West"),
    ("Weeknd songs", "The Weeknd"),
    ("Tyler songs", "Tyler, The Creator"),
    ("Daryl Hall songs", "Daryl Hall & John Oates"),
    # deliberately still None - separate, out-of-scope problem: "A$AP Rocky" tokenizes to
    # ("a","ap","rocky") because of the internal $, so typed "ASAP" (one token, no $) never
    # matches; fixing this needs a hand-curated exception, not alias derivation
    ("ASAP Rocky songs", None),
    # deliberately still None - genuine 3-way ambiguity (Jackson/Bublé/McDonald); the
    # collision check in build_artist_aliases must exclude "michael" from ever being trusted
    ("Michael songs", None),
    # full-name controls
    ("Kanye West songs", "Kanye West"), ("The Weeknd songs", "The Weeknd"),
    ("A$AP Rocky songs", "A$AP Rocky"), ("Tyler The Creator songs", "Tyler, The Creator"),
    ("Daryl Hall John Oates songs", "Daryl Hall & John Oates"),
    # Kanye's identity is genuinely split across 4 different primary_artist values in this
    # library's real metadata (Kanye West / Ye / DONDA / ¥$) - a data reality, not a bug
    ("ye songs", "Ye"), ("DONDA songs", "DONDA"),
    # punctuation robustness - both sides go through the same tokenizer, so typing an
    # artist's name with or without their stylized punctuation both work
    ("will.i.am songs", "will.i.am"), ("will i am songs", "will.i.am"),
    ("Red Hot Chili Peppers songs", "Red Hot Chili Peppers"), ("Black Eyed Peas songs", "Black Eyed Peas"),
    ("MF DOOM songs", "MF DOOM"), ("21 Savage songs", "21 Savage"), ("Twenty One Pilots songs", "Twenty One Pilots"),
    # bare
    ("Drake", "Drake"), ("Kanye West", "Kanye West"),
    # regression: MusicBrainz artist-margin fix (see audio_features.py) - Cochise used to
    # get misclassified via a low-confidence MusicBrainz name-collision match
    ("Cochise songs", "Cochise"),
    # alias-safety negative controls - these derived short forms MUST be excluded by the
    # collision check in build_artist_aliases, not just happen to be absent
    ("Lil songs", None),  # 4-way collision: Tecca/Tjay/Uzi Vert/Wayne
    ("Black songs", None),  # collision: Black Eyed Peas/Black Sabbath
    ("pop songs", None),  # "pop" excluded for colliding with the Pop genre word (Pop Smoke)
    # multi-artist queries share the same closest-to-anchor-wins behavior as multi-genre,
    # since both go through the same match_known_phrase
    ("Drake and Kendrick Lamar songs", "Kendrick Lamar"),
    ("Kanye West or Drake songs", "Drake"),
]


@pytest.mark.parametrize("query, expected", ARTIST_CASES)
def test_artist_detection(query, expected, known_artists, artist_aliases):
    assert detect_artist_mention(query, known_artists, artist_aliases) == expected


# ============================================================
# MOOD DETECTION - literal word match against two poles, plus
# negation-awareness (a nearby "not"/"without"/"never" excludes a mood
# word from counting, confirmed necessary by 18 real descriptions in
# this library using a "[trait] without being aggressive" phrasing).
# ============================================================

MOOD_QUERY_CASES = [
    ("calm songs", "gentle"), ("aggressive songs", "intense"),
    ("gentle relaxing music", "gentle"), ("intense hardcore songs", "intense"),
    ("calm aggressive songs", None),  # genuinely contradictory - don't guess which pole wins
    ("peaceful songs to unwind to", "gentle"), ("furious rage music", "intense"),
    ("just normal songs", None),
    # query-side negation: a negated mention doesn't count toward that pole, but also
    # doesn't flip to the opposite one - "not aggressive" isn't strong enough evidence
    # that "gentle" was actually meant
    ("not aggressive songs", None), ("without being too intense", None),
]


@pytest.mark.parametrize("query, expected", MOOD_QUERY_CASES)
def test_mood_detection(query, expected):
    assert detect_mood_preference(query) == expected


# name substrings used to look up real songs in the `songs` fixture for the negation test
NEGATED_DESCRIPTION_SONGS = [
    "Toxic", "Me And My Broken Heart", "My Ordinary Life", "Sunroof", "oui",
    "Praise The Lord (Da Shine) (feat. Skepta)", "Magic In The Hamptons (feat. Lil Yachty)",
    "Good Day", "It Was A Good Day",
]


@pytest.mark.parametrize("song_name", NEGATED_DESCRIPTION_SONGS)
def test_mood_contradiction_ignores_negation(song_name, songs):
    song = next(s for s in songs if s["name"] == song_name)
    # each of these descriptions explicitly says the song ISN'T aggressive/frantic
    # ("...without being aggressive", "...but never frantic") - a raw word-membership
    # check used to wrongly exclude all of them from "calm"/"gentle" searches
    assert contradicts_mood(song["description"], "gentle") is False


def test_mood_contradiction_still_fires_without_negation(songs):
    song = next(s for s in songs if s["name"] == "Break Stuff")
    assert contradicts_mood(song["description"], "gentle") is True


# ============================================================
# REFERENCE TRACK RESOLUTION - "songs like X" / "similar to X" extraction
# and matching against real song titles. No ML needed: pure substring
# containment on normalized text.
# ============================================================

# expected: (artist_substring, name_substring) both required in the resolved song, or
# None if resolution must correctly decline
REFERENCE_CASES = [
    ("songs like Sky by Playboi Carti", "Sky", "Playboi Carti"),
    ("songs like Sky", None, None),  # 1-word title, no artist given - must not guess
    ("something like Riptide by Vance Joy", "Riptide", "Vance Joy"),
    ("sounds like Starboy", None, None),  # 1-word title, no artist given
    ("reminds me of Fluorescent Adolescent", "Fluorescent Adolescent", "Arctic Monkeys"),
    # title collisions - multiple real songs share this exact title in this library
    ("songs like Breathe", None, None),  # 3-way collision (Jax Jones/Olly Alexander/Yeat)
    ("songs like Breathe by Yeat", "Breathe", "Yeat"),
    ("songs like Breathe by Jax Jones", "Breathe", "Jax Jones"),
    # parenthetical-stripping in normalize_title correctly handles a stage-name-in-parens
    ("songs like Breathe by Olly Alexander", "Breathe", "Olly Alexander"),
    ("songs like Maneater", None, None),  # 2-way collision (Nelly Furtado/Daryl Hall & John Oates)
    # fixed by artist_aliases - "daryl" now confirms "Daryl Hall & John Oates"
    ("songs like Maneater by Daryl Hall", "Maneater", "Daryl Hall"),
    ("songs like Maneater by Nelly Furtado", "Maneater", "Nelly Furtado"),
    ("songs like Don't by Ed Sheeran", "Don't", "Ed Sheeran"),
    # the ¥$ bug: an artist name made entirely of symbols normalizes to "", and "" is
    # trivially "contained" in any string - confirmed fixed, both the bare-miss case and
    # the "wrong artist despite a different one being named" case
    ("songs like Burn", None, None),
    ("songs like Burn by Kanye", "Burn", "Kanye West"),
    ("songs like Burn by Kanye West", "Burn", "Kanye West"),
    ("songs like some completely made up song title xyz123", None, None),
    ("songs like In My Feelings by Drake but more upbeat", "In My Feelings", "Drake"),
    ("calm songs like Riptide by Vance Joy", "Riptide", "Vance Joy"),
    # title-suffix stripping - all of these are stored with a " - <version tag>" suffix
    # (Remastered/Radio Edit/Single Version/...) that a real person would never type;
    # 39 of 649 songs in this library have one
    ("songs like Rock with You", "Rock with You", "Michael Jackson"),
    ("songs similar to rock with you", "Rock with You", "Michael Jackson"),
    ("songs like Come Together", "Come Together", "The Beatles"),
    ("songs like Hey Jude", "Hey Jude", "The Beatles"),
    ("songs like Stairway to Heaven", "Stairway to Heaven", "Led Zeppelin"),
    ("songs like The Chain", "The Chain", "Fleetwood Mac"),
]


@pytest.mark.parametrize("query, expected_name, expected_artist", REFERENCE_CASES)
def test_reference_resolution(query, expected_name, expected_artist, songs, artist_aliases):
    reference = extract_reference_mention(query)
    assert reference is not None, f"no trigger phrase extracted from {query!r}"
    resolved = resolve_reference_track(reference[0], songs, artist_aliases)
    if expected_name is None:
        assert resolved is None
    else:
        assert resolved is not None
        assert expected_name.lower() in resolved["name"].lower()
        assert expected_artist.lower() in resolved["primary_artist"].lower()


def test_reference_extraction_leaves_leading_qualifier_as_remaining_text():
    # detection_text after a resolved reference is built from remaining_text (everything
    # BEFORE the trigger), not the reference span itself - a leading qualifier must
    # survive so mood/genre detection can still see it
    reference = extract_reference_mention("calm songs like Riptide by Vance Joy")
    assert reference == ("Riptide by Vance Joy", "calm")


def test_reference_extraction_loses_trailing_qualifier():
    # known, accepted limitation: a qualifier placed AFTER the trigger is absorbed into
    # the reference span and never reaches detection_text - documented here so a change
    # in this behavior is a deliberate decision, not a silent drift
    reference = extract_reference_mention("songs like In My Feelings by Drake but more upbeat")
    assert reference[1] == ""  # nothing before the trigger
    assert "more upbeat" in reference[0]  # swallowed into the reference span instead
