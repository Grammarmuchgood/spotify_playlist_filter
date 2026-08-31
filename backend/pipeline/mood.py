from __future__ import annotations

import re

# Confirmed necessary: songs whose own generated description explicitly
# says "pure aggression" or "cathartic aggression" (Break Stuff, Chop
# Suey!, Last Resort) still ranked in the top 20 for "calm rock songs" /
# "gentle rock songs" - the reranker reads that language but doesn't
# weight it strongly enough against a query asking for the opposite mood.
# The energy_rms-derived "energy" tag doesn't reliably substitute for this
# either: Chop Suey! and Last Resort are both tagged "medium" energy (a
# loudness measurement), not "high", despite being unmistakably violent
# in content - loudness and aggression are different axes, and a
# dynamically-mixed song can be intense without being the loudest
# recording.
#
# Neither list below is exhaustive - unlike genre, there's no closed,
# authoritative vocabulary for "words that mean calm." That's an
# acceptable, safe kind of incomplete: a word neither list knows about
# just means the filter doesn't fire for it, falling back to today's
# normal ranking - the same behavior every mood word got before this file
# existed. A missed word costs nothing extra; a wrongly-excluded song
# would be the dangerous direction, so these stay reasonably generous but
# not exhaustive. Grown the same way every other gap in this pipeline was
# found this session: a real query surfaces a miss, it gets added.
GENTLE_MOOD_WORDS = {
    "calm", "calming", "gentle", "gently", "relax", "relaxed", "relaxing",
    "chill", "chilled", "mellow", "soothing", "soothe", "peaceful", "peace",
    "tranquil", "tranquility", "serene", "serenity", "laid-back", "laidback",
    "easygoing", "soft", "softer", "quiet", "subdued", "tender", "delicate",
    "cozy", "comforting", "unwind", "unwinding", "downtempo",
}

INTENSE_MOOD_WORDS = {
    "aggressive", "aggression", "rage", "raging", "furious", "fury",
    "violent", "violence", "angry", "anger", "hostile", "hostility",
    "brutal", "brutality", "punishing", "savage", "fierce", "chaotic",
    "chaos", "frantic", "frenzied", "explosive", "screaming", "screamed",
    "shouting", "shouted", "harsh", "abrasive", "hardcore", "relentless",
    "ferocious", "menacing", "unhinged",
}

# Confirmed real gap: 18 of this library's own generated descriptions use
# a "[trait] without being aggressive/frantic" or "[trait] but never
# frantic" phrasing (Toxic, Me And My Broken Heart, My Ordinary Life,
# Sunroof, ...) - explicitly saying a song ISN'T intense, not that it is.
# A raw word-membership check can't tell those apart from a description
# that genuinely calls a song aggressive, and was wrongly excluding all
# of them from "calm"/"gentle" searches for saying the opposite of what
# it was being penalized for. Kept small and unambiguous on purpose:
# fragments like "can" (from "can't") or "won" (from "won't") are common
# enough on their own to risk suppressing a real mood word by accident,
# so those are left out rather than chased for completeness.
NEGATION_WORDS = {"not", "no", "never", "without", "hardly", "barely", "nothing", "isn", "aren"}
NEGATION_WINDOW = 3  # words before a mood word that still negate it - same anchor-window
                      # technique genre/artist detection uses, applied in reverse: a nearby
                      # word here means "don't count this," not "do count this"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z-]+", text.lower())


def _matched_pole_words(words: list[str], pole: set[str]) -> set[str]:
    """Which words from `words` are in `pole`, excluding any that are
    immediately negated (see NEGATION_WORDS/NEGATION_WINDOW) - shared by
    both functions below so a negated mood word is treated consistently
    whether it shows up in the user's own query or in a song's
    description."""
    found = set()
    for i, word in enumerate(words):
        if word not in pole:
            continue
        window = words[max(0, i - NEGATION_WINDOW):i]
        if any(w in NEGATION_WORDS for w in window):
            continue
        found.add(word)
    return found


def detect_mood_preference(query: str) -> str | None:
    """Returns "gentle" or "intense" if the query expresses a clear
    intensity/mood preference, checked by literal word match - same
    reasoning as genre_buckets.detect_genre_mention: a smaller, more
    specific question than "what's the vibe" with a vocabulary narrow
    enough that matching it directly beats asking an embedding to infer
    it. Requires words from only one pole, not both - a query naming
    both ("calm aggressive rock songs") is genuinely contradictory, and
    guessing which pole wins is worse than not filtering at all. A
    negated mention ("not aggressive") doesn't count toward that pole,
    but also doesn't flip to the opposite one - inferring "not X" means
    "wants the opposite of X" is a stronger claim than this vocabulary
    can safely support, so it's treated the same as no mention at all."""
    words = _words(query)
    wants_gentle = bool(_matched_pole_words(words, GENTLE_MOOD_WORDS))
    wants_intense = bool(_matched_pole_words(words, INTENSE_MOOD_WORDS))
    if wants_gentle and not wants_intense:
        return "gentle"
    if wants_intense and not wants_gentle:
        return "intense"
    return None


def contradicts_mood(description: str, preference: str) -> bool:
    """True if a song's own generated description uses language from the
    opposite pole of the requested mood - e.g. a "gentle" preference
    paired with a description that says "pure aggression." Used to
    exclude a candidate outright rather than rank it: once both the
    query and the song's own description are this explicit, there's
    nothing left to hedge on. A negated mention ("without being
    aggressive") does NOT count as a contradiction - confirmed real gap:
    it means the opposite of what the raw words suggest."""
    words = _words(description)
    opposite = INTENSE_MOOD_WORDS if preference == "gentle" else GENTLE_MOOD_WORDS
    return bool(_matched_pole_words(words, opposite))
