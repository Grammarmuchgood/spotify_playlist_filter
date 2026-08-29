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


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z-]+", text.lower()))


def detect_mood_preference(query: str) -> str | None:
    """Returns "gentle" or "intense" if the query expresses a clear
    intensity/mood preference, checked by literal word match - same
    reasoning as genre_buckets.detect_genre_mention: a smaller, more
    specific question than "what's the vibe" with a vocabulary narrow
    enough that matching it directly beats asking an embedding to infer
    it. Requires words from only one pole, not both - a query naming
    both ("calm aggressive rock songs") is genuinely contradictory, and
    guessing which pole wins is worse than not filtering at all."""
    words = _words(query)
    wants_gentle = bool(words & GENTLE_MOOD_WORDS)
    wants_intense = bool(words & INTENSE_MOOD_WORDS)
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
    nothing left to hedge on."""
    words = _words(description)
    opposite = INTENSE_MOOD_WORDS if preference == "gentle" else GENTLE_MOOD_WORDS
    return bool(words & opposite)
