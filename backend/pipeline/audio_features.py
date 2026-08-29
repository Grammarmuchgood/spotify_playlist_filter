# Deferred type-hint evaluation - needed because this file uses "str |
# None" union syntax, which the installed Python 3.9 can't evaluate at
# runtime (that syntax needs 3.10+); this future-import defers evaluating
# annotations so they're just treated as strings and never crash.
from __future__ import annotations

import json
import re
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import requests

from db.database import get_connection
from pipeline.http_utils import get_with_retry, make_throttle

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_MIN_INTERVAL_SECONDS = 3.5  # stay under the 20 req/min rate limit
DURATION_TOLERANCE_MS = 3000  # how close a candidate's length must be to Spotify's duration_ms to count as the same recording
MIN_NAME_SIMILARITY = 0.5  # reject a match even if duration matches, if the combined title/artist text is too dissimilar
MIN_ARTIST_SIMILARITY = 0.5  # checked independently - a near-perfect title match must not be able to carry a bad artist match past this
# Catalog listings for karaoke/tribute/cover versions self-identify with
# these terms in their own raw title or artist - checked before any
# normalization strips them out (confirmed bug: normalization was
# stripping "(Instrumental Karaoke Version)" etc., making a fake title
# look like a perfect match after the fact).
COVER_VERSION_MARKERS = [
    "karaoke", "tribute", "in the style of", "made famous by",
    "as made famous by", "originally performed by", "instrumental version",
    "cover version",
]

MUSICBRAINZ_BASE_URL = "https://musicbrainz.org/ws/2"
# MusicBrainz's usage policy requires a descriptive User-Agent identifying
# the application - unlike iTunes, an anonymous/browser-like User-Agent
# can get requests throttled or blocked.
MUSICBRAINZ_USER_AGENT = "PlaylistVibeFilter/0.1 (personal project)"
MUSICBRAINZ_MIN_INTERVAL_SECONDS = 1.0  # MusicBrainz's rate limit: 1 request/second
MUSICBRAINZ_MIN_ARTIST_SCORE = 85  # MusicBrainz's own 0-100 relevance score for the artist-name search match
# Same-named-artist collisions are real: searching "Cochise" returned three
# unrelated acts (a krautrock band, the actual hip-hop producer, a
# country-rock band) scored 100/99/97 - the top score alone isn't enough
# to trust, since a 1-point edge is well within noise. Require the top
# candidate to be clearly ahead of the runner-up, not just above the
# absolute floor.
MUSICBRAINZ_MIN_ARTIST_MARGIN = 10
# Confidence bar for the artist-level genre tag itself (separate from the
# artist-match score above). Calibrated against real examples seen this
# session: Tame Impala's "psychedelic rock" at 13 votes with no close
# second was reliable; Octavian's genres tied at 1 vote each were not, and
# Tyler, The Creator's "rock" tag - genuinely his highest-count genre
# artist-wide - still doesn't describe several individual tracks. Below
# either bar, treat the tag as unknown rather than risk baking a
# coin-flip genre into downstream description generation.
MUSICBRAINZ_MIN_GENRE_VOTES = 3
MUSICBRAINZ_MIN_GENRE_MARGIN = 2

# Independent rate limiters per API - iTunes' 20 requests/minute limit
# (60s / 20 = 3s minimum; 3.5s adds a small safety margin) and
# MusicBrainz's 1 request/second are unrelated to each other, so each
# needs its own throttle state.
_throttle_itunes = make_throttle(ITUNES_MIN_INTERVAL_SECONDS)
_throttle_musicbrainz = make_throttle(MUSICBRAINZ_MIN_INTERVAL_SECONDS)


def _get_musicbrainz(url: str, params: dict) -> requests.Response:
    resp = get_with_retry(_throttle_musicbrainz, url, params, headers={"User-Agent": MUSICBRAINZ_USER_AGENT})
    # get_with_retry only retries 5xx - a 4xx here is a real error (bad
    # request, not "no results"), so it's still raised, just not retried.
    resp.raise_for_status()
    return resp


def _normalize(text: str) -> str:
    text = text.lower()
    # Strip "(Live)", "[Remix]"-style suffixes; the *? makes the regex
    # non-greedy so it stops at the first closing bracket, not the last.
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)
    # Strip "feat. Someone" and everything after it.
    text = re.sub(r"feat\.?.*", "", text)
    # Strip remaining punctuation, keeping only letters/digits/spaces.
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    # difflib.SequenceMatcher is part of Python's standard library (no
    # extra dependency). .ratio() returns 0.0-1.0 for how similar two
    # strings are, based on the length of matching subsequences -
    # tolerant of small differences in spacing/capitalization/wording
    # after normalization.
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def find_itunes_match(track_name: str, primary_artist: str, duration_ms: int) -> dict | None:
    # Takes the primary artist directly (from Spotify's own artist list,
    # stored at fetch time) rather than deriving it here by splitting a
    # joined artist string - that split silently breaks for any artist
    # whose own name contains a comma (e.g. "Tyler, The Creator", "Earth,
    # Wind & Fire"), since there's no way to tell "a comma separating two
    # artists" apart from "a comma inside one artist's name" after they've
    # already been joined into one string.
    resp = get_with_retry(
        _throttle_itunes,
        ITUNES_SEARCH_URL,
        {"term": f"{primary_artist} {track_name}", "entity": "song", "limit": 5},
    )
    # get_with_retry only retries 5xx - iTunes always returns 200 with an
    # empty results list for "no matches," so any other status here is a
    # real error worth raising, not retrying.
    resp.raise_for_status()
    candidates = resp.json().get("results", [])

    # "Track the best candidate seen so far" loop: score every candidate,
    # keep whichever scores highest.
    best = None
    best_score = 0.0
    for c in candidates:
        # Skip anything with no preview to analyze, or no duration to
        # compare against.
        if not c.get("previewUrl") or not c.get("trackTimeMillis"):
            continue
        # Hard filter: reject candidates whose length differs too much -
        # this is what distinguishes the correct version of a song from
        # an acoustic version, radio edit, or remix with the same title.
        if abs(c["trackTimeMillis"] - duration_ms) > DURATION_TOLERANCE_MS:
            continue
        # Reject on the raw, unnormalized text - checked before _similarity
        # normalizes it, since normalization strips exactly the parenthetical
        # text ("[Instrumental Karaoke Version]") that would otherwise
        # reveal this isn't the real recording.
        raw_text = f"{c.get('trackName', '')} {c.get('artistName', '')}".lower()
        if any(marker in raw_text for marker in COVER_VERSION_MARKERS):
            continue
        title_sim = _similarity(c.get("trackName", ""), track_name)
        artist_sim = _similarity(c.get("artistName", ""), primary_artist)
        # Checked independently, not just as part of the averaged score -
        # a near-perfect title match must not be able to carry a badly
        # wrong artist (e.g. a cover act) past the threshold on its own.
        if artist_sim < MIN_ARTIST_SIMILARITY:
            continue
        score = (title_sim + artist_sim) / 2
        if score > best_score:
            best_score = score
            best = c

    # Final sanity check: even the "best" candidate gets rejected if it's
    # still a weak match, rather than accepting a bad guess.
    if best is None or best_score < MIN_NAME_SIMILARITY:
        return None
    return best


def extract_features(preview_url: str) -> dict:
    audio_bytes = requests.get(preview_url, timeout=15).content
    # librosa.load can't decode this AAC/m4a audio from an in-memory
    # buffer without ffmpeg installed - it needs a real file on disk to
    # hand to the system's native decoder. NamedTemporaryFile creates one;
    # delete=False keeps it from being auto-deleted when the "with" block
    # exits, since librosa still needs to open it afterward.
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        # y = the audio waveform itself, as a NumPy array of amplitude
        # samples. sr = the sample rate (samples per second). sr=None
        # means "use the file's native rate" rather than resampling.
        y, sr = librosa.load(tmp_path, sr=None)
    finally:
        # try/finally guarantees the temp file is deleted whether loading
        # succeeds or raises an exception. missing_ok avoids an error if
        # it's already gone somehow.
        Path(tmp_path).unlink(missing_ok=True)

    # Genuine signal-processing, not a lookup: analyzes the waveform for
    # periodic onset patterns to estimate tempo in beats per minute.
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # RMS ("root mean square") is a standard measure of average
    # loudness/intensity, computed in short windows across the track,
    # then averaged into one number.
    rms = float(np.mean(librosa.feature.rms(y=y)))
    # The frequency spectrum's "center of mass" - a track dominated by
    # high frequencies (cymbals, bright synths) scores higher than one
    # dominated by bass/low mids. A genuine proxy for perceived brightness.
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Harmonic-percussive source separation: splits the waveform into two
    # new waveforms - one mostly tonal/sustained content (melody, chords),
    # one mostly transient/percussive content (drum hits).
    harmonic, percussive = librosa.effects.hpss(y)
    # Summing squared amplitude gives an energy measure for each part;
    # the ratio says which one dominates the mix. The tiny +1e-9 avoids a
    # divide-by-zero crash on a silent track.
    harmonic_energy = float(np.sum(harmonic**2))
    percussive_energy = float(np.sum(percussive**2))
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-9)

    return {
        "tempo_bpm": round(float(tempo), 1),
        "energy_rms": round(rms, 4),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "harmonic_ratio": round(harmonic_ratio, 3),
    }


def process_song(track_name: str, primary_artist: str, duration_ms: int) -> dict:
    match = find_itunes_match(track_name, primary_artist, duration_ms)
    if match is None:
        return {"status": "no_itunes_match"}
    try:
        features = extract_features(match["previewUrl"])
    except Exception as exc:
        # Catch-all so one bad/corrupt preview doesn't crash the whole
        # batch run - the failure gets recorded per-song instead.
        return {"status": "extraction_failed", "error": str(exc)}
    features["status"] = "ok"
    # primaryGenreName comes free in the same iTunes response used for
    # matching above - real genre metadata at zero extra API cost.
    features["itunes_genre"] = match.get("primaryGenreName")
    return features


def fetch_and_store_audio_features(limit: int | None = None) -> dict:
    conn = get_connection()
    # Only process rows that haven't been done yet - makes this safely
    # re-runnable/resumable if it's interrupted partway through.
    query = "SELECT track_id, name, artist, primary_artist, duration_ms FROM songs WHERE audio_features IS NULL"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    counts = {"ok": 0, "no_itunes_match": 0, "extraction_failed": 0, "skipped_network_error": 0}
    for row in rows:
        try:
            result = process_song(row["name"], row["primary_artist"] or row["artist"].split(",")[0].strip(), row["duration_ms"])
        except requests.exceptions.RequestException:
            # find_itunes_match's retries (in _get_with_retry) are already
            # exhausted by this point - a real, sustained connectivity
            # problem, not a one-off blip. Leave audio_features as NULL
            # (don't write anything) so this row gets retried on the next
            # run, rather than being wrongly marked "no_itunes_match" -
            # which would mean "confirmed, permanently, no match exists"
            # for a song we simply couldn't reach the network to check.
            counts["skipped_network_error"] += 1
            continue
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        # SQLite has no native JSON column type, so the result dict is
        # serialized to a JSON string and stored in the TEXT column -
        # json.loads() reverses this whenever the data is read back.
        conn.execute(
            "UPDATE songs SET audio_features = ? WHERE track_id = ?",
            (json.dumps(result), row["track_id"]),
        )
        # Commit after every row (not just at the end) so progress
        # survives if the process is killed partway through a long run.
        conn.commit()

    conn.close()
    return counts


def backfill_genre() -> int:
    """For rows processed before itunes_genre was captured: re-run just the cheap
    iTunes search (no audio download/Librosa) to fill in the missing field."""
    conn = get_connection()
    rows = conn.execute("SELECT track_id, name, artist, primary_artist, duration_ms, audio_features FROM songs WHERE audio_features IS NOT NULL").fetchall()

    count = 0
    for row in rows:
        data = json.loads(row["audio_features"])
        # Skip rows that failed/had no match, and rows that already have
        # genre from a run after this field was added.
        if data.get("status") != "ok" or "itunes_genre" in data:
            continue
        try:
            match = find_itunes_match(row["name"], row["primary_artist"] or row["artist"].split(",")[0].strip(), row["duration_ms"])
        except requests.exceptions.RequestException:
            # Same reasoning as fetch_and_store_audio_features: skip this
            # row without writing itunes_genre at all, so a sustained
            # network blip doesn't get wrongly recorded as "looked it up,
            # genuinely no genre" - a future re-run will retry it.
            continue
        data["itunes_genre"] = match.get("primaryGenreName") if match else None
        conn.execute("UPDATE songs SET audio_features = ? WHERE track_id = ?", (json.dumps(data), row["track_id"]))
        conn.commit()
        count += 1

    conn.close()
    return count


def _fetch_musicbrainz_artist_genre(artist_name: str) -> str | None:
    """Genre isn't a per-song field on MusicBrainz - it's tagged at the artist
    level. This is two separate calls, confirmed by testing: a search call to
    find the artist's ID (MBID) and relevance score, then a lookup call on
    that ID with genres included (the search endpoint doesn't return genres
    directly, even when asked)."""
    # limit=5 (not 1) costs nothing extra - same single request, just asks
    # for enough candidates back to see whether a same-named runner-up
    # exists and how close it scored, rather than blindly trusting whichever
    # entry happened to rank first.
    search_resp = _get_musicbrainz(
        f"{MUSICBRAINZ_BASE_URL}/artist/",
        {"query": f'artist:"{artist_name}"', "fmt": "json", "limit": 5},
    )
    results = search_resp.json().get("artists", [])
    if not results:
        return None
    top_score = results[0].get("score", 0)
    runner_up_score = results[1].get("score", 0) if len(results) > 1 else 0
    # Reject a weak OR ambiguous match rather than risk tagging the wrong
    # same-named artist - same defensive pattern as the iTunes matching
    # above, extended to cover name collisions (multiple distinct artists
    # sharing a name), not just low relevance in absolute terms.
    if top_score < MUSICBRAINZ_MIN_ARTIST_SCORE:
        return None
    if top_score - runner_up_score < MUSICBRAINZ_MIN_ARTIST_MARGIN:
        return None
    mbid = results[0]["id"]

    lookup_resp = _get_musicbrainz(f"{MUSICBRAINZ_BASE_URL}/artist/{mbid}", {"fmt": "json", "inc": "genres"})
    genres = lookup_resp.json().get("genres", [])
    if not genres:
        return None
    # Genres come with a "count" - how many MusicBrainz users tagged the
    # artist with that genre. Highest count = most agreed-upon genre.
    ordered = sorted(genres, key=lambda g: g.get("count", 0), reverse=True)
    top_genre = ordered[0]
    top_count = top_genre.get("count", 0)
    runner_up_count = ordered[1].get("count", 0) if len(ordered) > 1 else 0
    # Confirmed via this session's bugs: a tag is only trustworthy when it's
    # both reasonably well-agreed-upon in absolute terms and clearly ahead
    # of the alternatives - Octavian's genres tied at 1 vote each are the
    # clearest failure case. A margin check does have a real coverage cost
    # (tested and rejected two embedding-based ways to recover near-synonym
    # cases like Tame Impala's "psychedelic rock" vs "neo-psychedelia" -
    # both bucket-matching and direct term similarity gave unreliable
    # results on short genre labels, e.g. "rock" vs "pop" scored more
    # similar than the actual synonym pair). That cost is accepted:
    # returning None loses a usable signal for a few songs, but a bad
    # synonym detector would risk letting a genuinely wrong genre through,
    # which is worse.
    if top_count < MUSICBRAINZ_MIN_GENRE_VOTES:
        return None
    if top_count - runner_up_count < MUSICBRAINZ_MIN_GENRE_MARGIN:
        return None
    return top_genre["name"]


def backfill_genre_musicbrainz() -> int:
    """Adds genre from MusicBrainz specifically for tracks with no iTunes
    match at all - there's no audio and no itunes_genre possible for these,
    so this is a genuinely different data source, not a duplicate lookup.
    Caches by artist within this run (not per-song) since genre is really
    an artist-level fact on MusicBrainz, and several of the unmatched
    tracks share an artist (e.g. multiple Kanye West / MF DOOM tracks)."""
    conn = get_connection()
    rows = conn.execute("SELECT track_id, artist, primary_artist, audio_features FROM songs WHERE audio_features IS NOT NULL").fetchall()

    cache: dict[str, str | None] = {}
    count = 0
    for row in rows:
        data = json.loads(row["audio_features"])
        if data.get("status") != "no_itunes_match" or "musicbrainz_genre" in data:
            continue
        # Falls back to the old split only for rows somehow still missing
        # primary_artist (shouldn't happen once fetch_playlist has been
        # re-run) - see find_itunes_match for why the split itself is unsafe.
        primary_artist = row["primary_artist"] or row["artist"].split(",")[0].strip()
        if primary_artist not in cache:
            try:
                cache[primary_artist] = _fetch_musicbrainz_artist_genre(primary_artist)
            except requests.exceptions.RequestException:
                # Even after _get_musicbrainz's retries, this artist's
                # lookup never actually completed - skip this row without
                # writing musicbrainz_genre at all, so a future re-run
                # still retries it. (Writing None here would look
                # identical to "looked it up, found no genre" and this
                # artist would never be retried again.)
                continue
        data["musicbrainz_genre"] = cache[primary_artist]
        conn.execute("UPDATE songs SET audio_features = ? WHERE track_id = ?", (json.dumps(data), row["track_id"]))
        conn.commit()
        count += 1

    conn.close()
    return count


ENERGY_ERA_COHORTS = [
    ("pre_1990", None, 1990),
    ("1990s", 1990, 2000),
    ("2000s", 2000, 2010),
    ("2010s", 2010, 2020),
    ("2020s", 2020, None),
]


def _energy_era_cohort(year: int | None) -> str:
    # Falls back to the largest/most-recent cohort ("2020s") when a song
    # has no release year at all - an arbitrary but reasonable default,
    # since that's the modal era in this library.
    if year is None:
        return "2020s"
    for name, lo, hi in ENERGY_ERA_COHORTS:
        if (lo is None or year >= lo) and (hi is None or year < hi):
            return name
    return "2020s"


def compute_thresholds(conn) -> dict:
    """Percentile cutoffs (tertiles) computed from the actual corpus, so 'high energy'
    means high relative to this playlist rather than an arbitrarily guessed number.

    energy_rms is the one exception: confirmed via direct testing that it
    correlates strongly with release year (r=0.50, mean value roughly
    doubles from the 1960s-80s to the 2020s) - this is the recording
    industry's "loudness war," a mastering-era artifact, not a real
    difference in musical energy. Bucketing it against the whole
    multi-decade corpus was systematically mislabeling older, dynamically-
    mastered rock as "low energy" regardless of how driving the song
    actually is (confirmed: "Fortunate Son," "Paint It, Black," "Seven
    Nation Army" all landed in the bottom tertile). Ranking energy_rms
    within release-era cohorts instead removes that confound - a song is
    now only "high energy" relative to its own era's typical mastering.
    tempo_bpm, spectral_centroid_hz, and harmonic_ratio don't show this
    problem (|r| < 0.03, 0.06, 0.27 respectively) so they stay pooled."""
    rows = conn.execute("SELECT release_date, audio_features FROM songs WHERE audio_features IS NOT NULL").fetchall()
    parsed = [(r["release_date"], json.loads(r["audio_features"])) for r in rows]
    ok = [(release_date, f) for release_date, f in parsed if f.get("status") == "ok"]

    def tertiles_of(values: list[float]) -> tuple[float, float]:
        arr = np.array(values)
        return float(np.percentile(arr, 33)), float(np.percentile(arr, 66))

    def tertiles(key: str) -> tuple[float, float]:
        return tertiles_of([f[key] for _, f in ok])

    # Falls back to the global (pooled) tertile for any cohort that ends
    # up with no songs in it - avoids a KeyError for a library that
    # happens to have nothing from some era.
    global_energy_tertile = tertiles_of([f["energy_rms"] for _, f in ok])
    energy_by_cohort: dict[str, tuple[float, float]] = {}
    for cohort_name, _, _ in ENERGY_ERA_COHORTS:
        values = [
            f["energy_rms"] for release_date, f in ok
            if _energy_era_cohort(int(release_date[:4]) if release_date else None) == cohort_name
        ]
        energy_by_cohort[cohort_name] = tertiles_of(values) if values else global_energy_tertile

    return {
        "tempo_bpm": tertiles("tempo_bpm"),
        "energy_rms": energy_by_cohort,
        "spectral_centroid_hz": tertiles("spectral_centroid_hz"),
        "harmonic_ratio": tertiles("harmonic_ratio"),
    }


def bucket_tertile(value: float, cutoffs: tuple[float, float], labels: tuple[str, str, str] = ("low", "medium", "high")) -> str:
    """Below the lower cutoff = first label, between cutoffs = second, above = third.
    Shared by every tertile-bucketed feature in this pipeline (tempo, energy,
    timbre, texture here; energy again with different labels in describe.py)."""
    lo, hi = cutoffs
    low, mid, high = labels
    return low if value < lo else mid if value < hi else high


def describe_features(features: dict, thresholds: dict, year: int | None = None) -> str:
    tempo_word = bucket_tertile(features["tempo_bpm"], thresholds["tempo_bpm"], ("slow tempo", "moderate tempo", "fast tempo"))
    # energy_rms is bucketed against its release-era cohort, not the
    # pooled corpus - see compute_thresholds for why.
    energy_cutoffs = thresholds["energy_rms"][_energy_era_cohort(year)]
    energy_word = bucket_tertile(features["energy_rms"], energy_cutoffs, ("low energy", "medium energy", "high energy"))
    timbre_word = bucket_tertile(
        features["spectral_centroid_hz"], thresholds["spectral_centroid_hz"],
        ("warm, mellow timbre", "balanced timbre", "bright, sharp timbre"),
    )
    texture_word = bucket_tertile(
        features["harmonic_ratio"], thresholds["harmonic_ratio"],
        ("rhythm/percussion-dominant mix", "balanced mix of tone and rhythm", "tonal/melodic-dominant mix"),
    )

    return f"{tempo_word}, {energy_word}, {timbre_word}, {texture_word}"


def generate_descriptions() -> int:
    conn = get_connection()
    thresholds = compute_thresholds(conn)
    rows = conn.execute("SELECT track_id, release_date, audio_features FROM songs WHERE audio_features IS NOT NULL").fetchall()

    count = 0
    for row in rows:
        data = json.loads(row["audio_features"])
        if data.get("status") != "ok":
            continue
        # Re-describes every "ok" row using corpus-wide thresholds, not
        # just newly-added ones - needed because thresholds themselves
        # change as more songs get processed, so old descriptions could
        # otherwise go stale relative to the full corpus.
        year = int(row["release_date"][:4]) if row["release_date"] else None
        data["description"] = describe_features(data, thresholds, year)
        conn.execute("UPDATE songs SET audio_features = ? WHERE track_id = ?", (json.dumps(data), row["track_id"]))
        count += 1

    conn.commit()
    conn.close()
    return count
