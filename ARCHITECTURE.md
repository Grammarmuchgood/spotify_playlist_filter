# Architecture

How this actually works, the design decisions behind it, and the real bugs found building it. The [README](README.md) covers what it does and how to run it — this is the deeper technical narrative, for anyone (including a future me) who wants to know why it's built this way.

## System overview

```
Spotify OAuth login
        │
        ▼
Pick a playlist ──▶ browse its songs immediately (raw Spotify data, no processing needed)
        │
        ▼ (on demand)
Ingestion pipeline, per song:
  fetch → audio features (iTunes preview + Librosa) → lyrics (lyrics.ovh)
        → LLM vibe description (Claude Haiku) → embedding (Qwen3-Embedding-0.6B)
        → genre bucket (embedding match against a fixed taxonomy)
        │
        ▼
Search: type or speak a query ──▶ hybrid ranking engine ──▶ ranked results
        │
        ▼
Act: queue a track, add to a playlist, or queue/add all results at once
```

Every stage after "fetch" only touches songs whose relevant column is still `NULL`, so re-processing an already-processed playlist (new tracks added, or just re-run) does zero redundant work and costs zero additional LLM spend for songs already done.

## Data model

Three tables, one SQLite file per user (`backend/data/users/{spotify_user_id}/vibe_filter.db`) — not a shared database with a `user_id` column to filter by. A forgotten `WHERE user_id = ?` in a shared table is a real, common way multi-tenant apps leak data across users; a wrong file path just fails loudly instead.

- **`songs`** — one row per track, keyed by Spotify track ID. Holds both raw metadata (name, artist, album, ISRC) and every pipeline stage's output (`audio_features`, `lyrics`, `description`, `embedding`, `genre_bucket` — all `NULL` until that stage runs).
- **`playlists`** — one row per playlist, tracking `processing_status` and `processed_count` so the frontend can poll progress.
- **`playlist_songs`** — the many-to-many junction between them, composite primary key `(playlist_id, track_id)`. A song processed once is never reprocessed or duplicated just because it sits in three playlists — its computed data lives in exactly one place, and each playlist just references it.

`fetch_playlist.py` upserts into `songs` with `ON CONFLICT DO UPDATE` on just the columns it owns (name/artist/album/etc.), never a blanket `INSERT OR REPLACE` — see the bug list below for why that distinction mattered in practice.

## The search engine

The obvious design — embed every song, embed the query, rank by cosine similarity — was the first thing built (`search/similarity.py`), and it's still what powers the vibe-ranking step. But pure embedding similarity turned out to have a real, measured blind spot: it can't reliably tell "this text is *about* X" apart from "this text merely *resembles* X" in embedding space. **"songs similar to michael jackson" scored the Jazz genre bucket as its best match (0.436, a 0.040 margin over the runner-up)** despite never naming a genre — pure embedding-space coincidence. The fix isn't a better embedding model; it's recognizing that some questions ("does this query name a specific genre/artist that exists in this library?") have a discrete right answer that a *literal* check answers exactly, where a *similarity* check can only ever approximate.

**Stage 0 — literal detection, not similarity.** `detect_genre_mention`, `detect_artist_mention`, and `detect_mood_preference` all share one matcher (`match_known_phrase`): tokenize the query, check for an exact word/phrase from a known vocabulary, and — for anything longer than a 3-word query — require it to sit within two words of a "request" word (*songs, music, vibe, playlist...*) so a real word buried in unrelated prose doesn't fire by accident. This one function, reused three ways:
- **Genre** — 21 canonical buckets (`Hip-Hop/Rap`, `Pop`, `Rock`, ... down to narrow ones like `Trap`, `Drill`, `Grime`, `Jerk` added after specific false positives, below). A query naming one filters to just that bucket before any ranking happens.
- **Artist** — built fresh from whichever artists are actually in the library being searched (not a hand-maintained list), plus a small alias table (`Kanye` → `Kanye West`) since almost nobody types an artist's full stored name from memory. One name is blocklisted (`fun.` — "fun songs" is just ordinary language, and the anchor-word check that resolves every other collision can't help here since "songs" next to "fun" *is* the anchor pattern).
- **Mood** — two word sets (`GENTLE_MOOD_WORDS` / `INTENSE_MOOD_WORDS`) with negation-awareness (`"without being aggressive"` doesn't count as aggressive), used to exclude candidates whose own description contradicts the requested mood outright — added after confirming songs described as "pure aggression" still ranked top-20 for "calm rock songs."

Genre, artist, and mood all **compose** — a query can lock all three at once — rather than one overriding the others.

**Reference-track resolution** (`"songs like X"`, `"similar to Y"`, `"reminds me of Z"`) extracts the reference span, resolves it against the actual library by normalized substring containment (not fuzzy string similarity — trailing words like "...but more upbeat" would dilute a similarity ratio), requires a short 1-word title (`"Easy"`, `"Ghost"`) to be confirmed by a co-occurring artist mention, and — if resolved — uses *that track's own embedding* as the query vector instead of encoding the query text. Genre/artist/mood detection still run, but only against whatever text sits outside the resolved reference span, so a word inside the referenced title can't be mistaken for part of the user's own request.

**Stage 1 — Reciprocal Rank Fusion (cheap, full corpus).** Every eligible song gets ranked twice — once by vibe-embedding similarity to the query, once by genre-bucket similarity — and the two rankings are fused by *rank position* (`1/(60 + rank)` per list, summed), not raw score. Rank fusion sidesteps having to calibrate an arbitrary weight between two differently-scaled similarity signals. This produces a shortlist of 50.

**Stage 2 — cross-encoder rerank (slow, shortlist only).** A `Qwen3-Reranker-0.6B` cross-encoder reads the query and each candidate's actual generated description *jointly* for the final ordering — the one step in the pipeline that judges real vibe language rather than a precomputed vector or a bucket label. Confirmed via profiling that this step alone is the large majority of query latency, which is exactly why it only ever sees 50 candidates, never the full corpus.

**Backfill, when a lock leaves too few results.** Measured directly: **94.8% of every possible artist × genre pairing in this library has zero overlap** (Tame Impala has no Rock-bucketed songs; neither does Drake), so a combined genre+artist lock running dry is the *default* outcome for a combined query, not an edge case. Backfill tries dropping just one constraint before giving up on both, and an artist locked alone backfills first from whatever genre(s) that artist's *own* songs actually occupy (confirmed necessary: Future has 3 songs, all Hip-Hop/Rap, and used to backfill from Pop/Rock/Soundtrack — wherever "future songs" happened to embed near). Every result carries a `match_type` (`rrf`, `genre_locked`, `artist_genre_backfill`, `backfill`, ...) so the API response can distinguish "confident match" from "nearest available substitute" — surfaced as `exact_match_count` rather than left for the caller to re-derive.

**A real model-loading bug worth naming**: the official `Qwen/Qwen3-Reranker-0.6B` checkpoint isn't set up for `sentence-transformers`' `CrossEncoder` out of the box — loading it left the actual scoring layer randomly initialized (a `newly initialized: ['score.weight']` warning), silently producing meaningless scores rather than failing loudly. Fixed by switching to a community checkpoint (`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`) re-packaged specifically for `CrossEncoder` compatibility, confirmed to load with real trained weights.

### Explicitly considered and rejected

- **An LLM-based multi-category reranker** (explicit per-candidate artist/genre/energy/vibe judgments) — more reliable in principle, but costs a real LLM call per shortlist candidate per query, breaking the "instant, free, local" property the entire local-embedding search was built around.
- **Confidence-margin-triggered dynamic bucket creation** (an automatic version of the manual `Trap`/`Drill`/`Grime`/`Jerk` fix) — real value for an always-growing library, but not clearly justified for what's currently a batch-processed, mostly-static playlist snapshot.
- **A full-corpus LLM genre reclassification pass** — instead, `pipeline/genre_buckets.py` keeps bulk assignment as cheap embedding-similarity matching, and reserves an LLM-based `reclassify_with_llm()` as a targeted, cheap correction for specific *known*-mismatched tracks (using that song's own generated description as context) — a scalpel, not a wholesale replacement.

## Ingestion pipeline: real bugs found

Roughly chronological, each confirmed by direct testing rather than assumed:

1. **No raw audio from Spotify at all** — `preview_url` has been null for every app created after Nov 2024, and the Web Playback SDK is DRM'd playback-only. Fallback: match each track against the free iTunes Search API by artist/title text + duration (±3s tolerance), since ISRC lookup isn't available on the free endpoint.
2. **Two silent Spotify API renames** (Feb 2026 migration) — `/users/{id}/playlists` → `/me/playlists` for playlist creation on Dev Mode apps (spotipy's own helper still 403s against the old path), and the `track` key nested in playlist items renamed to `item` (`tracks.total` → `items.total` too, caught separately later).
3. **`librosa.load` can't decode an in-memory AAC/m4a buffer without `ffmpeg`** — needs a real temp file on disk first.
4. **Energy/timbre word-thresholds need corpus-relative calibration, not fixed cutoffs** — the first pass called nearly every track "high energy" regardless of genre; fixed by computing tertiles across the whole corpus and bucketing relative to that.
5. **Transient network failures during long rate-limited backfill runs** (DNS failure, a 503, a timeout, a "no route to host") — a shared retry-with-backoff helper, with both backfill loops skip-and-continue on a still-failing row rather than letting one flaky request kill a 40-minute run, or wrongly recording "confirmed no match."
6. **Genius blocks all non-browser requests site-wide** (not just their search endpoint — genuine bot protection, not fixable with headers), and their API never returns lyrics text anyway for licensing reasons. Switched to **lyrics.ovh**, a free no-auth API that scrapes multiple lyrics sites server-side and returns whichever responds first.
7. **Lyrics search silently failing on three common patterns**, each confirmed individually against the live API: a trailing `"- Remastered 2015"`/`"- Radio Edit"` suffix, punctuation in the title (`"Ain't No Sunshine"` → 404, `"Aint No Sunshine"` → 200), and a leading `"The "` in the artist name. Fixed with progressively-more-aggressive fallback retries; recovered 66 previously-"no lyrics" tracks.
8. **The iTunes matcher accepted cover/karaoke versions** — title similarity and artist similarity were averaged into one score, letting a near-perfect title match carry a badly-wrong artist match past the threshold once normalization stripped the giveaway `"[Instrumental Karaoke Version]"` text. Fixed by checking artist similarity independently, plus a raw-text check for cover/karaoke/tribute markers before normalization can hide them. Re-validating the existing "ok" matches under the fix found **36 silently wrong matches** beyond the handful already known.
9. **Qwen3-Embedding-0.6B needs its built-in asymmetric-retrieval query prompt** (`prompt_name="query"`) — omitting it produced a "hub song" pattern (a handful of songs ranking high for nearly every unrelated query), holding measured top-10-of-50 agreement against a hand-labeled eval set at the ~20% chance baseline; adding it raised agreement to 28%.
10. **`INSERT OR REPLACE` silently wiped every computed column on re-fetch** — confirmed via direct testing of SQLite's actual conflict behavior: it deletes and reinserts the whole row using only the columns named in the statement, so a routine re-fetch of an already-processed playlist reset `audio_features`/`lyrics`/`description`/`embedding`/`genre_bucket` back to `NULL` for every track, not just new ones. Fixed with a proper `ON CONFLICT DO UPDATE` touching only the metadata columns that function actually owns.

## Multi-user auth

OAuth against Spotify (`auth/spotify_oauth.py`), then a session identity carried via Starlette's `SessionMiddleware` — a signed (not encrypted) cookie holding `{"user_id": ...}`. Signed means tamper-evident, not private: nothing sensitive is ever stored in the session itself, just the user ID used to select which per-user SQLite file and token cache to read. `SESSION_COOKIE_SECURE` is off for local `http://` dev (a `Secure`-flagged cookie is silently never sent over plain HTTP) and must be on in any real HTTPS deployment.

A note on adding OAuth scopes later: granting a new scope (e.g. `user-modify-playback-state` for queueing) to an app a user already authorized does **not** retroactively apply to their existing token — confirmed via spotipy's real behavior (a 401 "Permissions missing" until they re-authorize).

## Testing

210 tests (`cd backend && pytest`), split roughly into fast detection-logic tests (no ML involved, sub-second), slower full-pipeline/reranker-backed tests, and auth/isolation/playlist-processing tests. The methodology behind nearly every bug listed above was the same: deliberately try to break each detection mechanism with adversarial and edge-case queries (a homograph like "trap" in its non-musical sense, a negated mood word, a one-word title, an artist name that's also an English word) rather than only the inputs expected to pass — that's how the genre/artist/mood/reference edge cases above were actually found, not guessed at in advance.

## Known limitations

- Genre/mood/reference detection are literal-match against curated or corpus-derived vocabularies, not learned — a real gap in phrasing (a mood word not yet in either list, a genre spelled unusually) silently falls through to plain vibe ranking rather than erroring, which is a deliberate "miss safely" choice but is still a miss.
- The reranker is the dominant cost of every query; scaling to a much larger per-user library would need either a smaller shortlist, a lighter second-stage model, or caching.
- Spotify's Development Mode restricts a live-hosted instance to an explicit allowlist of accounts — see the README for why this project favors strong local-run documentation over a hosted demo.
