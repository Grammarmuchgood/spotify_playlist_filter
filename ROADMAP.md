## Playlist Vibe Filter — Setup Checklist

**Progress: 9 / 10 steps complete**

---

✅ **0. Confirm audio source** — DONE
Spotify `preview_url` is null for all apps created after Nov 27, 2024, and Premium/Web Playback SDK doesn't provide raw audio either (playback-only, DRM'd). **Fallback: iTunes Search API** (`itunes.apple.com/search`, free, no auth) for 30s previews — match by artist/track text + `trackTimeMillis` vs. Spotify's `duration_ms` (~2-3s tolerance) since ISRC lookup isn't available on the free endpoint. Low-confidence matches get flagged as "no match" rather than guessed. *(research only, no push)*

✅ **1. Repo scaffold** — DONE
`git init`, create file structure, `.env.example`
→ **push:** initial scaffold commit

✅ **1b. Spotify dev app** — DONE
Register app, allowlist your account *(no code, no push)*

✅ **2. OAuth flow** — DONE
Login, callback, token storage/refresh in `auth/spotify_oauth.py`
→ **push:** working OAuth login

✅ **3. Write-back smoke test** — DONE
Throwaway "create empty playlist" call to confirm write scope. Found Spotify's Feb 2026 API migration killed `/users/{id}/playlists` for Dev Mode apps (`spotipy`'s `user_playlist_create()` still targets it and 403s) — replacement is `POST /me/playlists`, called directly via `sp._post()`. Same `/tracks` → `/items` rename will hit step 4 and step 7's track-adding call.
→ **push:** fold into OAuth commit or its own tiny commit

✅ **4. Playlist fetch + DB schema** — DONE
`fetch_playlist.py`, `db/models.py`, pull real playlist into SQLite; use `/playlists/{id}/items` not `/tracks` (see step 3 note). Target playlist: **"When"** (`4Jlag9nPT6xEKjNa515hUB`, 653 tracks). Also found the `track` key itself renamed to `item` in Feb 2026 migration. 647/653 stored — 4 local files (no catalog entry) and 2 null/removed tracks legitimately skipped; one track had a null artist name, handled with a fallback filter.
→ **push:** playlist fetch + schema

✅ **5. Hand-label eval set** — DONE
~50 songs, vibes, `tests/eval_labels.csv` *(runs in parallel with 6–8)*. Schema: `energy` (low/medium/high, sonic feel) split from `mood` (lyrical/emotional tone) — deliberately separate so compound cases (upbeat production + sad lyrics, etc.) are captured rather than averaged away. `context_tags` and `notes` describe theme and real-world listening context. Drafted from research, reviewed and corrected by hand.
→ **push:** eval labels file, whenever it's ready

✅ **6. Librosa audio features** — DONE (full batch run complete)
`pipeline/audio_features.py`; source audio via iTunes Search API preview fallback (see step 0). Matching by duration (±3s) validated against real data. Two bugs found and fixed: (1) `librosa.load` can't decode the AAC/m4a preview from an in-memory buffer without `ffmpeg` — needs a real temp file on disk. (2) Word-thresholds for energy/timbre/texture must be calibrated against the corpus's actual percentile distribution, not guessed absolute cutoffs — the first pass called almost every track "acoustic-leaning" and "high energy" regardless of genre. `generate_descriptions()` now computes tertiles across the whole corpus and buckets relative to that.
Full run on all 647 tracks: **557 matched + extracted (`ok`), 90 no confident iTunes match, 0 extraction failures.**
Genre backfill: iTunes' free `primaryGenreName` field covers all 557 `ok` tracks. For the 90 unmatched tracks (no audio possible either way), added a second source — **MusicBrainz** (`backfill_genre_musicbrainz()`) — since genre there is an artist-level fact, not per-song, so it's cached per unique artist (66 unique artists across 90 songs) rather than looked up per track. Found 66/90 real genres this way; the remaining 24 are genuinely obscure/unreleased tracks absent from both sources, left for step 7's LLM to infer from lyrics alone. **Total: 623/647 tracks (96%) now have real genre data.**
Along the way: both the iTunes and MusicBrainz calls hit real transient network failures during the ~40-60 min rate-limited runs (DNS resolution failure, a MusicBrainz 503, a read timeout, and a "no route to host" error - all genuine blips on this network, not simulated). Added a shared retry-with-backoff helper (`_get_with_retry`) used by both APIs, and made both backfill loops skip-and-continue on a row that still fails after retries (leaving it unmarked, not wrongly recorded as "confirmed no match") rather than letting one flaky request kill a 40-minute run.
→ **push:** audio feature extraction

✅ **7. Lyrics + LLM description** — DONE
`pipeline/lyrics.py`: originally planned to use the Genius API, but discovered `genius.com` (the whole site, not just the search endpoint) returns 403 to any non-browser request from this network — genuine site-wide bot protection, not fixable with headers, and Genius's own API never returns lyrics text anyway (licensing). Switched to **lyrics.ovh**, a free no-auth API that scrapes 6 lyrics sites server-side (Genius included) and returns whichever responds first — same underlying approach, just delegated to a server that isn't blocked. `GENIUS_API_KEY` left in `.env` unused in case this is revisited later.
Extracted the retry-with-backoff logic (previously duplicated for iTunes/MusicBrainz) into a shared `pipeline/http_utils.py`. Confirmed lyrics.ovh returns `404` for "no lyrics found" (a real answer, not a failure) vs. 5xx for actual transient errors, handled differently: 404 stores `''` (confirmed, won't be re-queried); network failure after retries leaves the row `NULL` (untried, retried next run). Full run: **490/647 found lyrics, 157 confirmed none** (instrumentals / not on any of the 6 sources), 0 network failures.

`pipeline/describe.py`: Claude Haiku 4.5 via `client.messages.parse()` with a Pydantic schema (`mood`, `context_tags`, `energy`, `description`) — structured output, not prompt-and-hope JSON. Combines lyrics + audio-feature words + genre + era per song; explicit rules for compound-mood cases (state conflicting audio/lyrical signals plainly rather than averaging), for missing lyrics (hedge, don't invent), and for missing audio (same). `energy` prefers the code-computed audio bucket over the LLM's own guess whenever real audio data exists — LLM only self-reports energy for the ~90 songs with no iTunes match.
Hit a real account-config error mid-build: personal Anthropic API keys not scoped to one workspace require an `anthropic-workspace-id` header on every request. Fixed by generating a workspace-scoped key instead of adding a hardcoded header.
Full run: **647/647 generated, 0 errors.** Cost ran ~48% over the initial estimate (~$1.23 actual vs. ~$0.83 estimated) — output length (mood/tags/description) came out longer than specified, and output tokens are priced higher than input. Spot-checked both degraded cases (no lyrics, no audio match) — both correctly hedge rather than fabricate detail.
→ **push:** description generation

✅ **8. Embedding + similarity search** — DONE
`pipeline/embed.py` + `search/similarity.py`: **Qwen3-Embedding-0.6B** via `sentence-transformers`, entirely local/free — chosen over paid APIs since embedding happens both once-per-song *and* once-per-query, so ongoing cost/latency mattered more here than for the one-time LLM description step. `tests/test_similarity.py` evaluates by comparing two independently-built rankings per query (hand-written eval notes vs. generated descriptions) rather than requiring new manual relevance judgments.
Found and fixed a real bug: Qwen3-Embedding-0.6B needs its built-in asymmetric-retrieval query prompt (`prompt_name="query"`) — missing it was producing a "hub song" pattern (certain songs ranking high for nearly every unrelated query) and held measured agreement at the ~20% chance-baseline for top-10-of-50; fixing it raised agreement to 28%.
**Real-world query testing surfaced two structural gaps**, not bugs: (1) the pipeline filters out bad matches well but under-prioritizes genre — e.g. a "chill rap" query doesn't reliably put rap tracks at the top, since genre is currently just soft context baked into the LLM's prose, never a real ranking signal; (2) no artist-specific or "songs like X" query support, since artist identity was deliberately excluded from the embedded text. **Designed but not yet built**: genre-aware ranking via Reciprocal Rank Fusion (blends a genre-relevance ranking with the vibe ranking by rank position, not raw score, so no arbitrary weight to tune) + canonical genre buckets (assigned via embedding similarity, not a hand-built keyword dictionary) + a `Qwen3-Reranker-0.6B` cross-encoder stage on the shortlist for final top-10-20 precision. Artist-priority and "songs similar to X" scoped as separate, cheaper follow-on features.

**Data-quality cleanup pass**, triggered by investigating why a "chill rap" query performed poorly — found and fixed four real bugs upstream of the ranking discussion entirely, then re-validated/regenerated the full corpus:
- **iTunes matcher accepted cover/karaoke versions**: title-similarity and artist-similarity were averaged into one score, letting a near-perfect title match (after normalization stripped the giveaway "[Instrumental Karaoke Version]" text) carry a badly-wrong artist match past the threshold. Confirmed on 2 Kanye tracks matched to an unrelated karaoke channel's audio. Fixed: artist similarity now checked independently, plus a raw-text check for explicit cover/karaoke/tribute markers before normalization can hide them. Re-validating all 557 previously-"ok" tracks under the fix found **36 silently wrong matches**, not just the 4 originally confirmed — corrected result: **521 ok / 126 no-match**.
- **Lyrics search silently failed on three common title/artist patterns**, confirmed individually against the live API: a trailing "- Remastered 2015"/"- Radio Edit" suffix, punctuation in the title ("Ain't No Sunshine" → 404, "Aint No Sunshine" → 200), and a leading "The " in the artist name ("The Temptations" → 404, "Temptations" → 200). Fixed with progressively-more-aggressive fallback retries. Recovered **66 tracks** across two fix rounds (157 → 91 genuinely-unavailable). One remaining case (Kanye's *Vultures 1* tracks) traced to a real 2024 distributor/licensing dispute that pulled the album from platforms and re-added it under fragmented metadata — an honest external gap, not a bug.
- Regenerated `description` + `embedding` for the **103 affected tracks** (36 corrected-match + 67 lyrics-recovered) rather than the full corpus — verified first that the corpus-wide percentile recalibration (removing 36 tracks from the "ok" pool) had **zero actual effect** on word-bucket boundaries before deciding a full ~$1.23 re-run wasn't justified.
→ **push:** search working + eval results noted; data-quality fixes

⬜ **9. Frontend** — NOT STARTED
Query box, results, "create playlist" button
→ **push:** working frontend

⬜ **10. README + failure modes + demo GIF** — NOT STARTED
→ **push:** final polish commit
