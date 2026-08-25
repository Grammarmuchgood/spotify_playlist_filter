## Playlist Vibe Filter — Setup Checklist

**Progress: 7 / 10 steps complete**

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

⬜ **7. Lyrics + LLM description** — NOT STARTED
`pipeline/lyrics.py`, `pipeline/describe.py`
→ **push:** description generation

⬜ **8. Embedding + similarity search** — NOT STARTED
`pipeline/embed.py`, `search/similarity.py`; run against eval set for first accuracy number
→ **push:** search working + eval results noted

⬜ **9. Frontend** — NOT STARTED
Query box, results, "create playlist" button
→ **push:** working frontend

⬜ **10. README + failure modes + demo GIF** — NOT STARTED
→ **push:** final polish commit
