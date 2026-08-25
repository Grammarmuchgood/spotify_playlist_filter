## Playlist Vibe Filter — Setup Checklist

- [x] **0. Confirm audio source** — Spotify `preview_url` is null for all apps created after Nov 27, 2024, and Premium/Web Playback SDK doesn't provide raw audio either (playback-only, DRM'd). **Fallback: iTunes Search API** (`itunes.apple.com/search`, free, no auth) for 30s previews — match by artist/track text + `trackTimeMillis` vs. Spotify's `duration_ms` (~2-3s tolerance) since ISRC lookup isn't available on the free endpoint. Low-confidence matches get flagged as "no match" rather than guessed. *(research only, no push)*
- [x] **1. Repo scaffold** — `git init`, create file structure, `.env.example`
  → **push:** initial scaffold commit
- [x] **1b. Spotify dev app** — register app, allowlist your account *(no code, no push)*
- [x] **2. OAuth flow** — login, callback, token storage/refresh in `auth/spotify_oauth.py`
  → **push:** working OAuth login
- [x] **3. Write-back smoke test** — throwaway "create empty playlist" call to confirm write scope. Found Spotify's Feb 2026 API migration killed `/users/{id}/playlists` for Dev Mode apps (`spotipy`'s `user_playlist_create()` still targets it and 403s) — replacement is `POST /me/playlists`, called directly via `sp._post()`. Same `/tracks` → `/items` rename will hit step 4 and step 7's track-adding call.
  → **push:** fold into OAuth commit or its own tiny commit
- [x] **4. Playlist fetch + DB schema** — `fetch_playlist.py`, `db/models.py`, pull real playlist into SQLite; use `/playlists/{id}/items` not `/tracks` (see step 3 note). Target playlist: **"When"** (`4Jlag9nPT6xEKjNa515hUB`, 653 tracks). Also found the `track` key itself renamed to `item` in Feb 2026 migration. 647/653 stored — 4 local files (no catalog entry) and 2 null/removed tracks legitimately skipped; one track had a null artist name, handled with a fallback filter.
  → **push:** playlist fetch + schema
- [x] **5. Hand-label eval set** — ~50 songs, vibes, `tests/eval_labels.csv` *(runs in parallel with 6–8)*. Schema: `energy` (low/medium/high, sonic feel) split from `mood` (lyrical/emotional tone) — deliberately separate so compound cases (upbeat production + sad lyrics, etc.) are captured rather than averaged away. `context_tags` and `notes` describe theme and real-world listening context. Drafted from research, reviewed and corrected by hand.
  → **push:** eval labels file, whenever it's ready
- [x] **6. Librosa audio features** — `pipeline/audio_features.py`; source audio via iTunes Search API preview fallback (see step 0). Matching by duration (±3s) validated against real data. Two bugs found and fixed: (1) `librosa.load` can't decode the AAC/m4a preview from an in-memory buffer without `ffmpeg` — needs a real temp file on disk. (2) Word-thresholds for energy/timbre/texture must be calibrated against the corpus's actual percentile distribution, not guessed absolute cutoffs — the first pass called almost every track "acoustic-leaning" and "high energy" regardless of genre, which would've made every song's description nearly identical. `generate_descriptions()` now computes tertiles across the whole corpus and buckets relative to that.
  → **push:** audio feature extraction
- [ ] **7. Lyrics + LLM description** — `pipeline/lyrics.py`, `pipeline/describe.py`
  → **push:** description generation
- [ ] **8. Embedding + similarity search** — `pipeline/embed.py`, `search/similarity.py`; run against eval set for first accuracy number
  → **push:** search working + eval results noted
- [ ] **9. Frontend** — query box, results, "create playlist" button
  → **push:** working frontend
- [ ] **10. README + failure modes + demo GIF**
  → **push:** final polish commit
