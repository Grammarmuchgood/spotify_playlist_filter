## Playlist Vibe Filter — Setup Checklist

- [ ] **0. Confirm audio source** — check `preview_url` coverage for your playlist's tracks; decide fallback if spotty. *(research only, no push)*
- [ ] **1. Repo scaffold** — `git init`, create file structure, `.env.example`
  → **push:** initial scaffold commit
- [ ] **1b. Spotify dev app** — register app, allowlist your account *(no code, no push)*
- [ ] **2. OAuth flow** — login, callback, token storage/refresh in `auth/spotify_oauth.py`
  → **push:** working OAuth login
- [ ] **3. Write-back smoke test** — throwaway "create empty playlist" call to confirm write scope
  → **push:** fold into OAuth commit or its own tiny commit
- [ ] **4. Playlist fetch + DB schema** — `fetch_playlist.py`, `db/models.py`, pull real playlist into SQLite
  → **push:** playlist fetch + schema
- [ ] **5. Hand-label eval set** — ~50 songs, vibes, `tests/eval_labels.csv` *(runs in parallel with 6–8)*
  → **push:** eval labels file, whenever it's ready
- [ ] **6. Librosa audio features** — `pipeline/audio_features.py`
  → **push:** audio feature extraction
- [ ] **7. Lyrics + LLM description** — `pipeline/lyrics.py`, `pipeline/describe.py`
  → **push:** description generation
- [ ] **8. Embedding + similarity search** — `pipeline/embed.py`, `search/similarity.py`; run against eval set for first accuracy number
  → **push:** search working + eval results noted
- [ ] **9. Frontend** — query box, results, "create playlist" button
  → **push:** working frontend
- [ ] **10. README + failure modes + demo GIF**
  → **push:** final polish commit
