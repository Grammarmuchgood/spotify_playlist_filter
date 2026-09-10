# Playlist Vibe Filter

Natural-language vibe search over your own Spotify library. Type — or say — "calm rainy day songs" or "songs like Rock with You" and get a precisely ranked list of tracks that actually fit, pulled from playlists you already own, not a generic recommendation feed.

Spotify's own search only matches titles and artist names; it has no idea what a song *feels* like. This does.

<!-- Demo: record a short screen capture (connect → pick a playlist → watch it process → search a vibe → queue a result) and drop it here, e.g. ![demo](docs/demo.gif) -->

## What it actually does

- **Connect your real Spotify account** — a proper OAuth login, never sees your password.
- **Pick any of your real playlists**, browse its songs immediately (works even before processing), and process it on demand — audio features, lyrics, an LLM-generated vibe description, an embedding, and a genre bucket per song.
- **Search by typing or by voice** — "confident trap songs," "songs like Rock with You," "gentle Tame Impala songs" all resolve differently and correctly: genre lock, reference-track resolution, artist lock, and mood filtering all compose rather than compete.
- **Act on results** — queue a song straight to whatever's actively playing, or add it to any of your playlists (existing or brand new), individually or all at once.
- **Multi-user from the ground up** — every user's data lives in their own isolated SQLite file, not a shared table a forgotten filter could leak across.

## Why the search is actually good

The obvious approach — embed everything, rank by cosine similarity — was tried first and dropped. It has a real, measured failure mode: "songs similar to michael jackson" scored the **Jazz** genre bucket highest by pure embedding-space coincidence, despite never naming a genre at all. A query's literal words are a known, fixed fact; matching them directly for genre/artist/mood sidesteps that entirely, while embeddings still do the actual vibe-ranking work they're good at. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full reasoning, the two-stage retrieval design (cheap rank-fusion across the whole corpus, then a cross-encoder reranker on just the shortlist — confirmed via profiling that the reranker alone is ~92% of query latency), and the 25+ real, specific bugs found by deliberately testing adversarial and edge-case queries rather than just the ones expected to pass.

## Tech stack

**Backend** — FastAPI, SQLite (one file per user), Qwen3-Embedding-0.6B + a Qwen3-Reranker-0.6B cross-encoder (both local, free, via `sentence-transformers`), Claude Haiku for vibe descriptions, spotipy for the Spotify Web API, Librosa for audio-feature extraction, lyrics.ovh for lyrics (free, no auth), session-cookie auth (`itsdangerous`, signed not encrypted).

**Frontend** — React + TypeScript, Vite, Tailwind CSS v4, TanStack Query for all data fetching/mutations/polling. No router — five linear screens someone moves through once per session, not bookmarkable destinations.

## Quick start

Requires Python 3.12+, Node 22+, your own [Spotify app credentials](https://developer.spotify.com/dashboard) (free), and an [Anthropic API key](https://console.anthropic.com/). Lyrics and audio-feature sources (lyrics.ovh, iTunes, MusicBrainz) need no key at all.

```bash
# Backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your own keys
cd backend && uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend-react
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. First backend start takes ~60s while the embedding and reranker models load into memory.

**A real cost to know about**: audio features, lyrics, and embeddings are all free/local. The vibe-description step calls Anthropic's API once per *new* song (a fraction of a cent each with Haiku) — reprocessing an already-processed playlist costs nothing.

## Testing

210 tests, split into fast detection-logic tests (no ML, run in under a second), slower full-pipeline tests (reranker-backed), and auth/isolation/playlist-processing tests. The dominant methodology throughout: every detection mechanism was deliberately stress-tested with adversarial and edge-case inputs, not just the cases expected to pass — that's how most of the real bugs catalogued in `ARCHITECTURE.md` were actually found.

```bash
cd backend && pytest
```

## Project structure

```
backend/
  auth/        OAuth + session token handling
  db/          Per-user SQLite schema and connections
  pipeline/    Ingestion: fetch → audio features → lyrics → describe → embed → genre bucket
  playlist/    Spotify write-back (create/add-to playlist, queue)
  search/      The hybrid ranking engine
  main.py      FastAPI routes
frontend-react/
  src/         React components (one per screen/feature)
scripts/
  run_pipeline.py   CLI entry point for processing a playlist outside the web app
```

## License

MIT — see [`LICENSE`](LICENSE).
