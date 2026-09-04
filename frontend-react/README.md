# Frontend

React + TypeScript + Vite, talking to the FastAPI backend in `../backend`. See the [project root README](../README.md) for what this app actually does and how to run the whole thing together.

## Local development

The dev server proxies API routes (`/login`, `/search`, `/queue`, etc.) to the backend, so both need to be running:

```bash
npm install
npm run dev
```

Requires the backend running separately at `http://127.0.0.1:8000` (see the root README) — the proxy targets are configured in `vite.config.ts`.

## Stack, briefly

- **Vite** — dev server + build tool. No bundling in dev at all (native ES modules, near-instant startup); bundles only for `npm run build`.
- **TypeScript** — strict mode.
- **Tailwind CSS v4** — utility classes directly in components, no separate stylesheet to keep in sync.
- **TanStack Query** — all data fetching, mutations, and polling (used for tracking playlist-processing progress).
- No router — five linear screens, not independently bookmarkable destinations, so a router wasn't earning its complexity here.

## Build for production

```bash
npm run build
```

Outputs to `dist/`, which the FastAPI backend serves directly in production (see `backend/main.py`'s static mount) — no separate frontend host needed.
