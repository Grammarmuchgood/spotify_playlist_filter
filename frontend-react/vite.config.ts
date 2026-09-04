import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const BACKEND = 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  // Tailwind v4 plugs in directly as a Vite plugin - no tailwind.config.js,
  // no postcss.config.js, no separate init step. It scans every source file
  // for class names itself and generates only the CSS actually used.
  plugins: [react(), tailwindcss()],
  server: {
    // Vite's default host binds to IPv6 loopback (::1) only, not
    // 127.0.0.1 - confirmed directly (lsof showed only an IPv6 listener).
    // The backend's Spotify redirect URI and every cookie this app sets
    // are all scoped to 127.0.0.1 specifically, not "localhost" - a
    // real, different hostname as far as cookies/CORS are concerned,
    // even though both reach the same machine. Binding here explicitly
    // keeps both servers on the exact same hostname.
    host: '127.0.0.1',
    // Forwards these specific paths to the real FastAPI backend, server-side -
    // the browser never sees 127.0.0.1:8000 at all, only ever talks to this
    // dev server. That's what keeps the login session cookie working: the
    // browser thinks every response (including /callback's Set-Cookie) came
    // from this same origin, so none of the cross-origin cookie rules that
    // blocked Swagger's "Try it out" on /login ever come into play here.
    proxy: {
      '/login': BACKEND,
      '/callback': BACKEND,
      '/logout': BACKEND,
      '/me': BACKEND,
      '/playlists': BACKEND,
      '/search': BACKEND,
      '/queue': BACKEND,
      '/add-to-playlist': BACKEND,
    },
  },
})
