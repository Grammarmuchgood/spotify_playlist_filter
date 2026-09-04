import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import PlaylistPicker, { type Playlist } from './PlaylistPicker'
import PlaylistSongs from './PlaylistSongs'
import ProcessingStatus, { fetchStatus } from './ProcessingStatus'
import Search from './Search'

type Profile = {
  display_name: string
}

function App() {
  const [profile, setProfile] = useState<Profile | null>(null)
  const [checkingSession, setCheckingSession] = useState(true)
  const [selectedPlaylist, setSelectedPlaylist] = useState<Playlist | null>(null)
  // Set only by double-clicking a playlist - a separate mode from just
  // having one selected, since selecting alone shouldn't navigate away
  // from the picker grid.
  const [viewingPlaylist, setViewingPlaylist] = useState<Playlist | null>(null)

  // Same queryKey + queryFn as ProcessingStatus's own internal query -
  // TanStack Query treats these as the same query and shares one cached
  // result/poll between the two components, rather than each fetching
  // independently. This is only ever read here (to decide whether Search
  // should render), never displayed directly - ProcessingStatus still
  // owns the actual progress UI.
  const activePlaylist = viewingPlaylist ?? selectedPlaylist
  const { data: status } = useQuery({
    queryKey: ['status', activePlaylist?.id],
    queryFn: () => fetchStatus(activePlaylist!.id),
    enabled: activePlaylist !== null,
  })

  useEffect(() => {
    fetch('/me')
      .then((response) => {
        if (!response.ok) {
          throw new Error('not logged in')
        }
        return response.json()
      })
      .then((data: Profile) => setProfile(data))
      .catch(() => setProfile(null))
      .finally(() => setCheckingSession(false))
  }, [])

  if (checkingSession) {
    return (
      <main className="flex min-h-screen items-center justify-center text-neutral-400">
        <p>Checking login status…</p>
      </main>
    )
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <header className="mb-8 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Playlist Vibe Filter</h1>
        {profile && (
          <p className="text-sm text-neutral-400">
            {profile.display_name} —{' '}
            <a className="underline hover:text-white" href="/logout">
              Log out
            </a>
          </p>
        )}
      </header>

      {!profile ? (
        <button
          type="button"
          onClick={() => {
            window.location.href = '/login'
          }}
          className="rounded-full bg-teal-400 px-6 py-3 font-medium text-black hover:bg-teal-300"
        >
          Connect Spotify
        </button>
      ) : viewingPlaylist ? (
        <div>
          <button
            type="button"
            onClick={() => setViewingPlaylist(null)}
            className="mb-4 text-sm text-neutral-400 hover:text-white"
          >
            ← Back to playlists
          </button>
          <h2 className="mb-2 text-xl font-semibold">{viewingPlaylist.name}</h2>
          <ProcessingStatus playlistId={viewingPlaylist.id} />
          {status?.processing_status === 'complete' && <Search playlistId={viewingPlaylist.id} />}
          {/* Always rendered, regardless of processing state - the
              backend falls back to Spotify directly for a playlist
              that's never been processed, so there's always something
              real to show here. */}
          <div className="mt-8">
            <h3 className="mb-1 text-sm font-medium text-neutral-400">All songs</h3>
            <PlaylistSongs playlistId={viewingPlaylist.id} />
          </div>
        </div>
      ) : (
        <>
          {selectedPlaylist && (
            <div className="mb-4">
              <p className="text-neutral-300">
                Selected: <span className="font-medium text-white">{selectedPlaylist.name}</span>
              </p>
              <div className="mt-2">
                <ProcessingStatus playlistId={selectedPlaylist.id} />
              </div>
              {status?.processing_status === 'complete' && (
                <Search playlistId={selectedPlaylist.id} />
              )}
            </div>
          )}
          <PlaylistPicker
            selectedId={selectedPlaylist?.id ?? null}
            onSelect={setSelectedPlaylist}
            onOpen={(playlist) => {
              setSelectedPlaylist(playlist)
              setViewingPlaylist(playlist)
            }}
          />
        </>
      )}
    </main>
  )
}

export default App
