import { useQuery } from '@tanstack/react-query'

export type Playlist = {
  id: string
  name: string
  track_count: number
  image_url: string | null
}

type PlaylistsResponse = {
  playlists: Playlist[]
}

function fetchPlaylists(): Promise<PlaylistsResponse> {
  return fetch('/playlists').then((response) => {
    if (!response.ok) {
      throw new Error('Failed to load playlists')
    }
    return response.json()
  })
}

type Props = {
  selectedId: string | null
  onSelect: (playlist: Playlist) => void
  onOpen: (playlist: Playlist) => void
}

function PlaylistPicker({ selectedId, onSelect, onOpen }: Props) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['playlists'],
    queryFn: fetchPlaylists,
  })

  if (isLoading) {
    return <p className="text-neutral-400">Loading your playlists…</p>
  }

  if (isError || !data) {
    return <p className="text-red-400">Couldn't load your playlists. Try refreshing the page.</p>
  }

  if (data.playlists.length === 0) {
    return <p className="text-neutral-400">No playlists found on this account.</p>
  }

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
      {data.playlists.map((playlist) => {
        const selected = playlist.id === selectedId
        return (
          <button
            key={playlist.id}
            type="button"
            onClick={() => onSelect(playlist)}
            onDoubleClick={() => onOpen(playlist)}
            title="Double-click to browse this playlist's songs"
            className={`flex flex-col items-start gap-2 rounded-xl border p-3 text-left transition ${
              selected
                ? 'border-teal-400 bg-teal-400/10'
                : 'border-white/10 hover:border-white/30 hover:bg-white/5'
            }`}
          >
            {playlist.image_url ? (
              <img
                src={playlist.image_url}
                alt=""
                className="aspect-square w-full rounded-lg object-cover"
              />
            ) : (
              <div className="flex aspect-square w-full items-center justify-center rounded-lg bg-white/5 text-3xl">
                🎵
              </div>
            )}
            <div className="w-full">
              <p className="truncate font-medium">{playlist.name}</p>
              <p className="text-sm text-neutral-400">{playlist.track_count} tracks</p>
            </div>
          </button>
        )
      })}
    </div>
  )
}

export default PlaylistPicker
