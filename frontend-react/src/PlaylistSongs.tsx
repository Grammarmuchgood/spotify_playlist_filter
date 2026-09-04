import { useQuery } from '@tanstack/react-query'
import QueueAllButton from './QueueAllButton'
import TrackActions from './TrackActions'

type Song = {
  track_id: string
  name: string
  artist: string
  genre_bucket: string | null
  description: string | null
}

type SongsResponse = {
  songs: Song[]
}

function fetchPlaylistSongs(playlistId: string): Promise<SongsResponse> {
  return fetch(`/playlists/${playlistId}/songs`).then(async (response) => {
    if (!response.ok) {
      // Surfaces the backend's real reason (e.g. a Spotify-side 403 on a
      // specific playlist's contents) instead of a generic message - this
      // is a real, expected failure mode for some playlists, not just an
      // exceptional case to hide from the user.
      const body = await response.json().catch(() => null)
      throw new Error(body?.detail || 'Failed to load songs')
    }
    return response.json()
  })
}

type Props = {
  playlistId: string
}

// Deliberately has no dependency on processing status - the backend
// falls back to asking Spotify directly for a playlist that's never been
// processed, so this list is always browsable. genre_bucket/description
// just come back null until the pipeline actually reaches each song.
function PlaylistSongs({ playlistId }: Props) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['playlist-songs', playlistId],
    queryFn: () => fetchPlaylistSongs(playlistId),
  })

  if (isLoading) {
    return <p className="text-neutral-400">Loading songs…</p>
  }

  if (isError || !data) {
    return <p className="text-red-400">{(error as Error)?.message || "Couldn't load this playlist's songs."}</p>
  }

  return (
    <div>
      {data.songs.length > 0 && (
        <div className="mb-2 flex justify-end">
          <QueueAllButton trackIds={data.songs.map((s) => s.track_id)} />
        </div>
      )}
      <ul className="divide-y divide-white/10">
        {data.songs.map((song) => (
          <li key={song.track_id} className="flex items-start justify-between gap-3 py-2.5">
            <div className="min-w-0">
              <p className="truncate font-medium">{song.name}</p>
              <p className="truncate text-sm text-neutral-400">{song.artist}</p>
            </div>
            <div className="flex flex-shrink-0 items-start gap-1.5">
              {song.genre_bucket ? (
                <span className="mt-1 rounded-full bg-white/10 px-2.5 py-0.5 text-xs text-neutral-300">
                  {song.genre_bucket}
                </span>
              ) : (
                <span className="mt-1 text-xs text-neutral-500">not yet processed</span>
              )}
              {/* No dependency on processing status here either - queueing
                  a raw Spotify track has never needed this app's own
                  pipeline data at all, only the track's ID. */}
              <TrackActions trackId={song.track_id} />
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default PlaylistSongs
