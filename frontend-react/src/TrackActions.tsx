import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { type Playlist } from './PlaylistPicker'

export function fetchPlaylists(): Promise<{ playlists: Playlist[] }> {
  return fetch('/playlists').then((response) => {
    if (!response.ok) throw new Error('Failed to load playlists')
    return response.json()
  })
}

export function queueTrack(trackId: string): Promise<boolean> {
  return fetch(`/queue?track_id=${encodeURIComponent(trackId)}`, { method: 'POST' }).then((r) => r.ok)
}

function addToExistingPlaylist(trackId: string, playlistId: string): Promise<void> {
  return fetch(`/add-to-playlist?track_id=${encodeURIComponent(trackId)}&playlist_id=${encodeURIComponent(playlistId)}`, {
    method: 'POST',
  }).then((response) => {
    if (!response.ok) throw new Error("Couldn't add to that playlist.")
  })
}

function createPlaylistAndAdd(trackId: string, name: string): Promise<{ name: string }> {
  return fetch(`/playlists/new?name=${encodeURIComponent(name)}&track_id=${encodeURIComponent(trackId)}`, {
    method: 'POST',
  }).then((response) => {
    if (!response.ok) throw new Error("Couldn't create that playlist.")
    return response.json()
  })
}

// Deliberately TWO independent pieces of state, not one shared "step" -
// queueing a song and adding it to a playlist are two separate actions
// on the same track, and doing one should never hide or block the
// other. Gating the playlist picker behind a queue failure was the
// actual bug in the first version; collapsing both into one step that
// locks after either succeeds would just recreate the same coupling in
// a different shape.
type PickerState = 'closed' | 'open'

type Props = {
  trackId: string
}

function TrackActions({ trackId }: Props) {
  const [picker, setPicker] = useState<PickerState>('closed')
  const [queueLabel, setQueueLabel] = useState<string | null>(null)
  const [playlistLabel, setPlaylistLabel] = useState<string | null>(null)
  const [newPlaylistName, setNewPlaylistName] = useState('')

  const { data: playlistsData } = useQuery({
    queryKey: ['playlists'],
    queryFn: fetchPlaylists,
    enabled: picker === 'open',
  })

  const queueMutation = useMutation({
    mutationFn: () => queueTrack(trackId),
    onSuccess: (queued) => {
      if (queued) {
        setQueueLabel('✓ Queued')
      } else {
        // Still offered automatically as a convenience on failure - just
        // no longer the only way to reach it.
        setPicker('open')
      }
    },
  })

  const addToExistingMutation = useMutation({
    mutationFn: (playlistId: string) => addToExistingPlaylist(trackId, playlistId),
    onSuccess: (_data, playlistId) => {
      const name = playlistsData?.playlists.find((p) => p.id === playlistId)?.name ?? 'playlist'
      setPlaylistLabel(`✓ Added to "${name}"`)
      setPicker('closed')
    },
  })

  const createMutation = useMutation({
    mutationFn: (name: string) => createPlaylistAndAdd(trackId, name),
    onSuccess: (data) => {
      setPlaylistLabel(`✓ Added to new "${data.name}"`)
      setPicker('closed')
      setNewPlaylistName('')
    },
  })

  return (
    <div className="flex flex-shrink-0 flex-col items-end gap-1.5">
      <div className="flex items-center gap-1.5">
        {queueLabel ? (
          <span className="text-xs text-teal-300">{queueLabel}</span>
        ) : (
          <button
            type="button"
            onClick={() => queueMutation.mutate()}
            disabled={queueMutation.isPending}
            className="rounded-full bg-white/10 px-3 py-1 text-xs text-white transition hover:bg-white/20 disabled:opacity-50"
          >
            {queueMutation.isPending ? '…' : '+ Queue'}
          </button>
        )}

        {playlistLabel ? (
          <span className="text-xs text-teal-300">{playlistLabel}</span>
        ) : (
          <button
            type="button"
            onClick={() => setPicker(picker === 'open' ? 'closed' : 'open')}
            className="rounded-full bg-white/10 px-3 py-1 text-xs text-white transition hover:bg-white/20"
          >
            + Playlist
          </button>
        )}
      </div>

      {picker === 'open' && (
        <div className="flex w-48 flex-col gap-1.5 rounded-lg bg-white/5 p-2">
          <select
            onChange={(event) => event.target.value && addToExistingMutation.mutate(event.target.value)}
            disabled={addToExistingMutation.isPending}
            defaultValue=""
            className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-white"
          >
            <option value="" disabled>
              Choose a playlist…
            </option>
            {playlistsData?.playlists.map((playlist) => (
              <option key={playlist.id} value={playlist.id}>
                {playlist.name}
              </option>
            ))}
          </select>
          <form
            onSubmit={(event) => {
              event.preventDefault()
              const trimmed = newPlaylistName.trim()
              if (trimmed) createMutation.mutate(trimmed)
            }}
            className="flex w-full gap-1"
          >
            <input
              type="text"
              value={newPlaylistName}
              onChange={(event) => setNewPlaylistName(event.target.value)}
              placeholder="or new playlist name…"
              className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-white placeholder-neutral-500"
            />
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="flex-shrink-0 rounded bg-teal-400 px-2 py-1 text-xs font-medium text-black disabled:opacity-50"
            >
              Create
            </button>
          </form>
        </div>
      )}
    </div>
  )
}

export default TrackActions
