import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

export type Status = {
  playlist_id: string
  name: string | null
  total_tracks: number | null
  processed_count: number
  processing_status: string
}

export const TERMINAL = new Set(['not_started', 'complete', 'failed'])

// Exported so App.tsx can run a useQuery with the exact same queryKey +
// queryFn - TanStack Query deduplicates identical queries across
// components automatically, so this component and App.tsx share one
// cached fetch/poll instead of each independently hitting the backend.
export function fetchStatus(playlistId: string): Promise<Status | null> {
  return fetch(`/playlists/${playlistId}/status`).then((response) => {
    if (response.status === 404) return null // never processed before - not an error
    if (!response.ok) throw new Error('Failed to check processing status')
    return response.json()
  })
}

function startProcessing(playlistId: string): Promise<void> {
  return fetch(`/playlists/${playlistId}/process`, { method: 'POST' }).then((response) => {
    // 409 means it's already running - not this click's fault, and the
    // status poll below will reflect the real state either way.
    if (!response.ok && response.status !== 409) {
      throw new Error('Failed to start processing')
    }
  })
}

type Props = {
  playlistId: string
}

function ProcessingStatus({ playlistId }: Props) {
  const queryClient = useQueryClient()

  const { data: status } = useQuery({
    queryKey: ['status', playlistId],
    queryFn: () => fetchStatus(playlistId),
    // Poll every 2s only while something is actually in progress -
    // refetchInterval can read the last result and decide, not just be a
    // fixed number. Returning false stops polling entirely once
    // complete/failed/never-started, instead of hitting the backend
    // forever for a playlist that isn't doing anything.
    refetchInterval: (query) => {
      const current = query.state.data
      return current && !TERMINAL.has(current.processing_status) ? 2000 : false
    },
  })

  const { mutate: triggerProcessing, isPending: starting } = useMutation({
    mutationFn: () => startProcessing(playlistId),
    // Refetch status immediately after triggering, rather than waiting
    // up to 2s for the next scheduled poll to notice anything changed.
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['status', playlistId] }),
  })

  if (status === undefined) {
    return <p className="text-neutral-400">Checking processing status…</p>
  }

  if (status?.processing_status === 'complete') {
    return (
      <p className="text-teal-300">
        Ready to search — {status.processed_count} of {status.total_tracks} songs processed.
      </p>
    )
  }

  if (status && !TERMINAL.has(status.processing_status)) {
    const total = status.total_tracks ?? 0
    const percent = total > 0 ? Math.round((status.processed_count / total) * 100) : 0
    return (
      <div>
        <p className="mb-2 text-neutral-300">
          {status.processing_status.replace(/_/g, ' ')}… {status.processed_count} of {total}
        </p>
        <div className="h-2 w-full rounded-full bg-white/10">
          <div
            className="h-2 rounded-full bg-teal-400 transition-all"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>
    )
  }

  // status is null (never processed) or processing_status is 'failed' -
  // either way, nothing is running, so this is the one place a real
  // Anthropic API call can actually be triggered, and only by an
  // explicit click, never automatically just from selecting a playlist.
  return (
    <div>
      {status?.processing_status === 'failed' && (
        <p className="mb-2 text-red-400">Something went wrong processing this playlist.</p>
      )}
      <button
        type="button"
        onClick={() => triggerProcessing()}
        disabled={starting}
        className="rounded-full bg-teal-400 px-5 py-2 font-medium text-black hover:bg-teal-300 disabled:opacity-50"
      >
        {starting ? 'Starting…' : 'Process this playlist'}
      </button>
    </div>
  )
}

export default ProcessingStatus
