import { useState } from 'react'
import { queueTrack } from './TrackActions'

type Props = {
  trackIds: string[]
}

function QueueAllButton({ trackIds }: Props) {
  const [status, setStatus] = useState<'idle' | 'running' | 'done'>('idle')
  const [succeeded, setSucceeded] = useState(0)
  const [attempted, setAttempted] = useState(0)

  async function handleClick() {
    setStatus('running')
    setSucceeded(0)
    setAttempted(0)
    let successCount = 0
    // Awaiting these one at a time instead of firing them all with
    // Promise.all - Spotify's queue is a real ordered list, and network
    // timing shouldn't be what decides what order things land in. A bit
    // slower, but the queue ends up matching what was actually clicked.
    for (const trackId of trackIds) {
      const ok = await queueTrack(trackId)
      if (ok) successCount++
      setSucceeded(successCount)
      setAttempted((prev) => prev + 1)
    }
    setStatus('done')
  }

  if (trackIds.length === 0) {
    return null
  }

  if (status === 'done') {
    const allSucceeded = succeeded === trackIds.length
    return (
      <p className={`text-xs ${allSucceeded ? 'text-teal-300' : 'text-amber-400'}`}>
        {allSucceeded
          ? `✓ Queued all ${succeeded}`
          : `Queued ${succeeded} of ${trackIds.length}${succeeded === 0 ? ' - nothing seems to be actively playing' : ' - the rest may have failed for the same reason'}`}
      </p>
    )
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={status === 'running'}
      className="rounded-full bg-white/10 px-3 py-1.5 text-xs text-white transition hover:bg-white/20 disabled:opacity-50"
    >
      {status === 'running' ? `Queueing… ${attempted}/${trackIds.length}` : `Queue all (${trackIds.length})`}
    </button>
  )
}

export default QueueAllButton
