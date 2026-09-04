import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import QueueAllButton from './QueueAllButton'
import TrackActions from './TrackActions'
import VoiceButton from './VoiceButton'

type SearchResult = {
  track_id: string
  name: string
  artist: string
  genre_bucket: string | null
  description: string
  match_type: string
}

type Detected = {
  genre: string | null
  artist: string | null
  mood: string | null
  reference_track: { name: string; artist: string } | null
}

type SearchResponse = {
  query: string
  results: SearchResult[]
  detected: Detected
  exact_match_count: number
}

// Mirrors backend/search/hybrid.py's match_type values so it's visible
// *why* a result is here, the same reasoning that kept match_type in the
// API response in the first place.
const MATCH_LABELS: Record<string, string> = {
  rrf: 'vibe match',
  backfill: 'related',
  genre_locked: 'genre match',
  artist_locked: 'artist match',
  'genre+artist_locked': 'genre + artist match',
  artist_only_backfill: 'same artist, different genre',
  genre_only_backfill: 'same genre, different artist',
  artist_genre_backfill: 'similar genre',
}

// A three-tier visual hierarchy: a locked exact match is the strongest
// signal (solid accent), a partial-honor backfill tier is the middle
// signal (outlined accent), and plain vibe/pure-backfill results share
// the same muted, default look - visually, "nothing special was locked
// for this one."
function badgeClass(matchType: string): string {
  const locked = new Set(['genre_locked', 'artist_locked', 'genre+artist_locked'])
  const partial = new Set(['artist_only_backfill', 'genre_only_backfill', 'artist_genre_backfill'])
  if (locked.has(matchType)) return 'bg-teal-400 text-black'
  if (partial.has(matchType)) return 'border border-teal-400 text-teal-300'
  return 'bg-white/10 text-neutral-300'
}

function detectedLabel(detected: Detected): string {
  const parts: string[] = []
  if (detected.genre) parts.push(detected.genre)
  if (detected.artist) parts.push(detected.artist)
  if (detected.reference_track) parts.push(`"${detected.reference_track.name}"`)
  return parts.join(' + ')
}

// Honest about what actually happened, not just a raw count - a genre +
// artist lock silently falling through to 100% backfill (confirmed common:
// 94.8% of artist x genre combos have zero overlap, see ARCHITECTURE.md)
// used to look identical to a fully successful search before this existed.
function buildStatusText(data: SearchResponse): string {
  const { results, query, detected, exact_match_count } = data
  const label = detectedLabel(detected)
  const base = `${results.length} results for "${query}"`
  if (!label || exact_match_count === results.length) return base
  if (exact_match_count === 0) return `No exact matches for ${label} - showing related songs instead.`
  const relatedCount = results.length - exact_match_count
  const matchWord = exact_match_count === 1 ? 'match' : 'matches'
  return `${exact_match_count} exact ${matchWord} for ${label}, plus ${relatedCount} related.`
}

function runSearch(query: string, playlistId: string): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: query, playlist_id: playlistId })
  return fetch(`/search?${params}`).then((response) => {
    if (!response.ok) throw new Error(`Search failed (${response.status})`)
    return response.json()
  })
}

type Props = {
  playlistId: string
}

function Search({ playlistId }: Props) {
  // Two separate pieces of state, deliberately not one: inputValue is the
  // controlled input's live value, updating on every keystroke. submittedQuery
  // only changes on form submit, and it's THAT value driving the actual
  // fetch (via queryKey below) - typing doesn't search, submitting does.
  const [inputValue, setInputValue] = useState('')
  const [submittedQuery, setSubmittedQuery] = useState('')

  // Search is a read (safe, cacheable, doesn't change server state) - so
  // useQuery is the right tool even though a person triggers it, not an
  // automatic mount or poll. enabled: false until a real query has been
  // submitted means this doesn't fire on first render with an empty string.
  const { data, isFetching, isError, error } = useQuery({
    queryKey: ['search', playlistId, submittedQuery],
    queryFn: () => runSearch(submittedQuery, playlistId),
    enabled: submittedQuery !== '',
  })

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const trimmed = inputValue.trim()
    if (trimmed) setSubmittedQuery(trimmed)
  }

  return (
    <div className="mt-8">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={inputValue}
          onChange={(event) => setInputValue(event.target.value)}
          placeholder="describe a vibe… e.g. calm rainy day songs"
          className="flex-1 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-white placeholder-neutral-500 focus:border-teal-400 focus:outline-none"
        />
        <VoiceButton
          onResult={(transcript) => {
            // Only fills the box - deliberately does NOT set
            // submittedQuery. Speech recognition is genuinely error-prone
            // (confirmed by trying it for real), so auto-searching the
            // instant it finishes gives no chance to notice and fix a
            // bad transcription before it's actually searched. This
            // makes voice behave exactly like typing: it produces text
            // in the box, and a real click on Search (or pressing enter)
            // is what submits it, same as every other way of getting
            // text into this input.
            setInputValue(transcript)
          }}
        />
        <button
          type="submit"
          disabled={isFetching}
          className="rounded-full bg-teal-400 px-5 py-2 font-medium text-black hover:bg-teal-300 disabled:opacity-50"
        >
          Search
        </button>
      </form>

      {/* The reranker is the dominant cost of every search (~4s, confirmed
          by profiling in the backend work) - a plain "Searching…" with no
          fake progress bar is honest about that rather than pretending
          it's instant. */}
      {isFetching && <p className="mt-4 text-neutral-400">Searching…</p>}
      {isError && <p className="mt-4 text-red-400">{(error as Error).message || 'Something went wrong.'}</p>}
      {data && !isFetching && (
        <div className="mt-4 flex items-center justify-between gap-3">
          <p className="text-neutral-300">{buildStatusText(data)}</p>
          {data.results.length > 0 && <QueueAllButton trackIds={data.results.map((r) => r.track_id)} />}
        </div>
      )}

      {data && !isFetching && (
        <ul className="mt-4 space-y-3">
          {data.results.length === 0 && <li className="text-neutral-400">No matches found.</li>}
          {data.results.map((track) => (
            <li key={track.track_id} className="rounded-xl border border-white/10 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-medium">{track.name}</p>
                  <p className="text-sm text-neutral-400">{track.artist}</p>
                </div>
                <div className="flex flex-shrink-0 items-start gap-1.5">
                  {track.genre_bucket && (
                    <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs text-neutral-300">
                      {track.genre_bucket}
                    </span>
                  )}
                  <span className={`rounded-full px-2.5 py-0.5 text-xs ${badgeClass(track.match_type)}`}>
                    {MATCH_LABELS[track.match_type] ?? track.match_type}
                  </span>
                  <TrackActions trackId={track.track_id} />
                </div>
              </div>
              {track.description && (
                <p className="mt-2 text-sm text-neutral-400">{track.description}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default Search
