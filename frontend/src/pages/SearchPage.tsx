import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, SearchResult } from '../services/api'
import { fmt } from '../utils'
import { Mic, Film, AlertTriangle } from 'lucide-react'

const SS_QUERY = 'vss_search_query'
const SS_RESULTS = 'vss_search_results'
const SS_THRESHOLD = 'vss_search_threshold'
const SS_SEARCH_VIDEO = 'vss_search_video'
const SS_SEARCH_AUDIO = 'vss_search_audio'
const SS_ENHANCED = 'vss_search_enhanced'

// BGE-M3 cosine similarities for semantic matches typically fall in 0.5–0.75,
// which feels low to users. Apply a power curve so strong matches display as 80%+.
const normalizeScore = (s: number) => 1 - Math.pow(Math.max(0, 1 - s), 1.5)
const rawThreshold = (t: number) => 1 - Math.pow(Math.max(0, 1 - t), 1 / 1.5)

export default function SearchPage() {
  const [query, setQuery] = useState(() => sessionStorage.getItem(SS_QUERY) ?? '')
  const [results, setResults] = useState<SearchResult[]>(() => {
    try { return JSON.parse(sessionStorage.getItem(SS_RESULTS) ?? '[]') } catch { return [] }
  })
  const [threshold, setThreshold] = useState(() => Number(sessionStorage.getItem(SS_THRESHOLD) ?? '0.50'))
  const [searchVideo, setSearchVideo] = useState(() => sessionStorage.getItem(SS_SEARCH_VIDEO) !== '0')
  const [searchAudio, setSearchAudio] = useState(() => sessionStorage.getItem(SS_SEARCH_AUDIO) !== '0')
  const [enhanced, setEnhanced] = useState(() => sessionStorage.getItem(SS_ENHANCED) === '1')
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(() => !!sessionStorage.getItem(SS_RESULTS))
  const [lastSearchedAudio, setLastSearchedAudio] = useState(false)
  const navigate = useNavigate()

  const sources = searchVideo && searchAudio ? 'both' : searchVideo ? 'video' : searchAudio ? 'audio' : 'both'

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true); setSearched(true); setLastSearchedAudio(searchAudio)
    try {
      const res = (await api.search(query, undefined, rawThreshold(threshold), sources, enhanced)).results
      setResults(res)
      sessionStorage.setItem(SS_QUERY, query)
      sessionStorage.setItem(SS_RESULTS, JSON.stringify(res))
      sessionStorage.setItem(SS_THRESHOLD, String(threshold))
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  const hasAudioMatches = results.some(r => r.segments.some(s => s.caption.startsWith('[transcript]')))
  const showNoTranscriptWarning = searched && !loading && lastSearchedAudio && !hasAudioMatches

  return (
    <div className="page" style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{
        maxWidth: 680, margin: '0 auto', width: '100%',
        ...(searched
          ? { paddingTop: '2rem' }
          : { flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', paddingBottom: '10rem' }
        )
      }}>
        {!searched && (
          <>
            <h1 style={{
              textAlign: 'center', marginBottom: '0.5rem',
              background: 'linear-gradient(135deg, var(--text) 20%, var(--accent-hover) 100%)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              fontSize: '2.4rem', letterSpacing: '-0.03em', lineHeight: 1.2
            }}>
              Local Video Search
            </h1>
            <p style={{ textAlign: 'center', color: 'var(--muted)', fontSize: '0.95rem', marginBottom: '2.5rem' }}>
              Ask anything — searches both visual content and spoken words
            </p>
          </>
        )}

        <form onSubmit={handleSearch} className="search-bar">
          <div style={{ position: 'relative', flex: 1, display: 'flex' }}>
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="e.g. 'what video was I talking about the AMD Strix Halo?'"
              style={{ fontSize: '1rem', padding: '0.75rem 1rem', paddingRight: query ? '2.25rem' : '1rem', flex: 1 }}
              autoFocus
            />
            {query && (
              <button type="button" onClick={() => {
                setQuery('')
                setResults([])
                setSearched(false)
                sessionStorage.removeItem(SS_QUERY)
                sessionStorage.removeItem(SS_RESULTS)
              }} style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', fontSize: '1.1rem', padding: '0.2rem 0.3rem', lineHeight: 1, opacity: 0.6, boxShadow: 'none' }}
                title="Clear search">×</button>
            )}
          </div>
          <button type="submit" disabled={loading} style={{ padding: '0.75rem 1.4rem' }}>
            {loading ? <span className="spinner" /> : 'Search'}
          </button>
        </form>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.75rem', color: 'var(--muted)', fontSize: '0.82rem' }}>
          <span style={{ flexShrink: 0 }}>Min relevance</span>
          <input
            type="range" min={0.1} max={0.9} step={0.05}
            value={threshold}
            onChange={e => setThreshold(Number(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--accent)', cursor: 'pointer' }}
          />
          <span style={{ flexShrink: 0, minWidth: 32, textAlign: 'right', color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
            {Math.round(threshold * 100)}%
          </span>
          <span style={{ width: 1, height: 16, background: 'var(--border)', flexShrink: 0 }} />
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', flexShrink: 0 }}>
            <input type="checkbox" checked={searchVideo} onChange={e => { setSearchVideo(e.target.checked); sessionStorage.setItem(SS_SEARCH_VIDEO, e.target.checked ? '1' : '0') }} />
            Video
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', flexShrink: 0 }}>
            <input type="checkbox" checked={searchAudio} onChange={e => { setSearchAudio(e.target.checked); sessionStorage.setItem(SS_SEARCH_AUDIO, e.target.checked ? '1' : '0') }} />
            Audio
          </label>
          {showNoTranscriptWarning && (
            <span title="Enable audio transcription from the video detail page" style={{ fontSize: '0.78rem', color: 'var(--warning)', flexShrink: 0, display: 'inline-flex', alignItems: 'center', gap: 4 }}><AlertTriangle size={12} /> No transcripts found</span>
          )}
          <span style={{ width: 1, height: 16, background: 'var(--border)', flexShrink: 0 }} />
          <label title="Use the AI to generate query variations before searching. Slower but finds more." style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', flexShrink: 0, color: enhanced ? 'var(--accent)' : 'var(--muted)' }}>
            <input type="checkbox" checked={enhanced} onChange={e => { setEnhanced(e.target.checked); sessionStorage.setItem(SS_ENHANCED, e.target.checked ? '1' : '0') }} />
            Enhanced
          </label>
        </div>

        {searched && !loading && results.length === 0 && (
          <div className="empty" style={{ paddingTop: '2rem' }}>
            <h3>No results</h3>
            <p>Try different search terms or make sure videos have been analyzed.</p>
            {showNoTranscriptWarning && <p style={{ color: 'var(--warning, #f59e0b)', fontSize: '0.85rem' }}>Audio search is on but no videos appear to have been transcribed yet. Open a video and use "Re-transcribe audio" to add audio search.</p>}
          </div>
        )}

        {results.length > 0 && (
          <div style={{ marginTop: '1.5rem' }}>
            <p style={{ color: 'var(--muted)', fontSize: '0.82rem', marginBottom: '1rem' }}>
              {results.length} video{results.length !== 1 ? 's' : ''} matched
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {results.map(r => (
                <div key={r.video_id} style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden', background: 'var(--surface)', boxShadow: '0 1px 3px rgba(0,0,0,0.3), 0 2px 8px rgba(0,0,0,0.15)' }}>
                  {/* Video header row */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.9rem 1rem' }}>
                    <div style={{ flex: 1 }}>
                      <Link to={`/videos/${r.video_id}?q=${encodeURIComponent(query)}`} style={{ textDecoration: 'none' }}>
                        <div style={{ fontWeight: 600, color: 'var(--accent)', fontSize: '1rem' }}>{r.name}</div>
                      </Link>
                      <div className="video-meta" style={{ marginTop: '0.2rem' }}>
                        <span>{fmt(r.duration)}</span>
                        <span>{r.matching_frames} matching segment{r.matching_frames !== 1 ? 's' : ''}</span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right', flexShrink: 0 }}>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent)' }}>
                        {Math.round(normalizeScore(r.score) * 100)}%
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>relevance</div>
                    </div>
                  </div>

                  {/* Matching segments */}
                  {r.segments && r.segments.length > 0 && (
                    <div style={{ borderTop: '1px solid var(--border)', maxHeight: 320, overflowY: 'auto' }}>
                      {r.segments.map((seg, i) => (
                        <div
                          key={i}
                          onClick={() => navigate(`/videos/${r.video_id}?t=${Math.floor(seg.start_time)}`)}
                          style={{
                            display: 'flex', alignItems: 'center', gap: '0.75rem',
                            padding: '0.6rem 1rem',
                            borderBottom: i < r.segments.length - 1 ? '1px solid var(--border)' : 'none',
                            cursor: 'pointer',
                            transition: 'background 0.1s',
                          }}
                          onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface2)')}
                          onMouseLeave={e => (e.currentTarget.style.background = '')}
                        >
                          {seg.frame_path && (
                            <img
                              src={`/api/frames/${r.video_id}/${encodeURIComponent(seg.frame_path.split('/').pop()!)}`}
                              alt="" loading="lazy"
                              style={{ width: 80, height: 52, objectFit: 'cover', borderRadius: 5, flexShrink: 0, background: 'var(--surface2)' }}
                            />
                          )}
                          <span title={seg.caption.startsWith('[transcript]') ? 'Audio transcript' : 'Visual caption'}
                            style={{ fontSize: '0.7rem', flexShrink: 0, opacity: 0.5 }}>
                            {seg.caption.startsWith('[transcript]') ? <Mic size={11} /> : <Film size={11} />}
                          </span>
                          <span style={{
                            fontFamily: 'monospace', fontSize: '0.78rem',
                            color: 'var(--accent)', flexShrink: 0,
                            background: 'rgba(99,102,241,0.12)', borderRadius: 4,
                            padding: '0.1rem 0.4rem', alignSelf: 'flex-start', marginTop: '0.1rem'
                          }}>
                            {fmt(seg.start_time)}
                          </span>
                          <span style={{ fontSize: '0.82rem', color: 'var(--text)', lineHeight: 1.4, flex: 1 }}>
                            {seg.caption.replace(/^\[transcript\]\s*/i, '')}
                          </span>
                          <span style={{ fontSize: '0.72rem', color: 'var(--muted)', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
                            {Math.round(normalizeScore(seg.score) * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
