import { useState, useEffect, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, SearchResult } from '../services/api'
import { fmt } from '../utils'
import { Mic, Film, SlidersHorizontal } from 'lucide-react'

const SS_QUERY = 'vss_search_query'
const SS_RESULTS = 'vss_search_results'
const SS_TIER = 'vss_search_tier'
const SS_SEARCH_VIDEO = 'vss_search_video'
const SS_SEARCH_AUDIO = 'vss_search_audio'
const SS_FILTERS = 'vss_search_filters'

// Relevance is the reranker's raw score (0–1; ~0.50 = "the model rejects this").
// Rather than dress that up as a misleading precise percentage, we expose four
// discrete confidence bands. Each slider stop maps to a FIXED raw threshold sent
// straight to the API, and every result is badged with the band its raw score
// falls into — no curve, no score→percentage translation.
// Ordered loosest → strictest (slider index 0–3). `pick` = the slider-stop label
// (filter intent); `badge` = the per-result quality label.
type RelTier = { min: number; pick: string; badge: string; color: string }
const REL_TIERS: RelTier[] = [
  { min: 0.50, pick: 'Nearly everything', badge: 'Loosely related',  color: '#94a3b8' },
  { min: 0.60, pick: 'Somewhat matches',  badge: 'Somewhat matches', color: '#fbbf24' },
  { min: 0.68, pick: 'Mostly matches',    badge: 'Mostly matches',   color: '#818cf8' },
  { min: 0.76, pick: 'Exact match',       badge: 'Exact match',      color: '#34d399' },
]
const tierForScore = (s: number): RelTier => {
  for (let i = REL_TIERS.length - 1; i > 0; i--) if (s >= REL_TIERS[i].min) return REL_TIERS[i]
  return REL_TIERS[0]
}

// Concise, teach-by-example placeholders. Describe what's on screen or said —
// short noun phrases rerank far better than verbose meta questions like
// "show me where I was talking about...".
const SEARCH_EXAMPLES = [
  'a man standing next to a red car',
  'kvm switch on a desk',
  'someone explaining pricing',
  'a laptop showing a blue screen',
  'person holding a coffee mug',
  'talking about graphics cards',
]

export default function SearchPage() {
  const [query, setQuery] = useState(() => sessionStorage.getItem(SS_QUERY) ?? '')
  const [results, setResults] = useState<SearchResult[]>(() => {
    try { return JSON.parse(sessionStorage.getItem(SS_RESULTS) ?? '[]') } catch { return [] }
  })
  const [tierIdx, setTierIdx] = useState(() => {
    const v = parseInt(sessionStorage.getItem(SS_TIER) ?? '', 10)
    return Number.isInteger(v) && v >= 0 && v <= 3 ? v : 1
  })
  const [searchVideo, setSearchVideo] = useState(() => sessionStorage.getItem(SS_SEARCH_VIDEO) !== '0')
  const [searchAudio, setSearchAudio] = useState(() => sessionStorage.getItem(SS_SEARCH_AUDIO) !== '0')
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(() => !!sessionStorage.getItem(SS_RESULTS))
  const navigate = useNavigate()

  // Rotate the example placeholder so users learn the concise phrasing pattern.
  // Uses an overlay span (below) so we can cross-fade — native placeholders can't animate.
  const [phIdx, setPhIdx] = useState(() => Math.floor(Math.random() * SEARCH_EXAMPLES.length))
  const [phVisible, setPhVisible] = useState(true)
  useEffect(() => {
    const cycle = setInterval(() => {
      setPhVisible(false)
      setTimeout(() => {
        setPhIdx(i => (i + 1) % SEARCH_EXAMPLES.length)
        setPhVisible(true)
      }, 350)
    }, 2200)
    return () => clearInterval(cycle)
  }, [])

  // --- optional metadata filters (explicit, opt-in via the popup) ---
  type Filters = { mediaTypes: string[]; resolution: string; dateFrom: string; dateTo: string }
  const emptyFilters: Filters = { mediaTypes: [], resolution: '', dateFrom: '', dateTo: '' }
  const [filters, setFilters] = useState<Filters>(() => {
    try { return { ...emptyFilters, ...JSON.parse(sessionStorage.getItem(SS_FILTERS) ?? '{}') } } catch { return emptyFilters }
  })
  const [facets, setFacets] = useState<{ media_types: string[]; resolutions: string[] }>({ media_types: [], resolutions: [] })
  const [showFilters, setShowFilters] = useState(false)
  const filterRef = useRef<HTMLDivElement>(null)

  const activeFilterCount =
    filters.mediaTypes.length + (filters.resolution ? 1 : 0) + (filters.dateFrom ? 1 : 0) + (filters.dateTo ? 1 : 0)

  // Filters-only browse (no text query) returns synthetic 100% scores, so the
  // relevance badges are meaningless and hidden in that mode.
  const isBrowse = searched && results.length > 0 && !query.trim()

  useEffect(() => {
    api.metadataFacets().then(setFacets).catch(() => {})
  }, [])

  // Close the popover when clicking outside it.
  useEffect(() => {
    if (!showFilters) return
    const onDown = (e: MouseEvent) => {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) setShowFilters(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [showFilters])

  const updateFilters = (next: Filters) => {
    setFilters(next)
    sessionStorage.setItem(SS_FILTERS, JSON.stringify(next))
  }
  const toggleMediaType = (t: string) => {
    const has = filters.mediaTypes.includes(t)
    updateFilters({ ...filters, mediaTypes: has ? filters.mediaTypes.filter(x => x !== t) : [...filters.mediaTypes, t] })
  }
  const clearFilters = () => updateFilters(emptyFilters)

  const sources = searchVideo && searchAudio ? 'both' : searchVideo ? 'video' : searchAudio ? 'audio' : 'both'

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!query.trim() && activeFilterCount === 0) return
    setLoading(true); setSearched(true)
    try {
      const res = (await api.search(query, undefined, REL_TIERS[tierIdx].min, sources, {
        media_types: filters.mediaTypes.join(','),
        resolution: filters.resolution,
        date_from: filters.dateFrom,
        date_to: filters.dateTo,
      })).results
      setResults(res)
      sessionStorage.setItem(SS_QUERY, query)
      sessionStorage.setItem(SS_RESULTS, JSON.stringify(res))
      sessionStorage.setItem(SS_TIER, String(tierIdx))
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

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
              placeholder=""
              style={{ fontSize: '1rem', padding: '0.75rem 1rem', paddingRight: query ? '2.25rem' : '1rem', flex: 1 }}
              autoFocus
            />
            {!query && (
              <span
                aria-hidden
                style={{
                  position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)',
                  color: 'var(--muted)', fontSize: '1rem', pointerEvents: 'none',
                  opacity: phVisible ? 0.65 : 0, transition: 'opacity 0.35s ease',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  maxWidth: 'calc(100% - 2rem)',
                }}
              >
                e.g. “{SEARCH_EXAMPLES[phIdx]}”
              </span>
            )}
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
            type="range" min={0} max={3} step={1}
            value={tierIdx}
            onChange={e => setTierIdx(Number(e.target.value))}
            style={{ flex: 1, accentColor: REL_TIERS[tierIdx].color, cursor: 'pointer' }}
          />
          <span style={{ flexShrink: 0, minWidth: 132, display: 'inline-flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.35rem', color: REL_TIERS[tierIdx].color, fontWeight: 600 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: REL_TIERS[tierIdx].color, flexShrink: 0 }} />
            {REL_TIERS[tierIdx].pick}
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

          <span style={{ width: 1, height: 16, background: 'var(--border)', flexShrink: 0 }} />
          <div ref={filterRef} style={{ position: 'relative', flexShrink: 0 }}>
            <button
              type="button"
              onClick={() => setShowFilters(s => !s)}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.3rem',
                background: activeFilterCount ? 'var(--accent)' : 'none',
                color: activeFilterCount ? '#fff' : 'var(--muted)',
                border: '1px solid var(--border)', borderRadius: 6,
                padding: '0.2rem 0.5rem', cursor: 'pointer', fontSize: '0.78rem', boxShadow: 'none',
              }}
              title="Filter by media type, resolution, or date"
            >
              <SlidersHorizontal size={13} />
              Filters{activeFilterCount ? ` (${activeFilterCount})` : ''}
            </button>

            {showFilters && (
              <div style={{
                position: 'absolute', top: 'calc(100% + 6px)', right: 0, zIndex: 20,
                width: 260, maxWidth: 'calc(100vw - 2rem)', boxSizing: 'border-box',
                background: 'var(--surface)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius)', boxShadow: '0 6px 24px rgba(0,0,0,0.35)',
                padding: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.85rem',
                textAlign: 'left',
              }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Media type</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                    {facets.media_types.length === 0 && <span style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>None available</span>}
                    {facets.media_types.map(t => {
                      const on = filters.mediaTypes.includes(t)
                      return (
                        <button key={t} type="button" onClick={() => toggleMediaType(t)}
                          style={{
                            fontSize: '0.74rem', padding: '0.18rem 0.5rem', borderRadius: 5, cursor: 'pointer',
                            border: '1px solid var(--border)', boxShadow: 'none',
                            background: on ? 'var(--accent)' : 'transparent', color: on ? '#fff' : 'var(--text)',
                          }}>
                          {t.toUpperCase()}
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Minimum resolution</div>
                  <select value={filters.resolution} onChange={e => updateFilters({ ...filters, resolution: e.target.value })}
                    style={{ width: '100%', padding: '0.35rem 0.5rem', fontSize: '0.8rem', background: 'var(--surface2)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 6 }}>
                    <option value="">Any</option>
                    {facets.resolutions.map(r => <option key={r} value={r}>{r === '4k' ? '4K' : r}+</option>)}
                  </select>
                </div>

                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Date created</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.76rem', color: 'var(--muted)' }}>
                      <span style={{ width: 32, flexShrink: 0 }}>From</span>
                      <input type="date" value={filters.dateFrom} onChange={e => updateFilters({ ...filters, dateFrom: e.target.value })}
                        style={{ flex: 1, minWidth: 0, padding: '0.3rem 0.4rem', fontSize: '0.76rem', background: 'var(--surface2)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 6 }} />
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.76rem', color: 'var(--muted)' }}>
                      <span style={{ width: 32, flexShrink: 0 }}>To</span>
                      <input type="date" value={filters.dateTo} onChange={e => updateFilters({ ...filters, dateTo: e.target.value })}
                        style={{ flex: 1, minWidth: 0, padding: '0.3rem 0.4rem', fontSize: '0.76rem', background: 'var(--surface2)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 6 }} />
                    </label>
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border)', paddingTop: '0.6rem' }}>
                  <button type="button" onClick={clearFilters} disabled={activeFilterCount === 0}
                    style={{ background: 'none', border: 'none', color: 'var(--muted)', cursor: activeFilterCount ? 'pointer' : 'default', fontSize: '0.76rem', padding: 0, boxShadow: 'none', opacity: activeFilterCount ? 1 : 0.5 }}>
                    Clear
                  </button>
                  <button type="button" onClick={() => { setShowFilters(false); handleSearch() }}
                    style={{ padding: '0.3rem 0.8rem', fontSize: '0.78rem' }}>
                    Apply
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

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
                      {!isBrowse && (() => {
                        const t = tierForScore(r.score)
                        return (
                          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                            <span style={{ width: 9, height: 9, borderRadius: '50%', background: t.color, flexShrink: 0 }} />
                            <span style={{ fontSize: '0.9rem', fontWeight: 600, color: t.color }}>{t.badge}</span>
                          </div>
                        )
                      })()}
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
                          {!isBrowse && (() => {
                            const t = tierForScore(seg.score)
                            return (
                              <span title={t.badge} style={{ width: 9, height: 9, borderRadius: '50%', background: t.color, flexShrink: 0 }} />
                            )
                          })()}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {searched && !loading && results.length === 0 && (
          <div style={{ marginTop: '2.5rem', textAlign: 'center', color: 'var(--muted)' }}>
            <p style={{ fontSize: '1rem', color: 'var(--text)', marginBottom: '0.4rem' }}>No matches found</p>
            <p style={{ fontSize: '0.85rem' }}>
              Try lowering the minimum relevance, rephrasing your search, or checking that the videos have been analyzed.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
