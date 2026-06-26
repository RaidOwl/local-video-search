import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, SearchResult } from '../services/api'

function fmt(s: number) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2,'0')}`
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true); setSearched(true)
    try { setResults((await api.search(query)).results) } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  return (
    <div className="page">
      <div style={{ maxWidth: 680, margin: '0 auto', paddingTop: '3rem' }}>
        <h1 style={{ textAlign: 'center', marginBottom: '0.5rem' }}>Search Your Videos</h1>
        <p style={{ textAlign: 'center', color: 'var(--muted)', fontSize: '0.9rem', marginBottom: '2rem' }}>
          Ask anything — searches both visual content and spoken words
        </p>

        <form onSubmit={handleSearch} className="search-bar">
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="e.g. 'what video was I talking about the AMD Strix Halo?'"
            style={{ fontSize: '1rem', padding: '0.75rem 1rem' }}
            autoFocus
          />
          <button type="submit" disabled={loading} style={{ padding: '0.75rem 1.4rem' }}>
            {loading ? <span className="spinner" /> : 'Search'}
          </button>
        </form>

        {searched && !loading && results.length === 0 && (
          <div className="empty" style={{ paddingTop: '2rem' }}>
            <h3>No results</h3>
            <p>Try different search terms or make sure videos have been analyzed.</p>
          </div>
        )}

        {results.length > 0 && (
          <div style={{ marginTop: '1.5rem' }}>
            <p style={{ color: 'var(--muted)', fontSize: '0.82rem', marginBottom: '1rem' }}>
              {results.length} video{results.length !== 1 ? 's' : ''} matched
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {results.map(r => (
                <Link to={`/videos/${r.video_id}?q=${encodeURIComponent(query)}`} key={r.video_id}
                  style={{ textDecoration: 'none' }}>
                  <div className="video-card" style={{ flexDirection: 'row', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>{r.name}</div>
                      <div className="video-meta">
                        <span>{fmt(r.duration)}</span>
                        <span>{r.matching_frames} matching segment{r.matching_frames !== 1 ? 's' : ''}</span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right', flexShrink: 0 }}>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent)' }}>
                        {Math.round(r.score * 100)}%
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>relevance</div>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
