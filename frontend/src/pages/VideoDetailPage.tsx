import { useState, useEffect, useRef } from 'react'
import { useParams, Link, useSearchParams } from 'react-router-dom'
import { api, Video, Segment, VideoFrame } from '../services/api'
import { fmt } from '../utils'
import { Pencil, Play, Square, RotateCcw, X, Download, Check, Mic, Film, Sparkles } from 'lucide-react'
import { AnalyzeModal } from './VideosPage'

export default function VideoDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [searchParams] = useSearchParams()
  const seekTo = searchParams.get('t') ? Number(searchParams.get('t')) : null
  const videoRef = useRef<HTMLVideoElement>(null)
  const [video, setVideo] = useState<Video | null>(null)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [segments, setSegments] = useState<Segment[]>([])
  const [searching, setSearching] = useState(false)
  const [summarizing, setSummarizing] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [editingName, setEditingName] = useState(false)
  const [nameInput, setNameInput] = useState('')
  const [tagInput, setTagInput] = useState('')
  const [frames, setFrames] = useState<VideoFrame[]>([])
  const [showFramePicker, setShowFramePicker] = useState(false)
  const [savingThumbnail, setSavingThumbnail] = useState(false)
  const [thumbnailSaved, setThumbnailSaved] = useState(false)
  const [transcript, setTranscript] = useState<Array<{ id: string; timestamp: number; text: string }> | null>(null)
  const [fused, setFused] = useState<Array<{ id: string; timestamp: number; caption: string; path: string | null }>>([])
  const [segFilter, setSegFilter] = useState<'all' | 'visual' | 'speech' | 'fused'>('all')
  const [editingSegId, setEditingSegId] = useState<string | null>(null)
  const [editingText, setEditingText] = useState('')
  const [savingSegId, setSavingSegId] = useState<string | null>(null)
  const [showAnalyzeModal, setShowAnalyzeModal] = useState(false)
  const pollRef = useRef<number | null>(null)

  // Seek to timestamp from ?t= param once video metadata is loaded
  useEffect(() => {
    if (seekTo === null || !videoRef.current) return
    const el = videoRef.current
    const doSeek = () => { el.currentTime = seekTo }
    if (el.readyState >= 1) { doSeek() } else { el.addEventListener('loadedmetadata', doSeek, { once: true }) }
  }, [seekTo, video?.id])

  const loadVideo = async () => {
    if (!id) return
    try { setVideo(await api.getVideo(id)) } catch { /* ignore */ } finally { setLoading(false) }
  }

  useEffect(() => {
    loadVideo()
  }, [id])

  // Load frames when video is analyzed
  useEffect(() => {
    if (!id || !video || (video.status !== 'analyzed' && video.status !== 'retranscribing')) return
    api.getVideoFrames(id).then(d => setFrames(d.frames)).catch(() => {})
    api.getTranscript(id).then(d => setTranscript(d.segments)).catch(() => {})
    api.getFused(id).then(d => setFused(d.segments)).catch(() => {})
  }, [id, video?.status])

  const setThumbnail = async (framePath: string) => {
    if (!id || !video || savingThumbnail) return
    setSavingThumbnail(true)
    try {
      const updated = await api.updateVideo(id, { thumbnail_path: framePath })
      setVideo(updated)
      setThumbnailSaved(true)
      setTimeout(() => { setThumbnailSaved(false); setShowFramePicker(false) }, 1000)
    } catch {
      alert('Failed to save thumbnail — check console for details.')
    } finally {
      setSavingThumbnail(false)
    }
  }

  // Poll while processing
  useEffect(() => {
    if (!video) return
    if (video.status === 'processing' || video.status === 'transcribing' || video.status === 'fusing') {
      pollRef.current = window.setInterval(async () => {
        const v = await api.getVideo(id!).catch(() => null)
        if (!v) return
        setVideo(v)
        if (v.status !== 'processing' && v.status !== 'transcribing' && v.status !== 'fusing') clearInterval(pollRef.current!)
      }, 4000)
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [video?.status])

  const handleSummarize = async () => {
    if (!id || !video) return
    setSummarizing(true)
    try {
      await api.summarizeVideo(id)
      // Poll for result
      const poll = setInterval(async () => {
        const v = await api.getVideo(id).catch(() => null)
        if (!v) return
        if (v.summary) { setVideo(v); clearInterval(poll); setSummarizing(false) }
      }, 3000)
      setTimeout(() => { clearInterval(poll); setSummarizing(false) }, 300000)
    } catch { setSummarizing(false) }
  }

  // On load: if no summary but status is analyzed, check if one was generated while away
  useEffect(() => {
    if (!video || video.summary || video.status !== 'analyzed') return
    api.getVideo(id!).then(v => { if (v.summary) setVideo(v) }).catch(() => null)
  }, [video?.id])

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || !id) return
    setSearching(true)
    try {
      const data = await api.searchSegments(id, query)
      setSegments(data.segments)
    } catch { /* ignore */ }
    finally { setSearching(false) }
  }

  const handleReprocess = () => {
    setShowAnalyzeModal(true)
  }

  const handleCancel = async () => {
    if (!id || cancelling) return
    setCancelling(true)
    try {
      await api.cancelAnalysis(id)
      setVideo(v => v ? { ...v, status: 'error' } : v)
    } catch { /* ignore */ } finally { setCancelling(false) }
  }

  const saveName = async () => {
    if (!id || !video) return
    const updated = await api.updateVideo(id, { display_name: nameInput.trim() || null as unknown as string })
    setVideo(updated); setEditingName(false)
  }

  const addTag = async (e: React.KeyboardEvent) => {
    if (e.key !== 'Enter' || !tagInput.trim() || !id || !video) return
    e.preventDefault()
    const newTag = tagInput.trim()
    if (video.tags.includes(newTag)) { setTagInput(''); return }
    const updated = await api.updateVideo(id, { tags: [...video.tags, newTag] })
    setVideo(updated); setTagInput('')
  }

  const removeTag = async (tag: string) => {
    if (!id || !video) return
    const updated = await api.updateVideo(id, { tags: video.tags.filter(t => t !== tag) })
    setVideo(updated)
  }

  const [retranscribing, setRetranscribing] = useState(false)
  const [retranscribeElapsed, setRetranscribeElapsed] = useState(0)
  const retranscribeStartRef = useRef<number | null>(null)
  const retranscribeTickRef = useRef<number | null>(null)

  // Auto-reconnect timer/polling when navigating to a page already retranscribing
  useEffect(() => {
    if (!id || !video || video.status !== 'retranscribing' || retranscribing) return
    setRetranscribing(true)
    setRetranscribeElapsed(0)
    retranscribeStartRef.current = Date.now()
    retranscribeTickRef.current = window.setInterval(() => {
      setRetranscribeElapsed(Math.floor((Date.now() - retranscribeStartRef.current!) / 1000))
    }, 1000)
    const stopTick = () => {
      if (retranscribeTickRef.current) { clearInterval(retranscribeTickRef.current); retranscribeTickRef.current = null }
    }
    const poll = window.setInterval(async () => {
      const v = await api.getVideo(id).catch(() => null)
      if (!v || v.status === 'analyzed' || v.status === 'error' || v.status === 'ready') {
        clearInterval(poll)
        stopTick()
        setRetranscribing(false)
        api.getTranscript(id).then(d => setTranscript(d.segments)).catch(() => {})
      }
    }, 3000)
    const timeout = setTimeout(() => { clearInterval(poll); stopTick(); setRetranscribing(false) }, 600000)
    return () => { clearInterval(poll); clearTimeout(timeout); stopTick() }
  }, [id, video?.status]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleRetranscribe = async () => {
    if (!id || retranscribing) return
    setRetranscribing(true)
    setRetranscribeElapsed(0)
    retranscribeStartRef.current = Date.now()
    retranscribeTickRef.current = window.setInterval(() => {
      setRetranscribeElapsed(Math.floor((Date.now() - retranscribeStartRef.current!) / 1000))
    }, 1000)
    const stopTick = () => {
      if (retranscribeTickRef.current) { clearInterval(retranscribeTickRef.current); retranscribeTickRef.current = null }
    }
    try {
      await api.retranscribeAudio(id)
      const poll = window.setInterval(async () => {
        const v = await api.getVideo(id).catch(() => null)
        if (!v || v.status === 'analyzed' || v.status === 'error' || v.status === 'ready') {
          clearInterval(poll)
          stopTick()
          setRetranscribing(false)
          api.getTranscript(id).then(d => setTranscript(d.segments)).catch(() => {})
        }
      }, 3000)
      setTimeout(() => { clearInterval(poll); stopTick(); setRetranscribing(false) }, 600000)
    } catch {
      stopTick()
      setRetranscribing(false)
    }
  }

  const handleSegSave = async (segId: string, type: 'visual' | 'speech' | 'fused') => {
    if (!id) return
    setSavingSegId(segId)
    try {
      await api.updateFrameCaption(id, segId, editingText.trim())
      if (type === 'visual') {
        setFrames(prev => prev.map(f => f.id === segId ? { ...f, caption: editingText.trim() } : f))
      } else if (type === 'speech') {
        setTranscript(prev => prev ? prev.map(t => t.id === segId ? { ...t, text: editingText.trim() } : t) : prev)
      }
      setEditingSegId(null)
    } catch { /* ignore */ }
    finally { setSavingSegId(null) }
  }

  const allSegs = [
    ...frames.map(f => ({ id: f.id, timestamp: f.timestamp, caption: f.caption || '', type: 'visual' as const, frame_path: f.path as string | undefined })),
    ...(transcript || []).map(t => ({ id: t.id, timestamp: t.timestamp, caption: t.text, type: 'speech' as const, frame_path: undefined as string | undefined })),
    ...fused.map(f => ({ id: f.id, timestamp: f.timestamp, caption: f.caption || '', type: 'fused' as const, frame_path: f.path || undefined }))
  ].sort((a, b) => a.timestamp - b.timestamp)

  if (loading) return <div className="page" style={{ display:'flex', alignItems:'center', gap:'0.75rem', justifyContent:'center', marginTop:'4rem', color:'var(--muted)' }}><span className="spinner" /> Loading…</div>
  if (!video) return <div className="page"><p>Video not found.</p></div>

  return (
    <div className="page">
      <div className="breadcrumb"><Link to="/videos">← Videos</Link></div>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ flex: 1 }}>
          {editingName ? (
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <input value={nameInput} onChange={e => setNameInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && saveName()} style={{ fontSize: '1.2rem', fontWeight: 700, maxWidth: 400 }} autoFocus />
              <button onClick={saveName}>Save</button>
              <button className="ghost" onClick={() => setEditingName(false)}>Cancel</button>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <h1 style={{ fontSize: '1.4rem' }}>{video.display_name || video.name}</h1>
              <button className="icon-btn" onClick={() => { setNameInput(video.display_name || video.name); setEditingName(true) }} title="Rename"><Pencil size={14} /></button>
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <span style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>{fmt(video.duration)} duration</span>
            {video.frame_count > 0 && <span style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>{video.frame_count} scenes analyzed</span>}
            {(video.status === 'analyzed' || video.status === 'retranscribing') && <>
              <span className="badge analyzed"><span className="dot" />Analyzed</span>
              {video.analyzed_at && <span style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>{(() => {
                const d = new Date(video.analyzed_at!)
                const now = new Date()
                const isToday = d.toDateString() === now.toDateString()
                const yesterday = new Date(now); yesterday.setDate(now.getDate() - 1)
                const isYesterday = d.toDateString() === yesterday.toDateString()
                const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
                if (isToday) return `Today at ${time}`
                if (isYesterday) return `Yesterday at ${time}`
                return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ` at ${time}`
              })()}</span>}
            </>}
            {video.status === 'ready' && <span className="badge queued">Ready</span>}
            {video.status === 'processing' && video.frame_count === 0 && <span className="badge queued">Queued</span>}
            {video.status === 'processing' && video.frame_count > 0 && <span className="badge analyzing"><span className="dot" />Analyzing…</span>}
            {video.status === 'transcribing' && <span className="badge analyzing"><span className="dot" />Analyzing…</span>}
            {video.status === 'error' && <span className="badge error"><span className="dot" />Error</span>}
          </div>
        </div>
        <button
          className="ghost"
          onClick={(video.status === 'processing' || video.status === 'transcribing' || video.status === 'retranscribing') ? handleCancel : handleReprocess}
          disabled={cancelling}
          style={{ fontSize: '0.82rem', whiteSpace: 'nowrap', marginTop: '0.25rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', color: (video.status === 'processing' || video.status === 'transcribing' || video.status === 'retranscribing') ? 'var(--error)' : undefined, borderColor: (video.status === 'processing' || video.status === 'transcribing' || video.status === 'retranscribing') ? 'var(--error)' : undefined }}
          title={(video.status === 'processing' || video.status === 'transcribing' || video.status === 'retranscribing') ? 'Stop' : 'Re-run full analysis'}
        >
          {cancelling ? <><span className="spinner" style={{ width: 13, height: 13 }} />Stopping…</> : (video.status === 'processing' || video.status === 'transcribing') ? <><Square size={12} /> Stop</> : video.status === 'retranscribing' ? <><Square size={12} /> Stop Audio</> : (video.status === 'analyzed' || video.status === 'error') ? <><RotateCcw size={12} /> Re-analyze</> : <><Play size={12} /> Analyze</>}
        </button>
      </div>

      {/* Video player */}
      <div style={{ marginBottom: '0.5rem' }}>
        <video
          ref={videoRef}
          key={video.id}
          src={`/api/videos/${video.id}/stream`}
          controls
          onError={() => console.error(`Failed to load video stream for ${video.id}`)}
          style={{ width: '100%', maxHeight: 480, borderRadius: 'var(--radius)', background: '#000', display: 'block', outline: 'none' }}
        />
      </div>
      {/* Action row: thumbnail picker + download */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
        {video.status === 'analyzed' && (
          <>
            <button className="ghost" style={{ fontSize: '0.82rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }} onClick={() => setShowFramePicker(v => !v)}>
              {showFramePicker ? <><X size={12} /> Hide frames</> : video.thumbnail_path ? 'Change thumbnail' : 'Set thumbnail'}
            </button>
            {thumbnailSaved && <span style={{ fontSize: '0.78rem', color: 'var(--success)', display: 'inline-flex', alignItems: 'center', gap: 3 }}><Check size={12} /> Saved</span>}
          </>
        )}
        <a
          href={`/api/videos/${video.id}/download`}
          download
          className="ghost"
          style={{ fontSize: '0.82rem', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
        >
          <Download size={13} /> Download
        </a>
      </div>

      {/* Frame picker (thumbnail selection) */}
      {video.status === 'analyzed' && (
        <div style={{ marginBottom: '1.5rem' }}>
          {showFramePicker && frames.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '0.5rem', marginTop: '0.75rem' }}>
              {frames.map(f => (
                <div
                  key={f.id}
                  onClick={() => setThumbnail(f.path)}
                  style={{
                    cursor: savingThumbnail ? 'wait' : 'pointer', borderRadius: 8, overflow: 'hidden',
                    border: `2px solid ${video.thumbnail_path === f.path ? 'var(--accent)' : 'var(--border)'}`,
                    transition: 'border-color 0.15s', opacity: savingThumbnail ? 0.6 : 1
                  }}
                  title={`${fmt(f.timestamp)} — click to set as thumbnail`}
                >
                  <img src={`/api/frames/${video.id}/${encodeURIComponent(f.path.split('/').pop()!)}`} alt={fmt(f.timestamp)} style={{ width: '100%', height: 90, objectFit: 'cover', display: 'block' }} />
                  <div style={{ fontSize: '0.72rem', color: 'var(--muted)', padding: '0.2rem 0.4rem', background: 'var(--surface)' }}>{fmt(f.timestamp)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tags */}
      <div className="section">
        <div className="section-header"><h2>Tags</h2></div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
          {video.tags.map(t => (
            <span key={t} className="tag">{t}<button onClick={() => removeTag(t)}>×</button></span>
          ))}
          {video.tags.length === 0 && <span style={{ color: 'var(--muted)', fontSize: '0.82rem' }}>No tags — add one below</span>}
        </div>
        <input
          value={tagInput} onChange={e => setTagInput(e.target.value)} onKeyDown={addTag}
          placeholder="Add tag and press Enter" style={{ maxWidth: 280 }}
        />
      </div>

      <hr className="divider" />

      {/* Summary */}
      <div className="section">
        <div className="section-header">
          <h2>AI Summary</h2>
          {video.status === 'analyzed' && !summarizing && (
            <button className="ghost" onClick={handleSummarize} style={{ fontSize: '0.82rem' }}>
              {video.summary ? 'Regenerate' : 'Generate Summary'}
            </button>
          )}
        </div>
        {(video.status === 'processing' || video.status === 'transcribing') && (
          <p style={{ color: 'var(--muted)', fontSize: '0.875rem' }}>
            <span className="spinner" style={{ marginRight: '0.5rem' }} />
            {video.status === 'transcribing' ? 'Transcribing audio…' : `Video is still being analyzed (${video.frame_count} scenes so far)…`}
          </p>
        )}
        {video.status === 'analyzed' && !video.summary && !summarizing && (
          <p style={{ color: 'var(--muted)', fontSize: '0.875rem' }}>No summary yet. Click "Generate Summary" to create one.</p>
        )}
        {summarizing && (
          <p style={{ color: 'var(--muted)', fontSize: '0.875rem' }}>
            <span className="spinner" style={{ marginRight: '0.5rem' }} />
            Generating summary…
          </p>
        )}
        {video.summary && !summarizing && <div className="summary-box">{video.summary}</div>}
      </div>

      <hr className="divider" />

      {/* Segments — unified visual + speech editor */}
      {(video.status === 'analyzed' || video.status === 'retranscribing') && (frames.length > 0 || (transcript && transcript.length > 0) || fused.length > 0) && (
        <div className="section">
          <div className="section-header">
            <h2>Segments
              <span style={{ marginLeft: '0.6rem', fontSize: '0.75rem', fontWeight: 400, color: 'var(--muted)' }}>
                {allSegs.filter(s => segFilter === 'all' || s.type === segFilter).length} shown
              </span>
            </h2>
            <div style={{ display: 'flex', gap: '0.3rem', alignItems: 'center' }}>
              {(video.status === 'analyzed' || video.status === 'retranscribing' || retranscribing) && (
                <button
                  onClick={handleRetranscribe}
                  disabled={retranscribing}
                  style={{
                    fontSize: '0.76rem', padding: '0.2rem 0.5rem', cursor: retranscribing ? 'default' : 'pointer',
                    background: 'transparent', color: 'var(--muted)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                    display: 'flex', alignItems: 'center', gap: '0.3rem',
                  }}
                  title="Re-transcribe audio only (keeps visual analysis)"
                >
                  {retranscribing ? (() => {
                    const elapsed = retranscribeElapsed
                    const estTotal = video.duration ? Math.ceil(video.duration / 4) : null
                    const elapsedStr = elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed/60)}m ${elapsed%60}s`
                    const etaStr = estTotal && elapsed < estTotal
                      ? ` · ~${Math.ceil((estTotal - elapsed) / 60)}m left`
                      : ''
                    return <><span className="spinner" style={{ width: 10, height: 10 }} /> {elapsedStr}{etaStr}</>
                  })() : video.status === 'retranscribing' ? <><RotateCcw size={12} /> Restart Audio</> : <><RotateCcw size={12} /> Audio</>}
                </button>
              )}
              {(['all', 'visual', 'speech', 'fused'] as const)
                .filter(f => f !== 'fused' || fused.length > 0)
                .map(f => (
                <button
                  key={f}
                  onClick={() => setSegFilter(f)}
                  style={{
                    fontSize: '0.76rem', padding: '0.2rem 0.5rem', cursor: 'pointer',
                    display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
                    background: segFilter === f ? 'var(--accent)' : 'transparent',
                    color: segFilter === f ? '#fff' : 'var(--muted)',
                    border: `1px solid ${segFilter === f ? 'var(--accent)' : 'var(--border)'}`,
                    borderRadius: 'var(--radius)',
                  }}
                >
                  {f === 'all' ? 'All' : f === 'visual' ? <><Film size={11} /> Visual</> : f === 'speech' ? <><Mic size={11} /> Speech</> : <><Sparkles size={11} /> Fused</>}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', maxHeight: 500, overflowY: 'auto', paddingRight: '0.25rem' }}>
            {allSegs
              .filter(s => segFilter === 'all' || s.type === segFilter)
              .map(seg => (
                <div
                  key={seg.id}
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
                    padding: '0.4rem 0.5rem', borderRadius: 6,
                    background: editingSegId === seg.id ? 'var(--surface)' : 'transparent',
                    border: `1px solid ${editingSegId === seg.id ? 'var(--border)' : 'transparent'}`,
                  }}
                >
                  {(seg.type === 'visual' || seg.type === 'fused') && seg.frame_path ? (
                    <div style={{ position: 'relative', flexShrink: 0, marginTop: 2 }}>
                      <img
                        src={`/api/frames/${video.id}/${encodeURIComponent(seg.frame_path.split('/').pop()!)}`}
                        alt=""
                        style={{ width: 60, height: 40, objectFit: 'cover', borderRadius: 4, display: 'block' }}
                      />
                      {seg.type === 'fused' && (
                        <span style={{ position: 'absolute', top: 2, left: 2, background: 'rgba(0,0,0,0.6)', borderRadius: 3, padding: '1px 2px', lineHeight: 0, display: 'flex' }}>
                          <Sparkles size={10} style={{ color: '#fff' }} />
                        </span>
                      )}
                    </div>
                  ) : (
                    <div style={{
                      width: 60, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: 'var(--surface)', borderRadius: 4, flexShrink: 0, marginTop: 2,
                    }}>{seg.type === 'fused' ? <Sparkles size={18} style={{ color: 'var(--muted)' }} /> : <Mic size={18} style={{ color: 'var(--muted)' }} />}</div>
                  )}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <span style={{ fontSize: '0.72rem', color: 'var(--accent)', fontVariantNumeric: 'tabular-nums', display: 'block', marginBottom: '0.2rem' }}>
                      {fmt(seg.timestamp)}
                    </span>
                    {editingSegId === seg.id ? (
                      <div>
                        <textarea
                          value={editingText}
                          onChange={e => setEditingText(e.target.value)}
                          autoFocus
                          rows={3}
                          style={{ width: '100%', resize: 'vertical', fontSize: '0.82rem', fontFamily: 'inherit', lineHeight: 1.5, boxSizing: 'border-box' }}
                        />
                        <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.3rem' }}>
                          <button
                            style={{ fontSize: '0.78rem', padding: '0.2rem 0.6rem' }}
                            disabled={savingSegId === seg.id}
                            onClick={() => handleSegSave(seg.id, seg.type)}
                          >
                            {savingSegId === seg.id ? 'Saving…' : 'Save'}
                          </button>
                          <button className="ghost" style={{ fontSize: '0.78rem' }} onClick={() => setEditingSegId(null)}>
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.4rem' }}>
                        <p
                          style={{ fontSize: '0.82rem', lineHeight: 1.5, margin: 0, flex: 1, color: seg.caption ? 'var(--text)' : 'var(--muted)', cursor: seg.type === 'fused' ? 'default' : 'text', fontStyle: seg.caption ? 'normal' : 'italic' }}
                          onClick={seg.type === 'fused' ? undefined : () => { setEditingSegId(seg.id); setEditingText(seg.caption) }}
                          title={seg.type === 'fused' ? 'Fused description (regenerated on re-analysis)' : 'Click to edit'}
                        >
                          {seg.caption || 'No caption'}
                        </p>
                        <a
                          href={`/api/videos/${video.id}/clip?start=${Math.max(0, seg.timestamp - 2).toFixed(1)}&end=${(seg.timestamp + 30).toFixed(1)}`}
                          download
                          onClick={e => e.stopPropagation()}
                          style={{ color: 'var(--muted)', textDecoration: 'none', flexShrink: 0, opacity: 0.45, marginTop: '0.15rem', lineHeight: 0 }}
                          title="Download ~30s clip"
                        >
                          <Download size={13} />
                        </a>
                      </div>
                    )}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      <hr className="divider" />

      {/* Segment search */}
      <div className="section">
        <div className="section-header"><h2>Search Within Video</h2></div>
        <form onSubmit={handleSearch} className="search-bar">
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="e.g. 'when does the AMD Strix Halo appear'" />
          <button type="submit" disabled={searching || video.status === 'processing' || video.status === 'transcribing'}>
            {searching ? 'Searching…' : 'Search'}
          </button>
        </form>
        {(video.status === 'processing' || video.status === 'transcribing') && (
          <p style={{ color: 'var(--muted)', fontSize: '0.82rem' }}>Search available once analysis is complete.</p>
        )}
        {segments.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginTop: '1rem' }}>
            {segments.map((seg, i) => {
              const isTranscript = seg.caption.startsWith('[transcript]')
              return (
                <div key={i} className="segment-card">
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <span className="segment-time">{fmt(seg.start_time)} – {fmt(seg.end_time)}</span>
                    <span className="segment-type">{isTranscript ? '🎤 speech' : '🖼 visual'}</span>
                  </div>
                  <p className="segment-caption">{isTranscript ? seg.caption.replace('[transcript] ', '') : seg.caption}</p>
                  <span className="segment-score">Score: {seg.score.toFixed(3)}</span>
                </div>
              )
            })}
          </div>
        )}
        {segments.length === 0 && query && !searching && (
          <p style={{ color: 'var(--muted)', fontSize: '0.875rem', marginTop: '1rem' }}>No matching moments found.</p>
        )}
      </div>

      {showAnalyzeModal && video && (
        <AnalyzeModal
          videoIds={[video.id]}
          videos={[video]}
          onClose={() => setShowAnalyzeModal(false)}
          onDone={() => { setShowAnalyzeModal(false); loadVideo() }}
        />
      )}
    </div>
  )
}
