import { useState, useEffect, useCallback } from 'react'
import { api, AppSettings, QdrantHealth, Video } from '../services/api'

type ModelStatus = { name: string; url: string; ok: boolean; models: string[] }

const URL_GROUPS: Array<{
  urlKey: keyof AppSettings
  modelKey: keyof AppSettings
  label: string
  urlDesc: string
  modelDesc: string
}> = [
  {
    urlKey: 'vllm_embed_url', modelKey: 'vllm_embed_model',
    label: 'Embeddings',
    urlDesc: 'vLLM endpoint serving the embedding model',
    modelDesc: 'Model ID used for embedding captions and queries',
  },
  {
    urlKey: 'vllm_vlm_url', modelKey: 'vllm_vlm_model',
    label: 'Vision / VLM',
    urlDesc: 'vLLM endpoint serving the visual captioning model',
    modelDesc: 'Model ID used for captioning video frames',
  },
  {
    urlKey: 'vllm_url', modelKey: 'vllm_model',
    label: 'LLM / Summarizer',
    urlDesc: 'vLLM endpoint serving the text LLM for summarization',
    modelDesc: 'Model ID used for generating video summaries',
  },
]

const WHISPER_SIZES = ['tiny', 'base', 'small', 'medium', 'large']

const BACKEND_LABELS: Record<string, string> = {
  custom: 'Custom',
  ollama: 'Ollama (Docker)',
  'ollama-native': 'Ollama (host)',
  vllm: 'vLLM (Docker)',
}

function BackendBanner({ form }: { form: AppSettings }) {
  const backend = form.ai_backend ?? 'custom'
  const isLocal = backend !== 'custom'
  const isWarning = !!form.backend_warning
  const isNative = backend === 'ollama-native'

  const color = isWarning ? 'var(--error)' : isNative ? 'var(--success)' : isLocal ? 'var(--accent)' : 'var(--muted)'
  const bg = isWarning ? 'rgba(239,68,68,0.08)' : isNative ? 'rgba(34,197,94,0.08)' : isLocal ? 'rgba(99,102,241,0.08)' : 'var(--surface)'

  return (
    <div style={{ border: `1px solid ${color}`, borderRadius: 'var(--radius)', padding: '0.85rem 1rem', marginBottom: '1.75rem', background: bg }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: isLocal ? '0.4rem' : 0 }}>
        <span style={{ color, fontWeight: 700, fontSize: '0.85rem' }}>
          {isWarning ? '⚠ ' : isNative ? '✓ ' : ''}{BACKEND_LABELS[backend] ?? backend}
        </span>
        {isLocal && form._resolved_family && (
          <span style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
            {form._resolved_family} · {form._resolved_tier}
          </span>
        )}
      </div>
      {isWarning && (
        <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--error)', lineHeight: 1.5 }}>{form.backend_warning}</p>
      )}
      {isLocal && !isWarning && form._resolved_model && (
        <div style={{ fontSize: '0.78rem', color: 'var(--muted)', lineHeight: 1.6 }}>
          <span style={{ fontFamily: 'monospace', color: 'var(--text)' }}>{form._resolved_model}</span>
          {' '}— vision + summarization
          <br />
          <span style={{ fontFamily: 'monospace', color: 'var(--text)' }}>{form._resolved_embed}</span>
          {' '}— embeddings
        </div>
      )}
      {isNative && !isWarning && (
        <p style={{ margin: '0.3rem 0 0', fontSize: '0.78rem', color: 'var(--muted)' }}>
          Ollama is running on your Mac host. Metal/MLX acceleration active.
        </p>
      )}
      {backend === 'custom' && (
        <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--muted)' }}>
          Configure model endpoints below. Change <code>AI_BACKEND</code> in <code>.env</code> to switch to a local mode.
        </p>
      )}
    </div>
  )
}

export default function SettingsPage() {
  const [form, setForm] = useState<AppSettings | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [statuses, setStatuses] = useState<ModelStatus[] | null>(null)
  const [probing, setProbing] = useState(false)
  const [qdrantHealth, setQdrantHealth] = useState<QdrantHealth | null>(null)
  const [checkingQdrant, setCheckingQdrant] = useState(false)
  const [purgingQdrant, setPurgingQdrant] = useState(false)
  const [blacklistInput, setBlacklistInput] = useState('')

  useEffect(() => {
    api.getSettings().then(setForm).catch(() => {})
    api.modelStatus().then(d => setStatuses(d.services)).catch(() => setStatuses([]))
  }, [])

  const set = (key: keyof AppSettings, value: unknown) =>
    setForm(f => f ? { ...f, [key]: value } : f)

  const handleSave = async () => {
    if (!form) return
    setSaving(true); setSaved(false); setError(null)
    try {
      setForm(await api.updateSettings(form))
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  const handleProbe = async () => {
    setProbing(true); setStatuses(null)
    try { setStatuses((await api.modelStatus()).services) }
    finally { setProbing(false) }
  }

  if (!form) return (
    <div className="page" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent: 'center', marginTop: '4rem', color: 'var(--muted)' }}>
      <span className="spinner" /> Loading…
    </div>
  )

  const probeByUrl: Record<string, string[]> = {}
  if (statuses) statuses.forEach(s => { probeByUrl[s.url] = s.models })
  const statusByUrl: Record<string, ModelStatus> = {}
  if (statuses) statuses.forEach(s => { statusByUrl[s.url] = s })

  return (
    <div className="page">
      <div style={{ maxWidth: 640, margin: '0 auto' }}>
        <h1 style={{ marginBottom: '0.25rem' }}>Settings</h1>
        <p style={{ color: 'var(--muted)', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
          Configure AI model endpoints and names. Changes take effect immediately without restart.
        </p>

        <BackendBanner form={form} />

        {/* Model connectivity — shown for all backends */}
        {statuses !== null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', marginBottom: '1.5rem', padding: '0.65rem 0.9rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
            {statuses.length === 0
              ? <span style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>No model services configured</span>
              : statuses.map(svc => (
                <div key={svc.name} title={svc.ok ? svc.models.join(', ') : `Unreachable: ${svc.url}`}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.78rem', color: svc.ok ? 'var(--text)' : 'var(--error)', cursor: 'default' }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: svc.ok ? 'var(--success)' : 'var(--error)', display: 'inline-block', flexShrink: 0 }} />
                  <span>{svc.name}</span>
                  {svc.ok && svc.models.length > 0 && (
                    <span style={{ fontFamily: 'monospace', fontSize: '0.72rem', color: 'var(--muted)' }}>({svc.models[0]})</span>
                  )}
                  {!svc.ok && <span style={{ fontSize: '0.72rem', color: 'var(--error)' }}>offline</span>}
                </div>
              ))
            }
            <span style={{ flex: 1 }} />
            <button className="ghost" onClick={handleProbe} disabled={probing} style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}>
              {probing ? <><span className="spinner" style={{ width: 11, height: 11, marginRight: '0.3rem' }} />…</> : '↺ Refresh'}
            </button>
          </div>
        )}

        {/* AI Models — only shown in custom mode (local modes are auto-configured) */}
        {form.ai_backend === 'custom' && (
          <div className="section" style={{ marginTop: 0 }}>
            <div className="section-header">
              <h2>AI Models</h2>
              <button className="ghost" onClick={handleProbe} disabled={probing} style={{ fontSize: '0.8rem' }}>
                {probing ? <><span className="spinner" style={{ width: 12, height: 12, marginRight: '0.4rem' }} />Probing…</> : '⚡ Test connections'}
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {URL_GROUPS.map(({ urlKey, modelKey, label, urlDesc, modelDesc }) => {
                const st = statusByUrl[form[urlKey] as string]
                const probedModels = probeByUrl[form[urlKey] as string] ?? []
                return (
                  <div key={urlKey} style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>{label}</span>
                      {st && (
                        <span className={`badge ${st.ok ? 'analyzed' : 'error'}`}>
                          <span className="dot" />{st.ok ? 'Connected' : 'Unreachable'}
                        </span>
                      )}
                    </div>

                    <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--muted)', marginBottom: '0.25rem' }}>Endpoint URL</label>
                    <input
                      type="url"
                      value={form[urlKey] as string}
                      onChange={e => set(urlKey, e.target.value)}
                      style={{ width: '100%', fontFamily: 'monospace', fontSize: '0.85rem', marginBottom: '0.25rem' }}
                      placeholder="http://host:port"
                    />
                    <p style={{ margin: '0 0 0.75rem', fontSize: '0.72rem', color: 'var(--muted)' }}>{urlDesc}</p>

                    <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--muted)', marginBottom: '0.25rem' }}>Model name</label>
                    <input
                      type="text"
                      value={form[modelKey] as string}
                      onChange={e => set(modelKey, e.target.value)}
                      style={{ width: '100%', fontFamily: 'monospace', fontSize: '0.85rem' }}
                      placeholder="org/model-name"
                    />
                    <p style={{ margin: '0.25rem 0 0', fontSize: '0.72rem', color: 'var(--muted)' }}>{modelDesc}</p>

                    {probedModels.length > 0 && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.6rem', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>Available:</span>
                        {probedModels.map(m => (
                          <button key={m} className="ghost" onClick={() => set(modelKey, m)}
                            style={{
                              fontSize: '0.72rem', padding: '0.15rem 0.5rem', fontFamily: 'monospace',
                              borderColor: form[modelKey] === m ? 'var(--accent)' : undefined,
                              color: form[modelKey] === m ? 'var(--accent)' : undefined,
                            }}>
                            {m}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}

              {/* Qdrant */}
              <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem' }}>
                <div style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '0.75rem' }}>Qdrant</div>
                <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--muted)', marginBottom: '0.25rem' }}>Endpoint URL</label>
                <input type="url" value={form.qdrant_url} onChange={e => set('qdrant_url', e.target.value)}
                  style={{ width: '100%', fontFamily: 'monospace', fontSize: '0.85rem' }}
                  placeholder="http://host:6333" />
              </div>
            </div>
          </div>
        )}

        {/* Video Libraries */}
        <div className="section">
          <div className="section-header"><h2>Video Libraries</h2></div>
          <p style={{ margin: '0 0 1rem', fontSize: '0.82rem', color: 'var(--muted)' }}>
            Set <code>VIDEO_DIR_1</code> through <code>VIDEO_DIR_5</code> in <code>.env</code> and restart to add library directories. Each path is mounted read-only into the container.
          </p>
          {form.video_dirs && form.video_dirs.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              {form.video_dirs.map((dir, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '0.5rem 0.75rem' }}>
                  <span style={{ color: 'var(--success)', fontSize: '0.75rem' }}>●</span>
                  <code style={{ fontSize: '0.82rem', color: 'var(--text)' }}>{dir}</code>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ fontSize: '0.82rem', color: 'var(--muted)', fontStyle: 'italic' }}>No library directories configured.</p>
          )}
        </div>

        {/* Processing */}
        <div className="section">
          <div className="section-header"><h2>Processing</h2></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.4rem' }}>Whisper model</label>
              <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                {WHISPER_SIZES.map(s => (
                  <button key={s} className={form.whisper_model === s ? '' : 'ghost'}
                    onClick={() => set('whisper_model', s)}
                    style={{ fontSize: '0.82rem', padding: '0.3rem 0.75rem' }}>{s}</button>
                ))}
              </div>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.75rem', color: 'var(--muted)' }}>
                Larger models are more accurate but slower. Requires re-analysis to affect existing videos.
              </p>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                Frame interval — <span style={{ color: 'var(--accent)', fontWeight: 400 }}>every {form.frame_extraction_interval}s</span>
              </label>
              <input type="range" min={1} max={60} step={1} value={form.frame_extraction_interval}
                onChange={e => set('frame_extraction_interval', Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--muted)', marginTop: '0.2rem' }}>
                <span>1s (dense)</span><span>60s (sparse)</span>
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                Embed batch size — <span style={{ color: 'var(--accent)', fontWeight: 400 }}>{form.embed_batch_size}</span>
              </label>
              <input type="range" min={1} max={64} step={1} value={form.embed_batch_size}
                onChange={e => set('embed_batch_size', Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--muted)', marginTop: '0.2rem' }}>
                <span>1</span><span>64</span>
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                Max concurrent jobs — <span style={{ color: 'var(--accent)', fontWeight: 400 }}>{form.max_concurrent_jobs}</span>
              </label>
              <input type="range" min={1} max={32} step={1} value={form.max_concurrent_jobs}
                onChange={e => set('max_concurrent_jobs', Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--muted)', marginTop: '0.2rem' }}>
                <span>1 (sequential)</span><span>32</span>
              </div>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.75rem', color: 'var(--muted)' }}>
                How many videos can be analyzed at the same time. Lower values reduce GPU/CPU pressure.
              </p>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.4rem' }}>Import blacklist</label>
              {(form.import_blacklist ?? []).length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginBottom: '0.5rem' }}>
                  {(form.import_blacklist ?? []).map((pat, i) => (
                    <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 4, padding: '0.15rem 0.5rem', fontSize: '0.78rem', fontFamily: 'monospace' }}>
                      {pat}
                      <button onClick={() => set('import_blacklist', (form.import_blacklist ?? []).filter((_, j) => j !== i))}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', fontSize: '0.9rem', padding: 0, lineHeight: 1 }}>×</button>
                    </span>
                  ))}
                </div>
              )}
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <input
                  type="text"
                  value={blacklistInput}
                  onChange={e => setBlacklistInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && blacklistInput.trim()) {
                      set('import_blacklist', [...(form.import_blacklist ?? []), blacklistInput.trim()])
                      setBlacklistInput('')
                    }
                  }}
                  placeholder="e.g. */broll/* or */_archive/*"
                  style={{ flex: 1, fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
                <button className="ghost" onClick={() => {
                  if (blacklistInput.trim()) {
                    set('import_blacklist', [...(form.import_blacklist ?? []), blacklistInput.trim()])
                    setBlacklistInput('')
                  }
                }} style={{ flexShrink: 0 }}>Add</button>
              </div>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.75rem', color: 'var(--muted)' }}>
                Glob patterns matched against the full file path. Matching videos are skipped on import and reimport. Existing videos in the library are not affected.
              </p>
            </div>
          </div>
        </div>

        {/* Vector Store */}
        <div className="section">
          <div className="section-header">
            <h2>Vector Store</h2>
            <button className="ghost" onClick={async () => {
              setCheckingQdrant(true)
              try { setQdrantHealth(await api.qdrantHealth()) }
              finally { setCheckingQdrant(false) }
            }} disabled={checkingQdrant} style={{ fontSize: '0.8rem' }}>
              {checkingQdrant ? <><span className="spinner" style={{ width: 12, height: 12, marginRight: '0.4rem' }} />Checking…</> : '⚡ Check health'}
            </button>
          </div>

          {qdrantHealth && (() => {
            const h = qdrantHealth
            const allOk = h.reachable && h.collection_exists && h.issues.length === 0
            const perVideoIssues = h.issues.filter(i => i.type === 'missing' || i.type === 'incomplete')
            const orphanIssue = h.issues.find(i => i.type === 'orphaned')
            return (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {/* Summary row */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', padding: '0.65rem 0.9rem', background: 'var(--surface)', border: `1px solid ${allOk ? 'var(--border)' : 'var(--error)'}`, borderRadius: 'var(--radius)', fontSize: '0.82rem' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: h.reachable ? 'var(--success)' : 'var(--error)', display: 'inline-block' }} />
                    {h.reachable ? 'Qdrant reachable' : 'Qdrant unreachable'}
                  </span>
                  {h.reachable && (
                    <>
                      <span style={{ color: 'var(--muted)' }}>{h.total_points.toLocaleString()} points</span>
                      <span style={{ color: 'var(--success)' }}>{h.summary.ok} ok</span>
                      {h.summary.missing > 0 && <span style={{ color: 'var(--error)' }}>{h.summary.missing} missing</span>}
                      {h.summary.incomplete > 0 && <span style={{ color: 'var(--warning, #f59e0b)' }}>{h.summary.incomplete} incomplete</span>}
                      {h.summary.orphaned_points > 0 && <span style={{ color: 'var(--muted)' }}>{h.summary.orphaned_points} orphaned pts</span>}
                      {allOk && <span style={{ color: 'var(--success)' }}>✓ All clean</span>}
                    </>
                  )}
                  {h.error && <span style={{ color: 'var(--error)', fontSize: '0.75rem' }}>{h.error}</span>}
                </div>

                {/* Per-video issues */}
                {perVideoIssues.length > 0 && (
                  <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
                    <div style={{ padding: '0.5rem 0.75rem', borderBottom: '1px solid var(--border)', fontSize: '0.78rem', fontWeight: 600, color: 'var(--muted)' }}>
                      Videos with embedding issues — reprocess to fix
                    </div>
                    {perVideoIssues.map(issue => (
                      <div key={issue.video_id} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.5rem 0.75rem', borderBottom: '1px solid var(--border)', fontSize: '0.82rem' }}>
                        <span className={`badge ${issue.type === 'missing' ? 'error' : 'processing'}`}>
                          <span className="dot" />{issue.type}
                        </span>
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{issue.name}</span>
                        <span style={{ fontSize: '0.72rem', color: 'var(--muted)', flexShrink: 0 }}>
                          {issue.qdrant_points}/{issue.db_frames} pts
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Orphaned points */}
                {orphanIssue && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.65rem 0.9rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', fontSize: '0.82rem' }}>
                    <span style={{ flex: 1, color: 'var(--muted)' }}>
                      {orphanIssue.qdrant_points} orphaned embedding{orphanIssue.qdrant_points !== 1 ? 's' : ''} from {orphanIssue.video_ids?.length} deleted video{(orphanIssue.video_ids?.length ?? 0) !== 1 ? 's' : ''}
                    </span>
                    <button className="ghost" disabled={purgingQdrant} onClick={async () => {
                      setPurgingQdrant(true)
                      try {
                        await api.purgeOrphanedEmbeddings()
                        setQdrantHealth(await api.qdrantHealth())
                      } finally {
                        setPurgingQdrant(false)
                      }
                    }} style={{ fontSize: '0.78rem', padding: '0.25rem 0.6rem' }}>
                      {purgingQdrant ? <><span className="spinner" style={{ width: 11, height: 11, marginRight: '0.3rem' }} />Purging…</> : 'Purge orphans'}
                    </button>
                  </div>
                )}
              </div>
            )
          })()}

          {!qdrantHealth && !checkingQdrant && (
            <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--muted)' }}>
              Run a health check to verify embedding counts match analyzed videos and detect orphaned data from deleted videos.
            </p>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '2rem' }}>
          <button onClick={handleSave} disabled={saving} style={{ minWidth: 100 }}>
            {saving ? <><span className="spinner" style={{ width: 13, height: 13, marginRight: '0.4rem' }} />Saving…</> : 'Save'}
          </button>
          {saved && <span style={{ color: 'var(--success)', fontSize: '0.85rem' }}>✓ Saved</span>}
          {error && <span style={{ color: 'var(--error)', fontSize: '0.85rem' }}>{error}</span>}
        </div>

        <HiddenVideos />
      </div>
    </div>
  )
}

function HiddenVideos() {
  const [videos, setVideos] = useState<Video[] | null>(null)
  const [unhiding, setUnhiding] = useState<string | null>(null)

  const load = useCallback(() => {
    api.listHiddenVideos().then(setVideos).catch(() => setVideos([]))
  }, [])

  useEffect(() => { load() }, [load])

  const handleUnhide = async (id: string) => {
    setUnhiding(id)
    try {
      await api.unhideVideo(id)
      setVideos(prev => prev ? prev.filter(v => v.id !== id) : prev)
    } catch { /* ignore */ } finally { setUnhiding(null) }
  }

  if (!videos || videos.length === 0) return null

  return (
    <div className="section" style={{ marginTop: '2rem' }}>
      <div className="section-header"><h2>Hidden Videos</h2></div>
      <p style={{ margin: '0 0 1rem', fontSize: '0.82rem', color: 'var(--muted)' }}>
        Videos you've hidden from the library. Unhide to restore them.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        {videos.map(v => (
          <div key={v.id} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '0.5rem 0.75rem' }}>
            <span style={{ flex: 1, fontSize: '0.85rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v.display_name || v.name}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--muted)', flexShrink: 0, fontFamily: 'monospace' }}>{v.path.split('/').slice(-2).join('/')}</span>
            <button className="ghost" onClick={() => handleUnhide(v.id)} disabled={unhiding === v.id}
              style={{ fontSize: '0.78rem', padding: '0.2rem 0.6rem', flexShrink: 0 }}>
              {unhiding === v.id ? '…' : 'Unhide'}
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

