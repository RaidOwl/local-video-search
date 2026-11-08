import { useState, useEffect, useCallback, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import { api, apiErrorMessage, Video, AppSettings, ModelStatus } from '../services/api'
import { fmt } from '../utils'
import { Video as VideoIcon, Square, ArrowRight, X, Check, Play, LayoutGrid, List, Folder, VolumeX, VideoOff, ChevronDown, ChevronRight, Mic, RotateCcw, Sparkles } from 'lucide-react'

type ViewMode = 'card' | 'tree' | 'folder'

// A full analysis run (created via /analyze) passes through these phases in
// order; 'processing' / 'transcribing' are legacy values kept for videos that
// were mid-run when this pipeline was split into finer-grained statuses.
const ANALYSIS_PHASE_STATUSES = new Set([
  'queued', 'scene_detection', 'video_processing', 'audio_processing', 'fusing',
  'processing', 'transcribing',
])
function isAnalysisActive(status: string): boolean {
  return ANALYSIS_PHASE_STATUSES.has(status)
}
function isAudioPhase(status: string): boolean {
  return status === 'audio_processing' || status === 'transcribing' || status === 'retranscribing'
}

interface FolderNode {
  name: string
  fullPath: string
  subfolders: Record<string, FolderNode>
  videos: Video[]
}

// Strip at the lib mount root (/data/videos/libN) rather than the deepest
// common ancestor, so intermediate folders like '2022_Before' stay visible.
const VIDEO_MOUNT_ROOT = '/data/videos'
const LIB_SEGMENT = /^lib\d+$/

function buildTree(videos: Video[]): { root: FolderNode; prefix: string } {
  const root: FolderNode = { name: 'videos', fullPath: VIDEO_MOUNT_ROOT, subfolders: {}, videos: [] }
  for (const v of videos) {
    const rel = v.path.startsWith(VIDEO_MOUNT_ROOT + '/') ? v.path.slice(VIDEO_MOUNT_ROOT.length + 1) : v.path
    const parts = rel.split('/')
    let node = root
    for (let i = 0; i < parts.length - 1; i++) {
      const seg = parts[i]
      if (!node.subfolders[seg]) {
        node.subfolders[seg] = { name: seg, fullPath: node.fullPath + '/' + seg, subfolders: {}, videos: [] }
      }
      node = node.subfolders[seg]
    }
    node.videos.push(v)
  }
  // Skip past the libN mount nodes — if only one lib is in use, drop into it
  // so users see their folder structure directly rather than a lib1/ wrapper.
  const libChildren = Object.values(root.subfolders)
  if (libChildren.length === 1 && LIB_SEGMENT.test(libChildren[0].name) && root.videos.length === 0) {
    return { root: libChildren[0], prefix: libChildren[0].fullPath }
  }
  // Multiple libs: show libN nodes at top level
  const allLib = libChildren.every(n => LIB_SEGMENT.test(n.name))
  if (allLib && root.videos.length === 0) {
    return { root, prefix: VIDEO_MOUNT_ROOT }
  }
  return { root, prefix: VIDEO_MOUNT_ROOT }
}

function getNodeByPath(root: FolderNode, fullPath: string): FolderNode | null {
  if (root.fullPath === fullPath) return root
  for (const child of Object.values(root.subfolders)) {
    const found = getNodeByPath(child, fullPath)
    if (found) return found
  }
  return null
}

function ModelStatusBar() {
  const [services, setServices] = useState<ModelStatus[] | null>(null)

  useEffect(() => {
    api.modelStatus().then(d => setServices(d.services)).catch(() => setServices([]))
    const id = setInterval(() => {
      api.modelStatus().then(d => setServices(d.services)).catch(() => {})
    }, 30000)
    return () => clearInterval(id)
  }, [])

  if (!services) return null

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
      {services.map(svc => (
        <div key={svc.name} title={svc.ok ? svc.models.join(', ') : svc.reason ?? `Unreachable: ${svc.url}`}
          style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: svc.ok ? 'var(--muted)' : 'var(--error)', cursor: 'default' }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: svc.ok ? 'var(--success)' : 'var(--error)', display: 'inline-block', flexShrink: 0 }} />
          {svc.name}
        </div>
      ))}
    </div>
  )
}

interface ProgressSnapshot { frameCount: number; ts: number; tProgress: number; tTs: number }

function etaLabel(snap: ProgressSnapshot | undefined, current: number, totalFrames: number): string | null {
  if (!snap || snap.ts === 0 || current === 0 || totalFrames <= 0) return null
  const elapsed = (Date.now() - snap.ts) / 1000
  const delta = current - snap.frameCount
  if (delta <= 0 || elapsed < 1) return null
  const rate = delta / elapsed // frames per second (cumulative average)
  const remaining = totalFrames - current
  if (remaining <= 0) return null
  const secs = Math.round(remaining / rate)
  if (secs < 10) return null // avoid meaningless "~0s" or "~3s"
  if (secs < 60) return `~${secs}s`
  return `~${Math.round(secs / 60)}m`
}

// ETA for audio transcription, derived from the fraction-complete (0..1) rate.
function transcribeEtaLabel(snap: ProgressSnapshot | undefined, progress: number): string | null {
  if (!snap || snap.tTs === 0 || progress <= 0 || progress >= 1) return null
  const elapsed = (Date.now() - snap.tTs) / 1000
  const delta = progress - snap.tProgress
  if (delta <= 0 || elapsed < 1) return null
  const rate = delta / elapsed // fraction per second
  const remaining = 1 - progress
  const secs = Math.round(remaining / rate)
  if (secs < 5) return null
  if (secs < 60) return `~${secs}s`
  return `~${Math.round(secs / 60)}m`
}

function StatusBadge({ status, frameCount, eta, transcribeProgress = 0, fuseProgress = 0 }: { status: Video['status']; frameCount: number; eta: string | null; transcribeProgress?: number; fuseProgress?: number }) {
  if (status === 'ready') return <span className="badge queued">Ready</span>
  if (status === 'analyzed') return <span className="badge analyzed"><span className="dot" />Analyzed</span>
  if (status === 'error') return <span className="badge error"><span className="dot" />Error</span>
  if (status === 'fusing') {
    const pct = Math.round((fuseProgress || 0) * 100)
    return <span className="badge fusing"><Sparkles size={11} />{pct > 0 ? `Fusing ${pct}%` : 'Fusing…'}</span>
  }
  if (status === 'transcribing' || status === 'audio_processing' || status === 'retranscribing') {
    const base = status === 'retranscribing' ? 'Audio' : 'Audio Processing'
    const pct = Math.round((transcribeProgress || 0) * 100)
    const label = pct > 0 ? `${base} ${pct}%` : `${base}…`
    return <span className="badge transcribing"><Mic size={11} />{label}{eta ? ` · ${eta}` : ''}</span>
  }
  if (status === 'scene_detection') {
    return <span className="badge scene-detection"><span className="dot" />Detecting Scenes</span>
  }
  if (status === 'queued') return <span className="badge queued">Queued</span>
  if (frameCount === 0) return <span className="badge queued">Queued</span>
  return (
    <span className="badge analyzing">
      <span className="dot" />
      {frameCount} frames{eta ? ` · ${eta}` : ''}
    </span>
  )
}

// ── Tree view ─────────────────────────────────────────────────────────────────

interface ViewProps {
  videos: Video[]
  selected: Set<string>
  removing: string | null
  snapshots: React.MutableRefObject<Record<string, ProgressSnapshot>>
  toggleSelect: (id: string) => void
  handleRemove: (e: React.MouseEvent, id: string) => void
  handleHideAll: (e: React.MouseEvent, videoIds: string[]) => void
  navigate: (path: string) => void
  treeToggle: { open: boolean; seq: number } | null
  frameInterval: number
}

function getAllVideoIds(node: FolderNode): string[] {
  return [
    ...node.videos.map(v => v.id),
    ...Object.values(node.subfolders).flatMap(getAllVideoIds),
  ]
}

function TreeNode({ node, depth, selected, removing, snapshots, toggleSelect, handleRemove, handleHideAll, navigate, treeToggle, frameInterval }: {
  node: FolderNode; depth: number
} & Omit<ViewProps, 'videos'>) {
  const [open, setOpen] = useState(true)
  const lastSeq = useRef(-1)

  useEffect(() => {
    if (treeToggle && treeToggle.seq !== lastSeq.current) {
      lastSeq.current = treeToggle.seq
      setOpen(treeToggle.open)
    }
  }, [treeToggle])
  const hasChildren = Object.keys(node.subfolders).length > 0 || node.videos.length > 0
  const indent = depth * 20

  return (
    <div>
      <div
        onClick={() => setOpen(o => !o)}
        style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.3rem 0.5rem', cursor: hasChildren ? 'pointer' : 'default', userSelect: 'none', color: 'var(--muted)', fontSize: '0.82rem', paddingLeft: indent + 8 }}
      >
        <span style={{ width: 10, display: 'inline-flex', alignItems: 'center' }}>{hasChildren ? (open ? <ChevronDown size={12} style={{ opacity: 0.6 }} /> : <ChevronRight size={12} style={{ opacity: 0.6 }} />) : ''}</span>
        <Folder size={13} style={{ flexShrink: 0, color: 'var(--muted)' }} />
        <span style={{ fontWeight: 500, color: 'var(--text)' }}>{node.name}</span>
        <span style={{ opacity: 0.5 }}>({node.videos.length + Object.values(node.subfolders).reduce(function count(s, n): number { return s + n.videos.length + Object.values(n.subfolders).reduce(count, 0) }, 0)})</span>
        <span style={{ flex: 1 }} />
        <button className="icon-btn" title="Hide folder" onClick={e => { e.stopPropagation(); handleHideAll(e, getAllVideoIds(node)) }}
          style={{ fontSize: '0.72rem', padding: '0.1rem 0.35rem', color: 'var(--muted)', opacity: 0.6 }}>Hide folder</button>
      </div>
      {open && <>
        {Object.values(node.subfolders).sort((a, b) => a.name.localeCompare(b.name)).map(child => (
          <TreeNode key={child.fullPath} node={child} depth={depth + 1} selected={selected} removing={removing} snapshots={snapshots} toggleSelect={toggleSelect} handleRemove={handleRemove} handleHideAll={handleHideAll} navigate={navigate} treeToggle={treeToggle} frameInterval={frameInterval} />
        ))}
        {node.videos.map(video => {
          const isSelected = selected.has(video.id)
          const totalFrames = Math.max(1, Math.floor(video.duration / frameInterval))
          const eta = (isAudioPhase(video.status))
            ? transcribeEtaLabel(snapshots.current[video.id], video.transcribe_progress ?? 0)
            : etaLabel(snapshots.current[video.id], video.frame_count, totalFrames)
          return (
            <div
              key={video.id}
              onClick={() => { if (removing !== video.id) toggleSelect(video.id) }}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.3rem 0.5rem', paddingLeft: indent + 34, cursor: 'pointer', opacity: removing === video.id ? 0.4 : 1, background: isSelected ? 'rgba(var(--accent-rgb, 99,102,241), 0.12)' : 'transparent', borderRadius: 6, marginBottom: 1 }}
            >
              <div style={{ flexShrink: 0, width: 14, height: 14, borderRadius: 3, background: isSelected ? 'var(--accent)' : 'transparent', border: `2px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {isSelected && <Check size={10} color="#fff" strokeWidth={3} />}
              </div>
              <VideoIcon size={12} style={{ color: 'var(--muted)', flexShrink: 0 }} />
              <span style={{ flex: 1, fontSize: '0.85rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{video.display_name || video.name}</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--muted)', flexShrink: 0 }}>{fmt(video.duration)}</span>
              <StatusBadge status={video.status} frameCount={video.frame_count} eta={eta} transcribeProgress={video.transcribe_progress} fuseProgress={video.fuse_progress} />
              {isAnalysisActive(video.status) && <button className="icon-btn" title="Stop analysis" onClick={e => { e.stopPropagation(); api.cancelAnalysis(video.id).catch(() => {}) }} style={{ padding: '0.1rem 0.2rem', color: 'var(--error)', flexShrink: 0 }}><Square size={12} /></button>}              {video.status === 'retranscribing' && <button className="icon-btn" title="Stop audio" onClick={e => { e.stopPropagation(); api.cancelAnalysis(video.id).catch(() => {}) }} style={{ padding: '0.1rem 0.2rem', color: 'var(--error)', flexShrink: 0 }}><Square size={12} /></button>}              <button className="icon-btn" title="View details" onClick={e => { e.stopPropagation(); navigate(`/videos/${video.id}`) }} style={{ padding: '0.1rem 0.3rem', color: 'var(--muted)', flexShrink: 0 }}><ArrowRight size={14} /></button>
              <button className="icon-btn" title="Hide" onClick={e => handleRemove(e, video.id)} style={{ padding: '0.1rem 0.2rem', color: 'var(--muted)', flexShrink: 0 }}><X size={14} /></button>
            </div>
          )
        })}
      </>}
    </div>
  )
}

function TreeView(props: ViewProps) {
  const { root } = buildTree(props.videos)
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
      {Object.values(root.subfolders).length === 0 && root.videos.length > 0
        ? <TreeNode node={root} depth={0} {...props} />
        : <>
            {Object.values(root.subfolders).sort((a, b) => a.name.localeCompare(b.name)).map(child => (
              <TreeNode key={child.fullPath} node={child} depth={0} {...props} />
            ))}
            {root.videos.map(video => {
              const isSelected = props.selected.has(video.id)
              const totalFrames = Math.max(1, Math.floor(video.duration / props.frameInterval))
              return (
                <div key={video.id} onClick={() => { if (props.removing !== video.id) props.toggleSelect(video.id) }}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.3rem 0.5rem', paddingLeft: 34, cursor: 'pointer', background: isSelected ? 'rgba(var(--accent-rgb, 99,102,241), 0.12)' : 'transparent' }}>
                  <div style={{ flexShrink: 0, width: 14, height: 14, borderRadius: 3, background: isSelected ? 'var(--accent)' : 'transparent', border: `2px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {isSelected && <Check size={10} color="#fff" strokeWidth={3} />}
                  </div>
                  <VideoIcon size={12} style={{ color: 'var(--muted)', flexShrink: 0 }} />
                  <span style={{ flex: 1, fontSize: '0.85rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{video.display_name || video.name}</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>{fmt(video.duration)}</span>
                  <StatusBadge status={video.status} frameCount={video.frame_count} eta={(isAudioPhase(video.status)) ? transcribeEtaLabel(props.snapshots.current[video.id], video.transcribe_progress ?? 0) : etaLabel(props.snapshots.current[video.id], video.frame_count, totalFrames)} transcribeProgress={video.transcribe_progress} fuseProgress={video.fuse_progress} />
                  {isAnalysisActive(video.status) && <button className="icon-btn" title="Stop analysis" onClick={e => { e.stopPropagation(); api.cancelAnalysis(video.id).catch(() => {}) }} style={{ padding: '0.1rem 0.2rem', color: 'var(--error)' }}><Square size={12} /></button>}                  {video.status === 'retranscribing' && <button className="icon-btn" title="Stop audio" onClick={e => { e.stopPropagation(); api.cancelAnalysis(video.id).catch(() => {}) }} style={{ padding: '0.1rem 0.2rem', color: 'var(--error)' }}><Square size={12} /></button>}                  <button className="icon-btn" title="View details" onClick={e => { e.stopPropagation(); props.navigate(`/videos/${video.id}`) }} style={{ padding: '0.1rem 0.3rem', color: 'var(--muted)' }}><ArrowRight size={14} /></button>
                  <button className="icon-btn" title="Hide" onClick={e => props.handleRemove(e, video.id)} style={{ padding: '0.1rem 0.2rem', color: 'var(--muted)' }}><X size={14} /></button>
                </div>
              )
            })}
          </>
      }
    </div>
  )
}

// ── Folder view ───────────────────────────────────────────────────────────────

function FolderView(props: ViewProps) {
  const { root, prefix } = buildTree(props.videos)
  const [currentPath, setCurrentPath] = useState(prefix)

  const node = getNodeByPath(root, currentPath) || root

  // Build breadcrumb segments relative to root
  const crumbs: { name: string; path: string }[] = []
  const rootParts = prefix.split('/')
  const curParts = currentPath.split('/')
  for (let i = rootParts.length - 1; i < curParts.length; i++) {
    crumbs.push({ name: curParts[i] || '/', path: curParts.slice(0, i + 1).join('/') })
  }

  const folders = Object.values(node.subfolders).sort((a, b) => a.name.localeCompare(b.name))
  const videos = node.videos

  return (
    <div>
      {/* Breadcrumb */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', marginBottom: '0.75rem', fontSize: '0.82rem', flexWrap: 'wrap' }}>
        {crumbs.map((c, i) => (
          <span key={c.path} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            {i > 0 && <span style={{ color: 'var(--muted)' }}>/</span>}
            <span
              onClick={() => setCurrentPath(c.path)}
              style={{ color: i === crumbs.length - 1 ? 'var(--text)' : 'var(--accent)', cursor: i === crumbs.length - 1 ? 'default' : 'pointer', fontWeight: i === crumbs.length - 1 ? 600 : 400 }}
            >{c.name}</span>
          </span>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '0.6rem' }}>
        {/* Folder tiles */}
        {folders.map(folder => (
          <div key={folder.fullPath} onDoubleClick={() => setCurrentPath(folder.fullPath)}
            style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '0.75rem 0.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', transition: 'border-color 0.15s' }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
            onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
          >
            <Folder size={32} style={{ color: 'var(--muted)' }} />
            <span style={{ fontSize: '0.75rem', textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '100%', color: 'var(--text)' }}>{folder.name}</span>
            <span style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>{folder.videos.length + Object.values(folder.subfolders).reduce(function c(s, n): number { return s + n.videos.length + Object.values(n.subfolders).reduce(c, 0) }, 0)} videos</span>
          </div>
        ))}

        {/* Video tiles */}
        {videos.map(video => {
          const isSelected = props.selected.has(video.id)
          const totalFrames = Math.max(1, Math.floor(video.duration / props.frameInterval))
          const eta = (isAudioPhase(video.status))
            ? transcribeEtaLabel(props.snapshots.current[video.id], video.transcribe_progress ?? 0)
            : etaLabel(props.snapshots.current[video.id], video.frame_count, totalFrames)
          return (
            <div key={video.id}
              onClick={() => { if (props.removing !== video.id) props.toggleSelect(video.id) }}
              style={{ background: 'var(--surface)', border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`, borderRadius: 'var(--radius)', overflow: 'hidden', display: 'flex', flexDirection: 'column', cursor: 'pointer', opacity: props.removing === video.id ? 0.4 : 1, transition: 'border-color 0.15s', outline: isSelected ? '2px solid var(--accent)' : 'none' }}
            >
              <div style={{ height: 80, background: '#000', position: 'relative', flexShrink: 0 }}>
                {video.thumbnail_path && (
                  <img src={`/api/frames/${video.id}/${encodeURIComponent(video.thumbnail_path.split('/').pop()!)}`} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                )}
                {isSelected && (
                  <div style={{ position: 'absolute', top: 4, left: 4, width: 16, height: 16, borderRadius: 4, background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Check size={10} color="#fff" strokeWidth={3} />
                  </div>
                )}
              </div>
              <div style={{ padding: '0.4rem 0.5rem', display: 'flex', flexDirection: 'column', gap: '0.2rem', flex: 1 }}>
                <span style={{ fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 500 }}>{video.display_name || video.name}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <span style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>{fmt(video.duration)}</span>
                  <span style={{ flex: 1 }} />
                  {isAnalysisActive(video.status) && <button className="icon-btn" title="Stop analysis" onClick={e => { e.stopPropagation(); api.cancelAnalysis(video.id).catch(() => {}) }} style={{ padding: '0.1rem 0.2rem', color: 'var(--error)' }}><Square size={12} /></button>}                  {video.status === 'retranscribing' && <button className="icon-btn" title="Stop audio" onClick={e => { e.stopPropagation(); api.cancelAnalysis(video.id).catch(() => {}) }} style={{ padding: '0.1rem 0.2rem', color: 'var(--error)' }}><Square size={12} /></button>}                  <button className="icon-btn" title="View details" onClick={e => { e.stopPropagation(); props.navigate(`/videos/${video.id}`) }} style={{ padding: '0.1rem 0.2rem', color: 'var(--muted)' }}><ArrowRight size={14} /></button>
                </div>
                <StatusBadge status={video.status} frameCount={video.frame_count} eta={eta} transcribeProgress={video.transcribe_progress} fuseProgress={video.fuse_progress} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function VideosPage() {
  const [videos, setVideos] = useState<Video[]>([])
  const [loading, setLoading] = useState(true)
  const [reimporting, setReimporting] = useState(false)
  const [reimportMsg, setReimportMsg] = useState('')
  const [removing, setRemoving] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [showModal, setShowModal] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>(() => (localStorage.getItem('vss_view_mode') as ViewMode) || 'card')
  const [filterText, setFilterText] = useState(() => localStorage.getItem('vss_filter') ?? '')
  const [sortBy, setSortBy] = useState<'name' | 'duration' | 'status'>(() => {
    const stored = localStorage.getItem('vss_sort_by')
    return (stored === 'name' || stored === 'duration' || stored === 'status') ? stored : 'name'
  })
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>(() => (localStorage.getItem('vss_sort_dir') as any) || 'asc')
  const [statusFilter, setStatusFilter] = useState<'all' | 'analyzed' | 'processing' | 'error' | 'ready'>(() => (localStorage.getItem('vss_status_filter') as any) || 'all')
  const [treeToggle, setTreeToggle] = useState<{ open: boolean; seq: number } | null>(null)
  const snapshots = useRef<Record<string, ProgressSnapshot>>({})
  const [frameInterval, setFrameInterval] = useState(5)

  const loadVideos = useCallback(async () => {
    try { setVideos(await api.listVideos()) } catch { /* ignore */ } finally { setLoading(false) }
  }, [])

  useEffect(() => {
    loadVideos()
    api.getSettings().then((s: AppSettings) => setFrameInterval(s.frame_extraction_interval || 5)).catch(() => {})
    const id = setInterval(async () => {
      const data = await api.listVideos().catch(() => null)
      if (!data) return
      setVideos(prev => {
        for (const v of prev) {
          const next = data.find(d => d.id === v.id)
          if (!next) continue
          const active = isAnalysisActive(v.status) || isAudioPhase(v.status)
          if (!active) continue
          let snap = snapshots.current[v.id]
          if (!snap) { snap = { frameCount: 0, ts: 0, tProgress: 0, tTs: 0 }; snapshots.current[v.id] = snap }
          // Frame-phase baseline: set on first observation using the *current*
          // frame count (not 0) so already-completed frames don't inflate rate.
          if ((v.status === 'video_processing' || v.status === 'processing') && next.frame_count > 0 && snap.ts === 0) {
            snap.frameCount = next.frame_count; snap.ts = Date.now()
          }
          // Transcription baseline — tracked separately since it begins only
          // after frame analysis finishes.
          if ((isAudioPhase(v.status)) && (next.transcribe_progress ?? 0) > 0 && snap.tTs === 0) {
            snap.tProgress = next.transcribe_progress ?? 0; snap.tTs = Date.now()
          }
        }
        return data
      })
    }, 5000)
    return () => clearInterval(id)
  }, [loadVideos])

  const toggleSelect = (videoId: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(videoId) ? next.delete(videoId) : next.add(videoId)
      return next
    })
  }

  const handleRemove = async (e: React.MouseEvent, videoId: string) => {
    e.preventDefault(); e.stopPropagation()
    if (!window.confirm('Hide this video from the library? You can restore it from Settings.')) return
    setRemoving(videoId)
    try {
      await api.hideVideo(videoId)
      setVideos(prev => prev.filter(v => v.id !== videoId))
      setSelected(prev => { const n = new Set(prev); n.delete(videoId); return n })
      delete snapshots.current[videoId]
    } catch { /* ignore */ } finally { setRemoving(null) }
  }

  const handleHideAll = async (e: React.MouseEvent, videoIds: string[]) => {
    e.preventDefault(); e.stopPropagation()
    if (!window.confirm(`Hide ${videoIds.length} video${videoIds.length !== 1 ? 's' : ''} in this folder? You can restore them from Settings.`)) return
    await Promise.all(videoIds.map(id => api.hideVideo(id).catch(() => {})))
    setVideos(prev => prev.filter(v => !videoIds.includes(v.id)))
    setSelected(prev => { const n = new Set(prev); videoIds.forEach(id => n.delete(id)); return n })
    videoIds.forEach(id => delete snapshots.current[id])
  }

  const handleRetranscribeSelected = async () => {
    const ids = [...selected]
    if (ids.length === 0) return
    if (!window.confirm(`Re-transcribe audio for ${ids.length} video${ids.length !== 1 ? 's' : ''}? This replaces only the transcript (visual frames are kept).`)) return
    await Promise.all(ids.map(id => api.retranscribeAudio(id).catch(() => {})))
    setSelected(new Set())
    loadVideos()
  }

  const handleReimport = async () => {
    setReimporting(true); setReimportMsg('')
    const before = videos.length
    try {
      await api.reimport()
      await loadVideos()
      const after = (await api.listVideos()).length
      const added = after - before
      setReimportMsg(added > 0 ? `${added} new video${added !== 1 ? 's' : ''} added.` : 'No new videos found.')
    } catch { setReimportMsg('Scan failed.') }
    finally { setReimporting(false) }
  }

  const processingCount = videos.filter(v => isAnalysisActive(v.status)).length
  const retranscribingCount = videos.filter(v => v.status === 'retranscribing').length
  const navigate = useNavigate()

  const setView = (v: ViewMode) => { setViewMode(v); localStorage.setItem('vss_view_mode', v) }
  const setSort = (by: typeof sortBy, dir: typeof sortDir) => {
    setSortBy(by); setSortDir(dir)
    localStorage.setItem('vss_sort_by', by); localStorage.setItem('vss_sort_dir', dir)
  }
  const toggleSortDir = () => {
    const next = sortDir === 'asc' ? 'desc' : 'asc'
    setSortDir(next); localStorage.setItem('vss_sort_dir', next)
  }

  const STATUS_ORDER: Record<string, number> = {
    queued: 0, scene_detection: 0, video_processing: 0, audio_processing: 0, fusing: 0,
    processing: 0, transcribing: 0, retranscribing: 0,
    analyzed: 1, ready: 2, error: 3,
  }

  const displayedVideos = [...videos]
    .filter(v => {
      if (statusFilter !== 'all') {
        const inProgress = statusFilter === 'processing'
        if (inProgress ? (!isAnalysisActive(v.status) && v.status !== 'retranscribing') : v.status !== statusFilter) return false
      }
      if (!filterText.trim()) return true
      const q = filterText.toLowerCase()
      return (v.display_name || v.name).toLowerCase().includes(q) || v.path.toLowerCase().includes(q) || v.tags.some(t => t.toLowerCase().includes(q))
    })
    .sort((a, b) => {
      let cmp = 0
      if (sortBy === 'name') cmp = (a.display_name || a.name).localeCompare(b.display_name || b.name)
      else if (sortBy === 'duration') cmp = a.duration - b.duration
      else if (sortBy === 'status') cmp = (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9)
      return sortDir === 'asc' ? cmp : -cmp
    })

  const viewProps: ViewProps = { videos: displayedVideos, selected, removing, snapshots, toggleSelect, handleRemove, handleHideAll, navigate, treeToggle, frameInterval }

  return (
    <div className="page">
      <div className="page-header" style={{ marginBottom: '1rem' }}>
        <div>
          <h1>Video Library</h1>
          <p style={{ color: 'var(--muted)', marginTop: '0.4rem', fontSize: '0.88rem' }}>
            {videos.length} video{videos.length !== 1 ? 's' : ''}{processingCount > 0 && ` · ${processingCount} analyzing`}{retranscribingCount > 0 && ` · ${retranscribingCount} re-transcribing audio`}
          </p>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.6rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {reimportMsg && <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>{reimportMsg}</span>}
            <button className="ghost" onClick={handleReimport} disabled={reimporting} style={{ fontSize: '0.85rem', whiteSpace: 'nowrap' }}>
              {reimporting ? <><span className="spinner" style={{ width: 13, height: 13, marginRight: '0.4rem' }} />Scanning…</> : '↻ Reimport'}
            </button>
          </div>
          <ModelStatusBar />
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '3rem', justifyContent: 'center', color: 'var(--muted)' }}>
          <span className="spinner" /> Loading…
        </div>
      ) : videos.length === 0 ? (
        <div className="empty"><h3>No videos yet</h3><p>Add video files to your configured library folders and they'll appear here automatically.</p></div>
      ) : (
        <>
            {/* Filter + sort bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <input
              value={filterText}
              onChange={e => { setFilterText(e.target.value); localStorage.setItem('vss_filter', e.target.value) }}
              placeholder="Filter by name, path or tag…"
              style={{ flex: 1, fontSize: '0.85rem', padding: '0.45rem 0.75rem' }}
            />
            <select
              value={statusFilter}
              onChange={e => { setStatusFilter(e.target.value as any); localStorage.setItem('vss_status_filter', e.target.value) }}
              style={{ fontSize: '0.82rem', padding: '0.45rem 0.5rem', background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', cursor: 'pointer', flexShrink: 0 }}
            >
              <option value="all">All</option>
              <option value="ready">Ready</option>
              <option value="analyzed">Analyzed</option>
              <option value="processing">In progress</option>
              <option value="error">Errored</option>
            </select>
            <select
              value={sortBy}
              onChange={e => setSort(e.target.value as any, sortDir)}
              style={{ fontSize: '0.82rem', padding: '0.45rem 0.5rem', background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', cursor: 'pointer' }}
            >
              <option value="name">Name</option>
              <option value="duration">Duration</option>
              <option value="status">Status</option>
            </select>
            <button className="ghost" onClick={toggleSortDir} title={sortDir === 'asc' ? 'Ascending' : 'Descending'}
              style={{ fontSize: '0.85rem', padding: '0.45rem 0.6rem', minWidth: 0 }}>
              {sortDir === 'asc' ? '↑' : '↓'}
            </button>
          </div>

          {/* Selection + view mode toolbar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem', padding: '0.6rem 1rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', minHeight: '2.6rem' }}>
            {selected.size > 0 ? (
              <>
                <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>{selected.size} selected</span>
                <button onClick={() => setShowModal(true)} style={{ fontSize: '0.85rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}><Play size={13} /> Analyze {selected.size} video{selected.size !== 1 ? 's' : ''}</button>
                <button className="ghost" onClick={handleRetranscribeSelected} title="Re-run audio transcription only (keeps visual frames)" style={{ fontSize: '0.85rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}><Mic size={13} /> Re-transcribe audio</button>
                <button className="ghost" style={{ fontSize: '0.85rem' }} onClick={() => setSelected(new Set())}>Clear</button>
              </>
            ) : (
              <>
                <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                  {filterText.trim() ? `${displayedVideos.length} of ${videos.length} videos` : 'Click videos to select, then press Analyze to process them'}
                </span>
                <button className="ghost" style={{ fontSize: '0.8rem', padding: '0.2rem 0.6rem' }} onClick={() => setSelected(new Set(displayedVideos.map(v => v.id)))}>Select all</button>
              </>
            )}
            <span style={{ flex: 1 }} />
            {(['card', 'tree', 'folder'] as ViewMode[]).map(mode => (
              <button key={mode} className={viewMode === mode ? '' : 'ghost'} onClick={() => setView(mode)}
                style={{ fontSize: '0.8rem', padding: '0.25rem 0.6rem', minWidth: 0 }}
                title={mode.charAt(0).toUpperCase() + mode.slice(1) + ' view'}
              >
                {mode === 'card' ? <LayoutGrid size={15} /> : mode === 'tree' ? <List size={15} /> : <Folder size={15} />}
              </button>
            ))}
            {viewMode === 'tree' && (
              <button className="ghost" title={(treeToggle?.open ?? true) ? 'Collapse all' : 'Expand all'}
                onClick={() => setTreeToggle(prev => ({ open: !(prev?.open ?? true), seq: (prev?.seq ?? 0) + 1 }))}
                style={{ fontSize: '0.8rem', padding: '0.25rem 0.6rem', minWidth: 0 }}>
                {(treeToggle?.open ?? true) ? '▲' : '▼'}
              </button>
            )}
          </div>
          {viewMode === 'card' && (
          <div className="video-grid">
            {displayedVideos.map(video => {
              const totalFrames = Math.max(1, Math.floor(video.duration / frameInterval))
              const eta = (isAudioPhase(video.status))
                ? transcribeEtaLabel(snapshots.current[video.id], video.transcribe_progress ?? 0)
                : etaLabel(snapshots.current[video.id], video.frame_count, totalFrames)
              const isSelected = selected.has(video.id)
              return (
                <div key={video.id} style={{ position: 'relative' }}>
                  <div
                    className="video-card"
                    onClick={() => { if (removing !== video.id) toggleSelect(video.id) }}
                    style={{ opacity: removing === video.id ? 0.4 : 1, pointerEvents: removing === video.id ? 'none' : undefined, outline: isSelected ? '2px solid var(--accent)' : undefined, cursor: 'pointer' }}
                  >
                    <div style={{ margin: '-0.75rem -0.75rem 0.5rem', borderRadius: 'var(--radius) var(--radius) 0 0', overflow: 'hidden', height: 120, background: 'var(--surface2)', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', flexShrink: 0 }}>
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.18, position: 'absolute' }}><path d="M15 10l4.553-2.277A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      <img
                        src={video.thumbnail_path ? `/api/frames/${video.id}/${encodeURIComponent(video.thumbnail_path.split('/').pop()!)}` : `/api/videos/${video.id}/thumbnail`}
                        alt="" loading="lazy"
                        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                        onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
                      />
                    </div>
                    <div className="video-card-header">
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', flex: 1, minWidth: 0 }}>
                        <div
                          style={{
                            flexShrink: 0, marginTop: 2,
                            width: 16, height: 16, borderRadius: 4,
                            background: isSelected ? 'var(--accent)' : 'transparent',
                            border: `2px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            transition: 'all 0.15s', pointerEvents: 'none'
                          }}
                        >
                          {isSelected && <Check size={10} color="#fff" strokeWidth={3} />}
                        </div>
                        <span className="video-title" title={video.display_name || video.name}>{video.display_name || video.name}</span>
                      </div>
                      <div style={{ display: 'flex', gap: '0.1rem' }}>
                        {isAnalysisActive(video.status) && (
                          <button className="icon-btn" title="Stop analysis" onClick={e => { e.stopPropagation(); api.cancelAnalysis(video.id).catch(() => {}) }} style={{ padding: '0.2rem 0.3rem', color: 'var(--error)' }}><Square size={12} /></button>
                        )}
                        <button className="icon-btn" title="View details" onClick={e => { e.stopPropagation(); navigate(`/videos/${video.id}`) }} style={{ padding: '0.2rem 0.4rem', color: 'var(--muted)' }}><ArrowRight size={14} /></button>
                        <button className="icon-btn" title="Remove from library" onClick={e => handleRemove(e, video.id)} style={{ padding: '0.2rem 0.3rem', color: 'var(--muted)' }}><X size={14} /></button>
                      </div>
                    </div>
                    <div className="video-meta" style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', rowGap: '0.3rem' }}>
                      <span>{fmt(video.duration)}</span>
                      <div style={{ marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', minWidth: 0 }}>
                        {video.status === 'analyzed' && !video.has_transcript && (
                          <span title="No audio transcription — spoken words won't appear in search results" style={{ display: 'inline-flex', flexShrink: 0, color: 'var(--warning)', marginRight: '0.4rem' }}>
                            <VolumeX size={12} />
                          </span>
                        )}
                        {video.status === 'analyzed' && !video.has_frames && (
                          <span title="No visual analysis — on-screen content won't appear in search results" style={{ display: 'inline-flex', flexShrink: 0, color: 'var(--warning)', marginRight: '0.4rem' }}>
                            <VideoOff size={12} />
                          </span>
                        )}
                        <StatusBadge status={video.status} frameCount={video.frame_count} eta={eta} transcribeProgress={video.transcribe_progress} />
                      </div>
                    </div>
                    {(video.status === 'video_processing' || video.status === 'processing') && video.frame_count > 0 && (
                      <div className="progress-wrap" style={{ marginTop: 4 }}>
                        <div className="progress-bar" style={{ width: `${Math.min(100, (video.frame_count / totalFrames) * 85)}%` }} />
                      </div>
                    )}
                    {(isAudioPhase(video.status)) && (
                      <div className="progress-wrap" style={{ marginTop: 4 }}>
                        <div className="progress-bar audio" style={{ width: `${Math.max(3, Math.min(100, (video.transcribe_progress || 0) * 100))}%` }} />
                      </div>
                    )}
                    {video.context && <p style={{ fontSize: '0.72rem', color: 'var(--muted)', fontStyle: 'italic', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={video.context}>"{video.context}"</p>}
                    {video.tags.length > 0 && (
                      <div className="video-tags">{video.tags.map(t => <span key={t} className="tag">{t}</span>)}</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
          )}
          {viewMode === 'tree' && <TreeView {...viewProps} />}
          {viewMode === 'folder' && <FolderView {...viewProps} />}
        </>
      )}

      {showModal && (
        <AnalyzeModal
          videoIds={[...selected]}
          videos={videos}
          onClose={() => setShowModal(false)}
          onDone={() => { setShowModal(false); setSelected(new Set()); loadVideos() }}
        />
      )}
    </div>
  )
}

const RECENT_CONTEXTS_KEY = 'vss_recent_contexts'

function getRecentContexts(): string[] {
  try { return JSON.parse(localStorage.getItem(RECENT_CONTEXTS_KEY) || '[]') } catch { return [] }
}

function saveContext(ctx: string) {
  if (!ctx.trim()) return
  const prev = getRecentContexts().filter(c => c !== ctx)
  localStorage.setItem(RECENT_CONTEXTS_KEY, JSON.stringify([ctx, ...prev].slice(0, 5)))
}

export function AnalyzeModal({ videoIds, videos, onClose, onDone }: {
  videoIds: string[]
  videos: Video[]
  onClose: () => void
  onDone: () => void
}) {
  const selectedVideos = videos.filter(v => videoIds.includes(v.id))
  const anyHasFrames = selectedVideos.some(v => v.has_frames)
  const anyHasTranscript = selectedVideos.some(v => v.has_transcript)
  // Each stage defaults on when some selected video is missing its output: a
  // never-analyzed (or errored) video pre-checks Video+Audio; a frames-only
  // "quick pass" video pre-checks just Audio (+Fuse); a fully-analyzed selection
  // pre-checks everything (a deliberate redo).
  const needsVideo = selectedVideos.some(v => v.status === 'ready' || v.status === 'error' || (v.status === 'analyzed' && !v.has_frames))
  const needsAudio = selectedVideos.some(v => v.status === 'ready' || v.status === 'error' || (v.status === 'analyzed' && !v.has_transcript))
  const allFullyAnalyzed = !needsVideo && !needsAudio

  const [context, setContext] = useState('')
  const [recents] = useState<string[]>(getRecentContexts)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [pendingIds, setPendingIds] = useState(videoIds)
  // Three independent stages: caption frames (Video), transcribe (Audio), and the
  // cross-modal fusion pass (Fuse). Fuse defaults on and auto-skips windows that
  // lack a channel, matching the old implicit auto-fuse.
  const [doVideo, setDoVideo] = useState(needsVideo || allFullyAnalyzed)
  const [doAudio, setDoAudio] = useState(needsAudio || allFullyAnalyzed)
  const [doFuse, setDoFuse] = useState(true)

  // Resume only applies to the Video stage on content that already has frames: it
  // gap-fills missing captions instead of wiping and re-captioning. Audio and Fuse
  // always rebuild, so resume vs start-over is a Video-only concept.
  const resumableCount = selectedVideos.filter(v => (v.status === 'analyzed' || v.status === 'error') && v.has_frames).length
  const showResumeChoice = doVideo && resumableCount > 0
  const [resume, setResume] = useState(true)

  // Fusion needs both channels present after this run. A video will have frames if
  // it already does or we're (re)captioning, and a transcript likewise. Disable the
  // Fuse stage when the selection won't end up with both.
  const willHaveFrames = doVideo || anyHasFrames
  const willHaveTranscript = doAudio || anyHasTranscript
  const canFuse = willHaveFrames && willHaveTranscript
  const fuseActive = doFuse && canFuse

  // Ask for context only when captioning fresh frames: a Video stage that isn't a
  // resume. Resume reuses each video's stored context; audio/fuse don't caption.
  const askContext = doVideo && !(showResumeChoice && resume)
  // When any selected video already has a context, offer to keep it (default on);
  // unchecking reveals the box to write a new one that overwrites it.
  const existingContexts = selectedVideos.map(v => (v.context || '').trim()).filter(Boolean)
  const hasExistingContext = existingContexts.length > 0
  const [keepContext, setKeepContext] = useState(true)
  // The free-text box shows only when we're NOT keeping an existing context.
  const showContextInput = askContext && (!hasExistingContext || !keepContext)

  const nothingSelected = !doVideo && !doAudio && !fuseActive

  // Subtle, combination-aware note about what happens to the cross-modal fused
  // segments for this stage/resume/fuse choice. Only meaningful for content that
  // already has both channels (so fused segments plausibly exist).
  const mightHaveFused = selectedVideos.some(v => v.status === 'analyzed' && v.has_frames && v.has_transcript)
  const fuseNote: { warn: boolean; text: string } | null = (() => {
    if (nothingSelected) return null
    if (fuseActive) return { warn: false, text: 'Fused segments will be rebuilt from the latest captions and transcript.' }
    if (!mightHaveFused) return null
    // Fuse is available for this selection but turned off.
    if (doVideo && (!showResumeChoice || !resume))
      return { warn: true, text: 'Fuse is off — re-running Video from scratch will also delete the existing fused segments.' }
    if (doVideo || doAudio)
      return { warn: false, text: 'Fuse is off — existing fused segments are kept, but won’t reflect these changes until you re-fuse.' }
    return null
  })()

  const handleSubmit = async () => {
    setSubmitting(true); setSubmitError(null)
    const useResume = showResumeChoice && resume
    // Keeping the existing context = send none, so the backend reuses each video's
    // own saved context (correct even when selected videos have different ones).
    const ctx = showContextInput ? (context.trim() || undefined) : undefined
    if (showContextInput && context.trim()) saveContext(context.trim())
    const outcomes = await Promise.allSettled(pendingIds.map(id => api.reprocessVideo(id, ctx, useResume, doVideo, doAudio, fuseActive)))
    const failed = outcomes.filter((outcome): outcome is PromiseRejectedResult => outcome.status === 'rejected')
    if (failed.length) {
      setPendingIds(pendingIds.filter((_, index) => outcomes[index].status === 'rejected'))
      setSubmitError(`${failed.length} of ${pendingIds.length} could not be queued: ${apiErrorMessage(failed[0].reason, 'Check the server and try again.')}`)
      setSubmitting(false)
    } else {
      onDone()
    }
  }

  return createPortal(
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}
      onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem', width: '100%', maxWidth: 480, maxHeight: 'calc(100vh - 2rem)', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '1.1rem' }}>Analyze {videoIds.length} video{videoIds.length !== 1 ? 's' : ''}</h2>
          <button className="icon-btn" onClick={onClose} style={{ lineHeight: 0 }}><X size={16} /></button>
        </div>

        {submitError && <p role="alert" style={{ color: 'var(--error)', fontSize: '0.82rem', margin: 0 }}>{submitError}</p>}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', maxHeight: 120, overflowY: 'auto' }}>
          {selectedVideos.map(v => (
            <div key={v.id} style={{ fontSize: '0.82rem', color: 'var(--muted)' }}>• {v.display_name || v.name}</div>
          ))}
        </div>

        <div>
          <label style={{ fontSize: '0.82rem', color: 'var(--muted)', display: 'block', marginBottom: '0.4rem' }}>
            Stages to run
          </label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            {([
              { key: 'video', active: doVideo, toggle: () => setDoVideo(v => !v), icon: <VideoIcon size={13} />, label: 'Video', desc: 'Caption frames', disabled: false },
              { key: 'audio', active: doAudio, toggle: () => setDoAudio(v => !v), icon: <Mic size={13} />, label: 'Audio', desc: 'Transcribe', disabled: false },
              { key: 'fuse', active: fuseActive, toggle: () => setDoFuse(v => !v), icon: <Sparkles size={13} />, label: 'Fuse', desc: canFuse ? 'Merge both' : 'Needs both', disabled: !canFuse },
            ] as const).map(s => (
              <button
                key={s.key}
                onClick={s.toggle}
                disabled={s.disabled}
                className={s.active ? '' : 'ghost'}
                title={s.key === 'fuse' && !canFuse ? 'Fusion needs both video captions and a transcript' : undefined}
                style={{ flex: 1, minWidth: 0, fontSize: '0.8rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.15rem', padding: '0.45rem 0.4rem', textAlign: 'center', whiteSpace: 'normal', opacity: s.disabled ? 0.45 : 1, cursor: s.disabled ? 'not-allowed' : 'pointer' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontWeight: 600 }}>{s.icon}{s.label}</span>
                <span style={{ fontSize: '0.68rem', opacity: 0.8, fontWeight: 400 }}>{s.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {showResumeChoice && (
          <div>
            <label style={{ fontSize: '0.82rem', color: 'var(--muted)', display: 'block', marginBottom: '0.4rem' }}>
              {resumableCount} of {selectedVideos.length} already analyzed{selectedVideos.some(v => v.status === 'error') ? ' or errored' : ''}
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={() => setResume(true)}
                className={resume ? '' : 'ghost'}
                style={{ flex: 1, minWidth: 0, fontSize: '0.82rem', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.15rem', padding: '0.5rem 0.7rem', textAlign: 'left', whiteSpace: 'normal' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontWeight: 600 }}><Play size={13} /> Resume</span>
                <span style={{ fontSize: '0.72rem', opacity: 0.8, fontWeight: 400, lineHeight: 1.3 }}>Keep existing results, fill in only what's missing</span>
              </button>
              <button
                onClick={() => setResume(false)}
                className={resume ? 'ghost' : ''}
                style={{ flex: 1, minWidth: 0, fontSize: '0.82rem', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.15rem', padding: '0.5rem 0.7rem', textAlign: 'left', whiteSpace: 'normal' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontWeight: 600 }}><RotateCcw size={13} /> Start over</span>
                <span style={{ fontSize: '0.72rem', opacity: 0.8, fontWeight: 400, lineHeight: 1.3 }}>Delete all results and re-analyze from scratch</span>
              </button>
            </div>
          </div>
        )}

        {askContext && hasExistingContext && (
          <div>
            <label style={{ fontSize: '0.82rem', color: 'var(--muted)', display: 'block', marginBottom: '0.4rem' }}>Context</label>
            <label style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', cursor: 'pointer', fontSize: '0.82rem' }}>
              <input type="checkbox" checked={keepContext} onChange={e => setKeepContext(e.target.checked)} style={{ marginTop: '0.2rem' }} />
              <span style={{ minWidth: 0 }}>
                <span style={{ fontWeight: 600 }}>Keep existing context</span>
                <span style={{ display: 'block', fontSize: '0.72rem', color: 'var(--muted)', marginTop: '0.15rem', lineHeight: 1.3, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {videoIds.length > 1 ? "Reuse each video's saved context" : `“${existingContexts[0]}”`}
                </span>
              </span>
            </label>
          </div>
        )}

        {showContextInput && (
          <>
            <div>
              <label style={{ fontSize: '0.82rem', color: 'var(--muted)', display: 'block', marginBottom: '0.4rem' }}>
                {hasExistingContext
                  ? <>New context <span style={{ fontWeight: 400 }}>(replaces the existing one)</span></>
                  : <>Context <span style={{ fontWeight: 400 }}>(optional — helps the AI better understand the video)</span></>}
              </label>
              <textarea
                value={context}
                onChange={e => setContext(e.target.value)}
                placeholder='e.g. "Screen recording of Home Assistant dashboard setup"'
                rows={3}
                style={{ resize: 'vertical', fontFamily: 'inherit' }}
                autoFocus
              />
            </div>

            {recents.length > 0 && (
              <div>
                <p style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '0.4rem' }}>Recent contexts:</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  {recents.map((r, i) => (
                    <button key={i} className="ghost" onClick={() => setContext(r)}
                      style={{ fontSize: '0.78rem', textAlign: 'left', padding: '0.3rem 0.6rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {r}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {showResumeChoice && resume && (
          <p style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>
            Resuming reuses each video's original context.
          </p>
        )}

        {fuseNote && (
          <p style={{ display: 'flex', alignItems: 'flex-start', gap: '0.4rem', fontSize: '0.76rem', lineHeight: 1.35, color: fuseNote.warn ? 'var(--warning)' : 'var(--muted)', margin: 0 }}>
            <Sparkles size={13} style={{ flexShrink: 0, marginTop: '0.1rem', opacity: 0.85 }} />
            <span>{fuseNote.text}</span>
          </p>
        )}

        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', alignItems: 'center', marginTop: '0.25rem' }}>
          {nothingSelected && <span style={{ fontSize: '0.75rem', color: 'var(--muted)', marginRight: 'auto' }}>Select at least one stage</span>}
          <button className="ghost" onClick={onClose} disabled={submitting}>Cancel</button>
          <button onClick={handleSubmit} disabled={submitting || nothingSelected}>
            {submitting ? <><span className="spinner" style={{ width: 13, height: 13, marginRight: '0.4rem' }} />Starting…</> : (
              showResumeChoice && !resume
                ? <><RotateCcw size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} /> Start Over</>
                : <><Play size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} /> {showResumeChoice ? 'Resume Analysis' : 'Start Analysis'}</>
            )}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}
