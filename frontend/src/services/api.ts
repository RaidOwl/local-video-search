import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface QdrantIssue {
  type: 'missing' | 'incomplete' | 'orphaned'
  video_id?: string
  video_ids?: string[]
  name?: string
  db_frames?: number
  qdrant_points: number
}

export interface QdrantHealth {
  reachable: boolean
  collection_exists: boolean
  total_points: number
  issues: QdrantIssue[]
  summary: {
    videos_checked: number
    ok: number
    missing: number
    incomplete: number
    orphaned_points: number
  }
  error?: string
}

export interface AppSettings {
  // Deployment mode (read-only from .env — not editable via UI)
  ai_backend: string
  vram_tier?: string
  model_family?: string
  backend_warning?: string
  video_dirs?: string[]
  // Resolved model info (local modes only)
  _resolved_model?: string
  _resolved_embed?: string
  _resolved_tier?: string
  _resolved_family?: string
  // Editable settings
  vllm_embed_url: string
  vllm_vlm_url: string
  vllm_url: string
  qdrant_url: string
  vllm_embed_model: string
  vllm_vlm_model: string
  vllm_model: string
  whisper_model: string
  frame_extraction_interval: number
  embed_batch_size: number
  max_concurrent_jobs: number
  import_blacklist: string[]
  auto_import_interval: number
  auto_analyze: boolean
  default_context: string
}

export interface Video {
  id: string
  name: string
  display_name: string | null
  path: string
  duration: number
  summary?: string
  tags: string[]
  status: 'ready' | 'processing' | 'analyzed' | 'error' | 'retranscribing'
  frame_count: number
  thumbnail_path: string | null
  context: string | null
  analyzed_at?: string | null
}

export interface VideoFrame {
  id: string
  timestamp: number
  caption: string
  path: string
}

export interface SearchResult {
  video_id: string
  name: string
  path: string
  duration: number
  score: number
  matching_frames: number
  segments: Array<{
    video_id: string
    start_time: number
    end_time: number
    caption: string
    score: number
    frame_path: string
  }>
}

export interface Segment {
  video_id: string
  video_name: string
  start_time: number
  end_time: number
  caption: string
  score: number
  frame_path: string
}

export const api = {
  health: async () => {
    const res = await client.get('/health')
    return res.data
  },

  reimport: async (): Promise<{ total: number }> => {
    const res = await client.post('/reimport')
    return res.data
  },

  search: async (query: string, limit?: number, threshold?: number, sources?: string, expand?: boolean): Promise<{ results: SearchResult[]; query: string }> => {
    const res = await client.get('/search', { params: { q: query, limit, threshold, sources, expand } })
    return res.data
  },

  searchSegments: async (videoId: string, query: string, limit?: number): Promise<{ segments: Segment[]; video_id: string; query: string }> => {
    const res = await client.get('/search/segments', { params: { video_id: videoId, q: query, limit } })
    return res.data
  },

  listVideos: async (): Promise<Video[]> => {
    const res = await client.get('/videos')
    return res.data
  },

  listHiddenVideos: async (): Promise<Video[]> => {
    const res = await client.get('/videos', { params: { hidden: true } })
    return res.data
  },

  getVideo: async (id: string): Promise<Video> => {
    const res = await client.get(`/videos/${id}`)
    return res.data
  },

  summarizeVideo: async (id: string): Promise<{ status: string; video_id: string }> => {
    const res = await client.post(`/summarize/${id}`)
    return res.data
  },

  updateVideo: async (id: string, data: { display_name?: string; tags?: string[]; thumbnail_path?: string }): Promise<Video> => {
    const res = await client.patch(`/videos/${id}`, data)
    return res.data
  },

  hideVideo: async (id: string): Promise<void> => {
    await client.delete(`/videos/${id}`)
  },

  unhideVideo: async (id: string): Promise<void> => {
    await client.post(`/videos/${id}/unhide`)
  },

  reprocessVideo: async (id: string, context?: string): Promise<{ status: string; video_id: string }> => {
    const res = await client.post(`/reprocess/${id}`, { context: context ?? null })
    return res.data
  },

  retranscribeAudio: async (id: string): Promise<{ status: string; video_id: string }> => {
    const res = await client.post(`/retranscribe/${id}`)
    return res.data
  },

  cancelAnalysis: async (id: string): Promise<{ status: string; video_id: string }> => {
    const res = await client.post(`/cancel/${id}`)
    return res.data
  },

  getVideoFrames: async (id: string): Promise<{ frames: VideoFrame[]; video_id: string }> => {
    const res = await client.get(`/videos/${id}/frames`)
    return res.data
  },

  getTranscript: async (id: string): Promise<{ segments: Array<{ id: string; timestamp: number; text: string }>; count: number }> => {
    const res = await client.get(`/videos/${id}/transcript`)
    return res.data
  },

  updateFrameCaption: async (videoId: string, frameId: string, caption: string): Promise<void> => {
    await client.patch(`/videos/${videoId}/frames/${frameId}`, { caption })
  },

  modelStatus: async (): Promise<{ services: Array<{ name: string; url: string; ok: boolean; models: string[] }> }> => {
    const res = await client.get('/status/models')
    return res.data
  },

  qdrantHealth: async (): Promise<QdrantHealth> => {
    const res = await client.get('/status/qdrant')
    return res.data
  },

  purgeOrphanedEmbeddings: async (): Promise<{ deleted_points: number; orphaned_video_ids: string[] }> => {
    const res = await client.post('/status/qdrant/purge-orphans')
    return res.data
  },

  getSettings: async (): Promise<AppSettings> => {
    const res = await client.get('/settings')
    return res.data
  },

  updateSettings: async (s: Partial<AppSettings>): Promise<AppSettings> => {
    const res = await client.put('/settings', s)
    return res.data
  },
}