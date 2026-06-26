import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

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
}

export interface Video {
  id: string
  name: string
  display_name: string | null
  path: string
  duration: number
  summary?: string
  tags: string[]
  status: 'ready' | 'processing' | 'analyzed' | 'error'
  frame_count: number
  thumbnail_path: string | null
  context: string | null
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

export interface IngestResponse {
  video_id: string
  status: string
  frames_processed: number
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

  ingest: async (directory?: string, videoPath?: string): Promise<IngestResponse> => {
    const res = await client.post('/ingest', { directory, video_path: videoPath })
    return res.data
  },

  browse: async (path: string): Promise<{ path: string; parent: string | null; entries: { name: string; path: string; type: 'dir' | 'video' }[] }> => {
    const res = await client.get('/browse', { params: { path } })
    return res.data
  },

  search: async (query: string, limit?: number, threshold?: number): Promise<{ results: SearchResult[]; query: string }> => {
    const res = await client.get('/search', { params: { q: query, limit, threshold } })
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

  getVideo: async (id: string): Promise<Video> => {
    const res = await client.get(`/videos/${id}`)
    return res.data
  },

  getVideoSummary: async (id: string): Promise<{ summary: string | null; status: string }> => {
    const res = await client.get(`/videos/${id}/summary`)
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

  deleteVideo: async (id: string): Promise<void> => {
    await client.delete(`/videos/${id}`)
  },

  reprocessVideo: async (id: string, context?: string): Promise<{ status: string; video_id: string }> => {
    const res = await client.post(`/reprocess/${id}`, { context: context ?? null })
    return res.data
  },

  getVideoFrames: async (id: string): Promise<{ frames: VideoFrame[]; video_id: string }> => {
    const res = await client.get(`/videos/${id}/frames`)
    return res.data
  },

  getTranscript: async (id: string): Promise<{ segments: Array<{ timestamp: number; text: string }>; count: number }> => {
    const res = await client.get(`/videos/${id}/transcript`)
    return res.data
  },

  modelStatus: async (): Promise<{ services: Array<{ name: string; url: string; ok: boolean; models: string[] }> }> => {
    const res = await client.get('/status/models')
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