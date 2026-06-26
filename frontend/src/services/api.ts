import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

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

  search: async (query: string, limit?: number): Promise<{ results: SearchResult[]; query: string }> => {
    const res = await client.get('/search', { params: { q: query, limit } })
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
}