d# Local Video Search & Summarization

A self-hosted video search and summarization system powered by local AI models. Index your video library, search in natural language, get AI-generated summaries, and browse every moment with a visual segments timeline.

## Features

- **Natural Language Search** — Ask questions like "Find the moment where the car crashes" or "Show me cooking scenes"
- **Segment-Level Results** — Results point to the exact timestamp within a video, not just which file
- **Visual Frame Analysis** — Vision-language model (VLM) captions every frame during analysis
- **Speech Transcription** — Whisper transcribes spoken audio; transcript segments are fully searchable
- **AI Summarization** — On-demand video summaries powered by a local LLM
- **Segments Timeline** — Browse all visual and speech segments side-by-side with inline caption editing
- **Download & Clip** — Download the full video or extract a time-range clip directly from the UI
- **Multiple AI Backends** — Bring your own vLLM / Ollama servers, or let Docker spin them up
- **Self-Hosted** — No cloud, no API keys, no data leaves your machine
- **REST API + Swagger UI** — Full documented API at `/docs` for building your own integrations

## Hardware Requirements

Requirements depend on the AI backend you choose:

| Backend | GPU VRAM | Notes |
|---------|----------|-------|
| `ollama` (low / 3B) | ≥ 8 GB | CPU also works (slow) |
| `ollama` (mid / 7B) | ≥ 10 GB | Recommended starting point |
| `ollama` (high / 32B) | ≥ 20 GB | Q4 quantized |
| `vllm` (mid / 7B FP8) | ≥ 16 GB | NVIDIA Hopper / Ada / Blackwell |
| `vllm` (high / 32B FP8) | ≥ 40 GB | e.g. DGX Spark GB10 |
| `vllm` (ultra / 72B FP8) | ≥ 128 GB | Multi-GPU server |
| `custom` | varies | Point at any OpenAI-compatible server |

Transcription (Whisper) runs on CPU inside the API container — no extra GPU needed.

## Quick Start

### Step 1 — Pick your AI backend

The system needs three AI models: an **embedding model** (converts text/frames to vectors for search), a **vision model** (captions video frames), and a **text LLM** (generates summaries). How you run them determines which backend to use.

---

#### Ollama (recommended for most users)

Ollama is the easiest way to get started. It downloads and runs quantized models automatically.

**Apple Silicon Mac (M1/M2/M3/M4)**

Docker on Mac runs inside a Linux VM which cuts off GPU access. Ollama must run on the host to use Metal acceleration:

```bash
brew install ollama
ollama serve          # keep this running in the background
```

Then set in `.env`:
```env
AI_BACKEND=ollama-native
VRAM_TIER=mid         # or low / high depending on your unified memory
```

Start the stack normally — no extra profile needed:
```bash
docker compose up -d
```

**Linux with NVIDIA GPU**

Ollama can run inside Docker with full GPU passthrough:

```env
AI_BACKEND=ollama
VRAM_TIER=mid
```

```bash
docker compose --profile ollama up -d
```

**CPU-only / no GPU**

Set `VRAM_TIER=low` and `AI_BACKEND=ollama`. The 3B model will run on CPU — slow but functional.

---

#### vLLM (NVIDIA only, higher quality)

vLLM runs models in FP8 rather than Ollama's Q4 quantization, giving better output quality at the cost of needing more VRAM and a modern NVIDIA GPU (Hopper H100, Ada RTX 40xx, or Blackwell GB10/B200).

```env
AI_BACKEND=vllm
VRAM_TIER=mid         # 7B FP8 needs ~16 GB VRAM
```

```bash
docker compose --profile vllm up -d
```

> **Note:** vLLM does not support Apple Silicon or AMD GPUs.

---

#### Custom (bring your own servers)

Use this if you already have Ollama, vLLM, LiteLLM, or any OpenAI-compatible server running. Point each URL at your existing endpoints:

```env
AI_BACKEND=custom

# Embedding model (OpenAI /v1/embeddings compatible)
VLLM_EMBED_URL=http://10.0.0.30:8001
VLLM_EMBED_MODEL=BAAI/bge-m3

# Vision / captioning model (supports image_url in messages)
VLLM_VLM_URL=http://10.0.0.19:8000
VLLM_VLM_MODEL=llava-hf/llava-v1.6-vicuna-7b-hf

# Text LLM for summaries
VLLM_URL=http://10.0.0.30:8000
VLLM_MODEL=Qwen/Qwen3-8B
```

```bash
docker compose up -d   # no profile needed
```

---

#### Model size tiers

`VRAM_TIER` controls which model size is selected for local backends (`ollama`, `ollama-native`, `vllm`). `MODEL_FAMILY` selects the model family (`qwen` = Qwen2.5-VL, `gemma` = Gemma 3).

| Tier | Ollama (Q4) | vLLM (FP8) | Approx. VRAM |
|------|-------------|------------|--------------|
| `low` | 3B | 3B | ≥ 8 GB |
| `mid` | 7B | 7B | ≥ 10–16 GB |
| `high` | 32B | 32B | ≥ 20–40 GB |
| `ultra` | 72B | 72B | ≥ 128 GB |

> vLLM FP8 quantizes only the LLM decoder; the vision encoder stays BF16 (~8 GB extra). On Apple Silicon, VRAM is unified memory shared with the system — leave headroom.

---

### Step 2 — Clone & configure

```bash
git clone https://github.com/RaidOwl/local-video-search.git
cd local-video-search
cp .env.example .env
```

Edit `.env` based on your chosen backend (see above), then set your video directory:

```env
VIDEO_DIR_1=/path/to/your/videos
# VIDEO_DIR_2=/another/path   # up to 5 directories supported
```

See `.env.example` for all options including port overrides and frame extraction settings.

### Step 3 — Start services

```bash
# ollama-native or custom
docker compose up -d

# ollama in Docker
docker compose --profile ollama up -d

# vllm in Docker
docker compose --profile vllm up -d
```

### Step 4 — Open the UI

Navigate to **http://localhost:3000**

Go to **Settings** to verify your AI backend, configure Whisper transcription quality (Low / Medium / High), set the frame extraction interval, and toggle auto-analyze on import.

### Step 5 — Add & analyze videos

1. **Videos** → **Import** — scans your configured directories and adds any new files
2. Select videos → **Analyze** — extracts frames, captions them via VLM, transcribes audio via Whisper
3. Once status shows **Analyzed**, search or browse segments immediately

### Step 6 — Search

```
cooking scenes with garlic
when did they mention the deadline
car driving through a tunnel at night
person writing on a whiteboard
```

Results link directly to the timestamp in the video.

## AI Backends

| Value | Description |
|-------|-------------|
| `custom` | Bring your own servers. Set `VLLM_EMBED_URL`, `VLLM_VLM_URL`, `VLLM_URL` in `.env` |
| `ollama` | Ollama runs in Docker. Good for NVIDIA Linux or CPU. |
| `ollama-native` | Ollama runs on your host machine. **Required for Apple Silicon** — Docker VMs can't access Metal. Run `ollama serve` first. |
| `vllm` | vLLM runs in Docker. Requires NVIDIA GPU (Hopper/Ada/Blackwell) + nvidia-container-toolkit. |

## API

The full interactive API is available at **http://localhost:3000/docs** (Swagger UI) or `/redoc` (ReDoc).

A server dropdown lets you switch between the proxied frontend URL (`/api`) and direct API access (`:8000`).

### Core workflow

```http
POST   /ingest                          # Add a video file to the library
POST   /analyze/{video_id}              # Start full vision + speech analysis
GET    /videos/{video_id}               # Poll status: ready → processing → analyzed
GET    /search?q=your+query             # Semantic search across all analyzed videos
GET    /search/segments?q=...&video_id= # Search within a specific video
```

### Key endpoints by group

| Group | Endpoints |
|-------|-----------|
| **Search** | `GET /search`, `GET /search/segments` |
| **Videos** | `GET /videos`, `GET /videos/{id}`, `PATCH /videos/{id}`, `DELETE /videos/{id}` |
| **Analysis** | `POST /ingest`, `POST /analyze/{id}`, `POST /retranscribe/{id}`, `POST /cancel/{id}`, `POST /summarize/{id}` |
| **Media** | `GET /videos/{id}/stream`, `GET /videos/{id}/download`, `GET /videos/{id}/clip?start=&end=`, `GET /videos/{id}/thumbnail` |
| **System** | `GET /health`, `GET /settings`, `PUT /settings`, `GET /status/models`, `GET /browse` |

Video status lifecycle: `ready` → `processing` → `analyzed` (or `error`). Audio-only re-transcription uses `retranscribing` status.

## Configuration

All runtime settings can be managed in the **Settings** page of the UI, or via `.env` / API:

| Setting | Description |
|---------|-------------|
| `FRAME_EXTRACTION_INTERVAL` | Seconds between extracted frames (default: 5) |
| `EMBED_BATCH_SIZE` | Captions embedded per batch (default: 16) |
| `AI_BACKEND` | AI provider selection (see above) |
| `VRAM_TIER` | Model size for local backends (low/mid/high/ultra) |
| `VIDEO_DIR_1..5` | Up to 5 video library directories |

## Architecture

```
Browser (port 3000)
    │
    ▼
 nginx (frontend)
    ├── / ──────────────────▶ React SPA
    └── /api/ ──────────────▶ FastAPI (port 8000)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                 SQLite        Qdrant        AI Models
              (metadata)     (vectors)   (VLM / Embed / LLM)
                                              │
                                        faster-whisper
                                        (transcription, CPU)
```

## Project Structure

```
local-video-search/
├── src/
│   ├── api/           # FastAPI server — all endpoints
│   ├── embed/         # Embedding generation + Qdrant indexing
│   ├── ingest/        # Video scanning and frame extraction (ffmpeg)
│   ├── search/        # Query processing and segment retrieval
│   ├── summarize/     # LLM summarization
│   ├── transcription/ # faster-whisper speech-to-text
│   ├── config.py      # Environment + settings management
│   └── database.py    # SQLAlchemy models (Video, Frame)
├── frontend/
│   ├── src/
│   │   ├── components/ # NavBar, shared UI
│   │   ├── pages/      # SearchPage, VideosPage, VideoDetailPage, SettingsPage
│   │   └── services/   # Typed API client (api.ts)
│   ├── nginx.conf      # Reverse proxy config
│   └── Dockerfile
├── docker/
│   └── app/Dockerfile  # Python API container
├── docker-compose.yml  # Full stack orchestration
├── .env.example        # All configuration options with documentation
└── requirements.txt    # Python dependencies
```

## License

MIT
