# Local Video Search & Summarization

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
| `vllm` (ultra / 32B FP8) | ≥ 80 GB | e.g. DGX Spark GB10 |
| `custom` | varies | Point at any OpenAI-compatible server |

Transcription (Whisper) runs on CPU inside the API container — no extra GPU needed.

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/RaidOwl/local-video-search.git
cd local-video-search
cp .env.example .env
```

Edit `.env` — the key settings:

```env
# Choose your AI backend: custom | ollama | ollama-native | vllm
AI_BACKEND=ollama

# Model size for local backends (low / mid / high / ultra)
VRAM_TIER=mid

# One or more video library directories (read-only mounts)
VIDEO_DIR_1=/path/to/your/videos
```

See `.env.example` for all options including custom model endpoints and port overrides.

### 2. Start services

```bash
# Default (custom backend or ollama-native)
docker compose up -d

# Ollama in Docker (CPU or NVIDIA Linux)
docker compose --profile ollama up -d

# vLLM in Docker (NVIDIA GPU required)
docker compose --profile vllm up -d
```

### 3. Open the UI

Navigate to **http://localhost:3000**

On first launch, go to **Settings** to verify your AI backend, configure Whisper quality (Low / Medium / High), set the frame extraction interval, and toggle auto-analyze.

### 4. Add & analyze videos

1. Go to **Videos** → click **Import** to scan your configured video directories
2. Select videos and click **Analyze** — the system extracts frames, captions them via VLM, and transcribes audio via Whisper
3. Once status shows **Analyzed**, search or browse segments immediately

### 5. Search

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
| `ollama-native` | Ollama runs on your host machine. **Best for Apple Silicon** — full Metal acceleration. Run `ollama serve` first. |
| `vllm` | vLLM runs in Docker. Requires NVIDIA GPU + nvidia-container-toolkit. |

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


## Hardware Requirements

- **Embedding**: NVIDIA GPU with ~8GB VRAM (for Cosmos-Embed1 via Ollama)
- **VLM**: NVIDIA GPU with ~8GB VRAM (for LLaVA via Ollama)
- **LLM Summarization**: NVIDIA GPU with ~80GB VRAM (for vLLM, e.g., Llama-3.3-70B)
- **Storage**: SSD recommended for video processing

## Software Requirements

- Docker & Docker Compose
- Ollama (for embedding and VLM models)
- vLLM (for summarization LLM)
- Qdrant (vector database - included in docker-compose)

## Quick Start

### 1. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

Key settings in `.env`:
- `OLLAMA_EMBED_URL` - Ollama endpoint for Cosmos-Embed1
- `OLLAMA_VLM_URL` - Ollama endpoint for LLaVA
- `VLLM_URL` - vLLM endpoint for summarization
- `QDRANT_URL` - Qdrant vector database URL
- `VIDEO_DIR_PATH` - Path to your video library

### 2. Start Services

```bash
# Build and start all services
docker compose up -d

# Check status
docker compose ps
```

### 3. Install AI Models

**Ollama Models:**
```bash
# Pull embedding model
ollama pull cosmos-embed1

# Pull VLM for captioning
ollama pull llava:7b

# Verify
ollama list
```

**vLLM Model:**
```bash
# On your vLLM server (e.g., Spark cluster)
vllm serve meta-llama/Llama-3.3-70B-Instruct
```

### 4. Index Videos

Open http://localhost:3000 and:
1. Navigate to the **Videos** page
2. Enter the path to your video directory (or use the volume-mounted path)
3. Click **Index Videos**

Processing happens in the background. Videos will appear as frames are indexed.

### 5. Search

Use natural language queries:
- "Find the moment where the car crashes"
- "Show me cooking scenes"
- "When did they discuss the project"
- "Find scenes with the ocean"

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI   │────▶│   Qdrant    │
│   (React)   │     │   Server   │     │  (Vectors)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   ▲
                           ▼                   │
                    ┌─────────────┐     ┌─────────────┐
                    │  Ollama     │────▶│   Frames    │
                    │  (VLM+Emb)  │     │  (Extracted)│
                    └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   vLLM      │
                    │ (Summarize) │
                    └─────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Index a video or directory |
| GET | `/search?q=<query>` | Search videos by natural language |
| GET | `/search/segments?video_id=<id>&q=<query>` | Find specific moments within a video |
| GET | `/videos` | List all indexed videos |
| GET | `/videos/{id}` | Get video details |
| GET | `/videos/{id}/summary` | Get video summary |
| POST | `/summarize/{id}` | Generate/regenerate summary |

## Configuration

### Frame Extraction

- `FRAME_EXTRACTION_INTERVAL` - Seconds between extracted frames (default: 5)
- `EMBED_BATCH_SIZE` - Number of frames to embed at once (default: 16)

### Ollama

Ensure Ollama is running with appropriate models:

```bash
# Start Ollama
ollama serve

# List models
ollama list
```

### vLLM

vLLM should be running on a remote server (e.g., your DGX Spark cluster):

```bash
# On remote server
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

## Development

### Run Backend Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python -m src.main
```

### Run Frontend Locally

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
local-video-search/
├── src/
│   ├── api/           # FastAPI server and routes
│   ├── embed/         # Ollama + Qdrant embedding
│   ├── ingest/        # Video scanning and frame extraction
│   ├── search/        # Query processing and segment retrieval
│   ├── summarize/     # vLLM summarization
│   ├── config.py      # Environment configuration
│   └── database.py   # SQLAlchemy models
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/        # Page components
│   │   └── services/     # API client
│   └── Dockerfile
├── docker/
│   └── app/Dockerfile   # Python API container
├── docker-compose.yml   # Full stack orchestration
├── requirements.txt     # Python dependencies
└── .env.example        # Environment template
```

## License

MIT