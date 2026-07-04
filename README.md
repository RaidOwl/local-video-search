# Local Video Search & Summarization

A self-hosted video search and summarization system powered by local AI models. Index your video library, search in natural language, get AI-generated summaries, and browse every moment with a visual segments timeline.

## Features

- **Natural Language Search** — Ask questions like "Find the moment where the car crashes" or "Show me cooking scenes" ([how it works under the hood](docs/QUERY-MECHANICS.md))
- **Segment-Level Results** — Results point to the exact timestamp within a video, not just which file
- **Hybrid Retrieval + Precision Reranking** — Every search runs both a meaning (semantic) and a keyword (BM25) recall pass, then a local cross-encoder reranks the results so they respect fine details (e.g. "blonde woman" vs "blonde man"), with a relevance slider to tune strictness
- **Cross-Modal Search** — When searching both sources, the system correlates what's shown and what's spoken in the same moment, and a fused per-moment description surfaces matches that neither track alone would catch; spoken-only hits borrow the nearest frame for visual context
- **Compositional Precision** — Multi-word searches are checked by a local LLM so a query's attributes must describe the *same* object (e.g. "mini pc with wooden accent" won't match a separate mini PC and a separate wooden desk)
- **Scene-Aware Frame Sampling** — Frames are sampled on a time grid *and* at detected scene cuts (with optional GPU-accelerated decode for faster analysis), so brief subjects between grid points still get captioned
- **Metadata Filters** — Optionally narrow any search by media type, minimum resolution, or creation date — or apply the filters alone to browse your library by those criteria
- **Visual Frame Analysis** — Vision-language model (VLM) captions every frame during analysis
- **Speech Transcription** — Whisper transcribes spoken audio; transcript segments are fully searchable and show surrounding spoken context
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
| `ollama` (low / 4B) | ≥ 8 GB | CPU also works (slow) |
| `ollama` (mid / 8B) | ≥ 10 GB | Recommended starting point |
| `ollama` (high / 32B) | ≥ 20 GB | Q4 quantized |
| `vllm` (mid / 8B FP8) | ≥ 16 GB | NVIDIA Hopper / Ada / Blackwell |
| `vllm` (high / 32B FP8) | ≥ 40 GB | e.g. DGX Spark GB10 |
| `vllm` (ultra / 235B FP8) | ≥ 128 GB | Multi-GPU server |
| `vllm` (AMD ROCm) | ≥ 16 GB | AMD GPU with ROCm (e.g. Strix Halo) |
| `custom` | varies | Point at any OpenAI-compatible server |

Transcription (Whisper) runs on CPU inside the API container — no extra GPU needed.

Video decode (FFmpeg scene detection + frame extraction) can optionally use GPU acceleration when configured — see [FFmpeg Hardware Acceleration](#ffmpeg-hardware-acceleration).

## Quick Start

### 🚀 Smart Setup (Recommended)

The easiest way to get started — auto-detects your hardware (NVIDIA/AMD/Apple/CPU), configures `.env`, validates prerequisites, and starts the stack:

```bash
git clone https://github.com/RaidOwl/local-video-search.git
cd local-video-search
./setup.sh          # Linux/macOS
# or
./setup.ps1         # Windows PowerShell
```

Then open **http://localhost:3000** — done!

The script handles everything:
- Detects GPU type and VRAM → sets optimal `VRAM_TIER`
- Validates Docker Compose version for AMD ROCm (`>=2.24.0`)
- Checks `nvidia-container-toolkit` for NVIDIA
- Verifies `ollama serve` is running on Apple Silicon
- Auto-pulls Ollama models on Mac
- Writes `.env` with detected config (backs up existing)
- Starts the stack with `docker compose up -d`

---

### Step 1 — Clone the repo (Manual)

If you prefer manual setup:

```bash
git clone https://github.com/RaidOwl/local-video-search.git
cd local-video-search
cp .env.example .env
```

### Step 2 — Pick your AI backend

The system needs three AI models: an **embedding model** (converts text/frames to vectors for search), a **vision model** (captions video frames), and a **text LLM** (generates summaries). How you run them determines which backend to use. Set your choice in `.env`.

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
HARDWARE_TYPE=apple
VRAM_TIER=mid         # or low / high depending on your unified memory
```

Because Ollama runs on your host, **you pull the models yourself** — unlike the dockerized Ollama backend, `ollama-native` has no auto-pull. Grab the embedding model plus the vision/LLM model for your tier (see the [model size tiers](#model-size-tiers) below):

```bash
ollama pull bge-m3               # embeddings (all tiers)
ollama pull qwen3-vl:8b          # vision + summaries — mid/qwen (4b/8b/32b/235b per tier)
```

Start the stack normally — no extra profile needed:
```bash
docker compose up -d
```

**Linux with AMD GPU**

vLLM runs inside Docker with AMD GPU (ROCm) device passthrough. The `docker-compose.rocm.yml` override file swaps the NVIDIA deploy block for AMD device passthrough (`/dev/kfd`, `/dev/dri`) and sets the ROCm vLLM image. Requires **Docker Compose >= 2.24.0** (the override uses the `!reset` merge tag to clear the NVIDIA device block — run `docker compose version` to check if device passthrough isn't working):

```env
HARDWARE_TYPE=amd
VRAM_TIER=mid
COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml
```

```bash
docker compose up -d
```

> **Note:** FP8 quantization requires MI300+ (CDNA). RDNA iGPUs like the Strix Halo (gfx1151) must run unquantized — the ROCm override already defaults `VLLM_QUANTIZATION` to empty (BF16) for you, so no extra step is needed. The override also sets `HSA_OVERRIDE_GFX_VERSION=11.5.1` automatically.

**CPU-only / no GPU**

Set `VRAM_TIER=low` and `HARDWARE_TYPE=cpu`. The 4B model will run on CPU — slow but functional.

---

#### vLLM (NVIDIA or AMD, higher quality)

vLLM runs models in FP8 rather than Ollama's Q4 quantization, giving better output quality at the cost of needing more VRAM and a modern GPU (NVIDIA Hopper H100, Ada RTX 40xx, Blackwell GB10/B200; or AMD with ROCm).

**NVIDIA:**
```env
HARDWARE_TYPE=nvidia
VRAM_TIER=mid              # 8B FP8 needs ~16 GB VRAM
```

**AMD (ROCm):**
```env
HARDWARE_TYPE=amd
VRAM_TIER=mid
COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml
```

```bash
docker compose up -d
```

> **Note:** The base compose uses the official NVIDIA vLLM image. For AMD, the `docker-compose.rocm.yml` override swaps in the ROCm build with device passthrough (requires Docker Compose >= 2.24.0 for the `!reset` merge tag) and defaults quantization to unquantized, since FP8 requires MI300+ (CDNA) and RDNA iGPUs don't support it. For Apple Silicon, use `HARDWARE_TYPE=apple` instead.

---

#### Custom (bring your own servers)

Use this if you already have Ollama, vLLM, LiteLLM, or any OpenAI-compatible server running. Point each URL at your existing endpoints:

```env
HARDWARE_TYPE=custom

# Embedding model (OpenAI /v1/embeddings compatible)
EMBED_URL=http://10.0.0.30:8001
EMBED_MODEL=BAAI/bge-m3

# Vision / captioning model (supports image_url in messages)
VLM_URL=http://10.0.0.19:8000
VLM_MODEL=llava-hf/llava-v1.6-vicuna-7b-hf

# Text LLM for summaries
LLM_URL=http://10.0.0.30:8000
LLM_MODEL=Qwen/Qwen3-8B
```

```bash
docker compose up -d
```

---

#### Model size tiers

`VRAM_TIER` controls which model size is selected for local backends (`ollama`, `ollama-native`, `vllm`). `MODEL_FAMILY` selects the model family (`qwen` = Qwen3-VL, `gemma` = Gemma 3).

| Tier | Ollama (Q4) | vLLM (FP8) | Approx. VRAM |
|------|-------------|------------|--------------|
| `low` | 4B | 4B | ≥ 8 GB |
| `mid` | 8B | 8B | ≥ 10–16 GB |
| `high` | 32B | 32B | ≥ 20–40 GB |
| `ultra` | 235B | 235B | ≥ 128 GB |

> vLLM FP8 quantizes only the LLM decoder; the vision encoder stays BF16 (~8 GB extra). On Apple Silicon, VRAM is unified memory shared with the system — leave headroom.
>
> By default vLLM's memory reservation auto-scales to `VRAM_TIER`, so a smaller model won't reserve your entire GPU (handy on large unified-memory boxes). Override it with `VLLM_GPU_MEMORY_UTIL` in `.env`.

---

#### Set your video directories

After choosing a backend above, point `.env` at your videos:

```env
VIDEO_DIR_1=/path/to/your/videos
# VIDEO_DIR_2=/another/path   # up to 5 directories supported
```

See `.env.example` for all options including port overrides and frame extraction settings.

#### Videos on a NAS / network share

If your videos live on a network share (SMB/NFS), **don't** point `VIDEO_DIR_*`
at a host mount like `/Volumes/NAS` when using **Docker Desktop** (macOS/Windows).
Docker Desktop's file sharing lets host-mounted network shares go stale inside
the container (`Not a directory` / `Bad file descriptor`), so imports fail
intermittently.

Instead, mount the share into the container directly. Create a
`docker-compose.override.yml` next to `docker-compose.yml` (it's auto-loaded and
gitignored) — for an SMB/CIFS share:

```yaml
services:
  local-vss-api:
    volumes:
      - videos_lib1:/data/videos/lib1:ro

volumes:
  videos_lib1:
    driver: local
    driver_opts:
      type: cifs
      o: "addr=NAS_IP,username=${SMB_USER},password=${SMB_PASS},ro,uid=0,gid=0,file_mode=0444,dir_mode=0555,vers=3.0"
      device: "//NAS_IP/Share/Path/To/Videos"
```

Put `SMB_USER` / `SMB_PASS` in `.env`, replace `NAS_IP` and the share path, then
`docker compose up -d`. The container's own kernel mounts the share, bypassing
the flaky host bind. (On native Linux Docker, host bind mounts of NFS work fine
and this isn't needed.)

### Step 3 — Start services

```bash
docker compose up -d
```

That's it for every backend. `HARDWARE_TYPE` is the only switch you set — Docker
starts the matching model server automatically (or none, for `custom` and
`apple`).

### Step 4 — Open the UI

Navigate to **http://localhost:3000**

Go to **Settings** to verify your AI backend — the banner now shows detected hardware (GPU model, VRAM, ROCm device status), resolved model names, and service connectivity. Configure Whisper transcription quality (Low / Medium / High), set the frame extraction interval, cap the CPU threads FFmpeg uses during analysis (defaults to half your cores so the host stays responsive), tune max concurrent jobs, toggle auto-analyze on import, and choose a hardware acceleration backend for FFmpeg video decoding.

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

Results link directly to the timestamp in the video. Use the **Min relevance** slider to control how strict matches must be — higher values return only near-exact matches, lower values cast a wider net. Every search is reranked by a local cross-encoder for precision, so results respect fine details rather than just keyword overlap.

> Need to narrow results by file type, resolution, or creation date? Open the **Filters** popup beside the search bar. Filters apply as an exact match on top of your query — or apply them with an empty search box to browse your whole library by those criteria.

> On first launch the reranker model (`BAAI/bge-reranker-base`, ~1.1 GB) downloads automatically and is cached under `./data` so it persists across restarts. It runs on CPU inside the API container — no extra GPU required.

## Hardware Types

| Value | AI Backend | Description |
|-------|------------|-------------|
| `nvidia` | vllm | vLLM runs in Docker. Requires NVIDIA GPU (Hopper/Ada/Blackwell) + nvidia-container-toolkit. |
| `amd` | vllm | vLLM runs in Docker with AMD ROCm device passthrough. Set `COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml` in `.env`. |
| `apple` | ollama-native | Ollama runs on your host machine. **Required for Apple Silicon** — Docker VMs can't access Metal. Run `ollama serve` first. |
| `cpu` | ollama | Ollama runs in Docker on CPU only. |
| `custom` | custom | Bring your own servers. Set `EMBED_URL`, `VLM_URL`, `LLM_URL` in `.env` |

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
| **Search** | `GET /search`, `GET /search/segments`, `GET /metadata/facets` |
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
| `HARDWARE_TYPE` | Hardware platform selection (nvidia/amd/apple/cpu/custom) |
| `VRAM_TIER` | Model size for local backends (low/mid/high/ultra) |
| `VIDEO_DIR_1..5` | Up to 5 video library directories |
| `FFMPEG_HWACCEL` | FFmpeg video decode acceleration: `off`, `auto`, `cuda`, `vaapi`, `drm`, or `vulkan` (default: `off`) |

## FFmpeg Hardware Acceleration

Video decoding is the most CPU-intensive step during analysis — every video must be fully decoded to detect scene cuts, then decoded again at each extraction timestamp. Hardware acceleration offloads this decode step to your GPU, dramatically speeding up analysis on long videos.

### Supported backends

| Backend | Platforms | Notes |
|---------|-----------|-------|
| `cuda` | NVIDIA GPUs (DGX Spark / GB10) | Uses NVDEC (`h264_cuvid`, `hevc_cuvid`, `av1_cuvid`). Requires `nvidia-container-toolkit` and `--gpus` on the API container. |
| `vaapi` | AMD / Intel GPUs on Linux | Works with AMD Radeon (RDNA) and ROCm. Requires `mesa-va-drivers` in container (included in the default Docker image) and `/dev/dri` device passthrough on the API container. |
| `drm` | Linux (kernel DRM) | Lightweight kernel-level decode. Available on most Linux systems with GPU drivers. |
| `vulkan` | Cross-platform | GPU decode via Vulkan video APIs. |

### How to enable

1. **Set the mode** in `.env` or via the **Settings UI** → **Processing** → **FFmpeg hardware acceleration**:
   - `auto` — auto-detects the best available backend (recommended)
   - `cuda` / `vaapi` / `drm` / `vulkan` — pin a specific backend
   - `off` — software decode only (default)

2. **For containers, ensure GPU access** — the API container needs to see the GPU hardware:

   **NVIDIA (spark3 / any DGX Spark / GB10):** Add GPU device access to the `local-vss-api` service in `docker-compose.override.yml`:
   ```yaml
   services:
     local-vss-api:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

   **AMD ROCm (Strix Halo):** The `docker-compose.rocm.yml` override already passes `/dev/kfd` and `/dev/dri` to the vLLM containers. Add the same to `local-vss-api` in a `docker-compose.override.yml`:
   ```yaml
   services:
     local-vss-api:
       devices:
         - /dev/kfd:/dev/kfd
         - /dev/dri:/dev/dri
       security_opt:
         - seccomp:unconfined
   ```

3. **Verify** — open the **Settings** UI. With `auto`, it shows the resolved backend (e.g. `cuda` or `vaapi`). If auto-detect can't find a GPU decoder, it gracefully falls back to CPU.

> **Note:** The scene detection filter (`select='gt(scene,...)'`) runs on CPU memory even when the decode is GPU-accelerated. Frames are decoded on the GPU then transferred to CPU for analysis — you still get the speedup from GPU decode, just not the full GPU-native pipeline (which scene detection can't use since it needs per-frame pixel comparison).

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
                                        faster-whisper (transcription, CPU)
                                        bge-reranker   (search reranking, CPU)
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
│   ├── transcribe/    # faster-whisper speech-to-text
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
