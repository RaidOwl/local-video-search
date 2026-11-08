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

The default local backend is **Ollama** (llama.cpp, Q4 quantization). The intended target is a 16 GB consumer GPU, but that configuration has not yet been verified on a discrete GPU:

| Platform | `HARDWARE_TYPE` | Backend | GPU memory |
|----------|-----------------|---------|------------|
| NVIDIA GPU, Linux or Windows (WSL2), incl. DGX Spark | `nvidia` | Ollama in Docker (CUDA) | **12 GB+** (mid) · 8 GB+ (low) |
| AMD GPU, Linux x86-64 (RDNA 2+, Strix Halo, Instinct) | `amd` | Ollama in Docker (ROCm) | **12 GB+** (mid) · 8 GB+ (low) |
| Apple Silicon | `apple` | Ollama on the host (Metal) | 16 GB+ unified |
| Intel Arc, AMD on Windows or Linux ARM64, other GPUs | `native` | Ollama on the host (Vulkan/ROCm/CUDA) | 12 GB+ (mid) · 8 GB+ (low) |
| No GPU | `cpu` | Ollama in Docker | — (slow) |
| Your own servers | `custom` | any OpenAI-compatible API | — |
| *Opt-in:* NVIDIA, 32 GB+ recommended | `nvidia-vllm` | vLLM in Docker, FP8 | see [vLLM](#vllm-opt-in-nvidia-or-amd) |
| *Opt-in:* AMD RDNA 3+ / MI300+, 32 GB+ recommended | `amd-vllm` | vLLM on ROCm | see [vLLM](#vllm-opt-in-nvidia-or-amd) |

On one AMD Strix Halo unified-memory system, both models together used about **11 GB** for the mid tier (Qwen3-VL-8B) and **9 GB** for the low tier (Qwen3-VL-4B). On a 24 GB NVIDIA RTX PRO 4000 SFF, the mid-tier models both ran fully on GPU and a post-search snapshot showed **12,143 MiB** in use. Both machines successfully indexed and searched a short video; this is a smoke test, not a broad quality benchmark or a peak-memory measurement. Actual memory use and CPU offloading vary by platform; 8 GB and 16 GB discrete GPUs have not yet been tested. `./setup.sh` / `./setup.ps1` select a [model tier](#model-size-tiers) based on detected memory.

> **Disk space:** The API image is ~0.7 GB to download, the Ollama image ~2 GB (~4 GB for ROCm), and the models ~9 GB. The opt-in vLLM images are much larger (~10 GB NVIDIA, ~31 GB ROCm) and its models ~26 GB. Use `MODELS_DIR` in `.env` to relocate model downloads to a larger volume, and point `QDRANT_STORAGE_PATH` at a data drive for vector collections.

Transcription (Whisper) runs on CPU with INT8 inside the API container by default — no extra GPU needed. The default model is `medium`; set `WHISPER_MODEL` in `.env` (or the Whisper model buttons in **Settings**) to trade accuracy against speed. GPU transcription requires a compatible custom API runtime and GPU passthrough.

Video decode (FFmpeg scene detection + frame extraction) can optionally use GPU acceleration when configured — see [FFmpeg Hardware Acceleration](#ffmpeg-hardware-acceleration).

## Quick Start

### 🚀 Smart Setup (Recommended)

The setup wizard detects your system, suggests a model-memory budget, then lets you confirm the platform, backend, model family/tier, video library, and model storage folder (`MODELS_DIR`, default `./data/models`) before changing `.env` or starting containers. It warns when the chosen drive has too little free space, and it does not move models already downloaded elsewhere. With host Ollama (macOS, or `native` mode), Ollama keeps its own models in its own storage (`OLLAMA_MODELS`); the folder only holds the reranker and speech models. External endpoints are checked for the selected model IDs before setup continues.

```bash
git clone https://github.com/RaidOwl/local-video-search.git
cd local-video-search
./setup.sh          # Linux/macOS          (add --non-interactive to accept detected defaults)
# or
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # Windows (add -NonInteractive to accept defaults)
```

For an advanced local vLLM deployment, choose it in the wizard or pass `--vllm` (`-Vllm` on Windows) to preselect it.

The wizard preserves existing settings unless you choose a new value. For external servers, it currently supports unauthenticated OpenAI-compatible endpoints; API-key authentication is not yet supported. After starting, setup waits for the API and all three selected model endpoints to report ready, then prints the local URL. Model downloads may take a while; failures include service status and relevant logs.

Ollama downloads the selected models on first start (~9 GB for the mid tier); setup reports progress through the service logs if readiness times out.

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

#### Ollama (default)

Ollama is the default local backend. In a limited Strix Halo comparison, its Q4 embedder produced similar retrieval results to vLLM and used less memory; this is not a guarantee of equivalent quality on every library or platform.

**Linux or Windows with an NVIDIA or AMD GPU**

Ollama runs in Docker and downloads its models on first start:

```env
HARDWARE_TYPE=nvidia       # or amd
VRAM_TIER=mid              # low for GPUs under 12 GB
# AMD only — adds GPU video decode to the API container (Docker Compose >= 2.24.0):
# COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml
```

NVIDIA needs `nvidia-container-toolkit` on Linux (built into Docker Desktop's WSL2 backend on Windows). AMD needs x86-64 Linux with ROCm-capable drivers (RDNA 2 or newer, Strix Halo, Instinct); Ollama's ROCm image detects the GPU itself. Its pinned ROCm image is not available for ARM64, so AMD Linux ARM64 systems need host Ollama (`HARDWARE_TYPE=native`). `OLLAMA_CONTEXT_LENGTH` (default 8192) and `OLLAMA_NUM_PARALLEL` (default 1) in `.env` tune its memory use — raising either increases it.

**Apple Silicon Mac (M1/M2/M3/M4)**

Docker on Mac runs inside a Linux VM which cuts off GPU access. Ollama must run on the host to use Metal acceleration:

```bash
brew install ollama
ollama serve          # keep this running in the background
```

Then set in `.env`:
```env
HARDWARE_TYPE=apple
VRAM_TIER=mid         # low on 8 GB Macs
```

Because Ollama runs on your host, **you pull the models yourself** — unlike the dockerized Ollama backend, `ollama-native` has no auto-pull. Grab the embedding model plus the vision/LLM model for your tier (see the [model size tiers](#model-size-tiers) below):

```bash
ollama pull qwen3-embedding:4b   # embeddings (all tiers)
ollama pull qwen3-vl:8b-instruct # vision + summaries — mid tier (qwen3-vl:4b-instruct for low)
```

Use the `-instruct` tags: the plain `qwen3-vl:8b` / `:4b` tags are "thinking" models that return empty captions.

**Keep models loaded in memory** — by default Ollama unloads a model ~5 minutes
after the last request. During video analysis, the VLM is called hundreds of
times per video, so a short keep_alive causes constant reloads (each one tens
of seconds). For the best experience, keep models loaded indefinitely (same
behavior as vLLM). Start Ollama with:

```bash
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

Or set it globally via the systemd override:

```bash
printf '[Service]\nEnvironment="OLLAMA_KEEP_ALIVE=-1"\n' | sudo tee /etc/systemd/system/ollama.service.d/keep_alive.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

If you share your Mac's GPU with other tools and need VRAM freed when not
using LVS, use a finite duration instead (e.g., `OLLAMA_KEEP_ALIVE=30m`).

Start the stack normally — no extra profile needed:
```bash
docker compose up -d
```

**Linux with Intel Arc (or any GPU Docker can't pass through)**

Run [Ollama](https://ollama.com/download) on the host — it drives Intel Arc through Vulkan — and point the containers at it:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0:11434"\nEnvironment="OLLAMA_VULKAN=1"\n' | sudo tee /etc/systemd/system/ollama.service.d/lvs.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

```env
HARDWARE_TYPE=native
VRAM_TIER=mid
```

`OLLAMA_HOST=0.0.0.0` is required on Linux: the containers reach Ollama over the Docker bridge, not loopback. `setup.sh` detects a running host Ollama, uses it automatically, and pulls the models.

**Windows**

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL2 backend. Run `powershell -ExecutionPolicy Bypass -File .\setup.ps1` for automatic detection, or set `.env` yourself:

- **NVIDIA GPU** — Ollama runs in Docker; Docker Desktop's WSL2 backend passes the GPU through (needs a current NVIDIA driver):
  ```env
  HARDWARE_TYPE=nvidia
  VRAM_TIER=mid        # low for cards under 12 GB
  ```
- **AMD or Intel GPU** — Docker Desktop has no ROCm passthrough, so run [Ollama for Windows](https://ollama.com/download/windows) natively (it uses the GPU directly) and point the containers at it:
  ```env
  HARDWARE_TYPE=native
  VRAM_TIER=mid
  ```
  Then pull the models (`setup.ps1` does this for you): `ollama pull qwen3-embedding:4b` and `ollama pull qwen3-vl:8b-instruct`.

Then start it the same way as everywhere else:
```powershell
docker compose up -d
```

**CPU-only / no GPU**

Set `VRAM_TIER=low` and `HARDWARE_TYPE=cpu`. The 4B model will run on CPU — slow but functional.

---

#### vLLM (opt-in, NVIDIA or AMD)

vLLM serves the same model families at higher precision (FP8, or BF16 where the GPU has no FP8 kernel) with higher throughput under concurrent load. The limited retrieval comparison was broadly similar to Ollama's, so it is only worth it on a GPU with memory to spare. Setup conservatively recommends 32 GB for low and 48 GB for mid, budgeting BF16, the embedder and KV cache for either model family. FP8 may work with less memory; free memory and other workloads also matter. Run `./setup.sh --vllm` (`setup.ps1 -Vllm`), or set:

**NVIDIA** (Ampere or newer, incl. DGX Spark GB10):
```env
HARDWARE_TYPE=nvidia-vllm
VRAM_TIER=low              # use mid when sufficient memory is available
```

**AMD (ROCm)** — RDNA 3 / 3.5 / 4 (Radeon RX 7000/9000, Radeon AI PRO, Strix Halo / Point) and Instinct MI300+:
```env
HARDWARE_TYPE=amd-vllm
VRAM_TIER=low
COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml
```

Quantization is chosen automatically: **FP8** where the GPU has native FP8 hardware (NVIDIA Ampere+, AMD MI300+), otherwise **BF16** — RDNA has no FP8 kernel in vLLM. Set `VLLM_QUANTIZATION=fp8` or `bf16` in `.env` to force one. `setup.sh` sets `VLLM_ROCM_IMAGE` from your GPU's gfx target; the `rocm10.x` images are unified, so one tag covers RDNA and Instinct.

Both vLLM paths pin a specific release rather than `:latest` (which tracks nightly builds). ROCm publishes less often, so the AMD image trails the NVIDIA one. Override either with `VLLM_IMAGE` / `VLLM_ROCM_IMAGE` in `.env`; the launcher logs the vLLM version it detects at startup.

> **Switching between Ollama and vLLM requires a separate index and re-analysis.** Even vectors with the same dimensions may be incompatible. Follow [Embedding index migration](#embedding-index-migration) to preserve your existing library.

```bash
docker compose up -d
```

> **Note:** The base compose uses the official vLLM image (amd64 + arm64). For AMD, the `docker-compose.rocm.yml` override swaps in the ROCm build with device passthrough (requires Docker Compose >= 2.24.0 for the `!reset` merge tag). Both vLLM containers start through `docker/vllm/serve.py`, which picks FP8 or BF16 for your GPU and sizes GPU memory in GB from what is actually free — so they share a GPU (or unified memory) without crash-looping. For Apple Silicon, use `HARDWARE_TYPE=apple` instead.

---

#### Custom (bring your own servers)

Use this if you already have Ollama, vLLM, LiteLLM, or any OpenAI-compatible server running. Point each URL at your existing endpoints:

```env
HARDWARE_TYPE=custom

# Embedding model (OpenAI /v1/embeddings compatible)
EMBED_URL=http://your-server:8001
EMBED_MODEL=Qwen/Qwen3-Embedding-4B

# Vision / captioning model (supports image_url in messages)
VLM_URL=http://your-server:8000
VLM_MODEL=llava-hf/llava-v1.6-vicuna-7b-hf

# Text LLM for summaries
LLM_URL=http://your-server:8000
LLM_MODEL=Qwen/Qwen3-8B
```

> **Changing the embedding model or backend requires a separate index.** See
> [Embedding index migration](#embedding-index-migration); do not delete the
> old collection to make a new model work.

```bash
docker compose up -d
```

---

#### Model size tiers

`VRAM_TIER` controls which model size is selected for local backends (`ollama`, `ollama-native`, `vllm`). `MODEL_FAMILY` selects the model family (`qwen` = Qwen3-VL, `gemma` = Gemma 3).

| Tier | Model (qwen) | Ollama tag | Ollama Q4 (measured) | vLLM FP8 | vLLM BF16 (AMD RDNA) |
|------|--------------|------------|----------------------|----------|----------------------|
| `low` | Qwen3-VL-4B | `qwen3-vl:4b-instruct` | ~9 GB | ≥ 23 GB | ≥ 22 GB |
| `mid` | Qwen3-VL-8B | `qwen3-vl:8b-instruct` | ~11 GB | ≥ 28 GB | ≥ 26 GB |

Memory figures include the embedding model (Qwen3-Embedding-4B) loaded alongside. Ollama figures are from one Strix Halo test, not a discrete-GPU certification.

> Only `low` and `mid` are served locally — they are the two tiers that fit on hardware people actually run this on. `high` (Qwen3-VL-32B, ~67 GB) and `ultra` (Qwen3-VL-235B-A22B, ~471 GB) need a multi-GPU host, so they are **external-endpoint only**: set `HARDWARE_TYPE=custom` and point the app at your own vLLM/Ollama server instead. `setup.sh` never selects them, and the app falls back to `mid` if one is set by hand.
>
> vLLM figures include the embedder (~10 GB BF16 on vLLM) and KV cache. FP8 quantizes only the LLM decoder; the vision encoder stays BF16. On unified-memory machines (Apple Silicon, DGX Spark, Strix Halo) memory is shared with the system — `setup.sh` leaves ~20% free and stays on `mid` there by default.
>
> vLLM's memory reservation is sized in GB to the model and to what is free at startup, so a small model never reserves a whole 128 GB unified pool, and another process holding GPU memory produces a clear error instead of a crash loop. Override with `VLLM_GPU_MEMORY_GB` (budget) or `VLLM_GPU_MEMORY_UTIL` (raw fraction) in `.env`.
>
> The embedder runs with a capped context (`VLLM_EMBED_MAX_MODEL_LEN`, default 8192). Embeddings are computed over short chunks, and the model's native 40960-token window would need a KV cache far larger than the embedder's budget — vLLM refuses to start rather than degrading, so do not raise this without also raising the embedder's memory budget.

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
starts the matching model server automatically (or none, for `custom`,
`apple` and `native`). If you change `HARDWARE_TYPE` later, run
`docker compose down` first so the previous backend's containers stop.

### Step 4 — Open the UI

Navigate to **http://localhost:3000**

Go to **Settings** to verify your AI backend — the banner now shows detected hardware (GPU model, VRAM, ROCm device status), resolved model names, and service connectivity. Configure Whisper transcription quality (Low / Medium / High / Very high / Fast large), set the frame extraction interval, cap the CPU threads FFmpeg uses during analysis (defaults to half your cores so the host stays responsive), tune max concurrent jobs, toggle auto-analyze on import, and choose a hardware acceleration backend for FFmpeg video decoding.

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

> On first launch the reranker model (`BAAI/bge-reranker-v2-m3`, ~2.3 GB) downloads automatically and is cached under `./data` so it persists across restarts. It runs on CPU inside the API container — no extra GPU required.

## Hardware Types

| Value | AI Backend | Description |
|-------|------------|-------------|
| `nvidia` | ollama | Ollama runs in Docker on an NVIDIA GPU. Requires nvidia-container-toolkit (built into Docker Desktop on Windows). |
| `amd` | ollama | Ollama runs in Docker on an AMD GPU via ROCm (Linux). Set `COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml` in `.env` for GPU video decode. |
| `nvidia-vllm` | vllm | Opt-in: vLLM runs in Docker. NVIDIA Ampere or newer (incl. DGX Spark), 32 GB+ recommended. |
| `amd-vllm` | vllm | Opt-in: vLLM on ROCm (RDNA 3+ / MI300+, Linux, 32 GB+ recommended). Requires `COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml`. |
| `apple` | ollama-native | Ollama runs on your host machine. **Required for Apple Silicon** — Docker VMs can't access Metal. Run `ollama serve` first. |
| `native` | ollama-native | Ollama runs on your host and drives any GPU it supports (Intel Arc via Vulkan, AMD/NVIDIA on Windows, older AMD). On Linux set `OLLAMA_HOST=0.0.0.0`. |
| `cpu` | ollama | Ollama runs in Docker on CPU only. |
| `custom` | custom | Bring your own servers. Set `EMBED_URL`, `VLM_URL`, `LLM_URL` in `.env` |

## Embedding index migration

`HARDWARE_TYPE=nvidia` and `amd` now select Ollama; older versions used vLLM.
Changing embedding model, backend, or quantization requires re-indexing, even
when the vector dimensions match. The app records the configured model and
backend beside the vectors and blocks search, analysis, editing, and imports
when they do not match. An old index without identity metadata is treated as
unknown, not guessed from its dimensions.

To switch to Ollama while keeping your old library intact:

1. Stop the old stack before changing `.env`: `docker compose --profile '*' down`.
2. Set `APP_DATA_DIR=./data-ollama` and select the new `HARDWARE_TYPE` in `.env`.
   If `QDRANT_STORAGE_PATH` is explicitly set, remove it or set it to a new,
   empty path such as `./data-ollama/qdrant`. Keep `MODELS_DIR` unchanged to
   reuse downloaded models.
3. Start the new stack, import the video files again, and analyze them. This is
   a new library: old captions, tags, and edits remain in the original data
   directory, not in the new one. Switching both the data path and backend back
   restores the original library.

To continue using a **verified** legacy index instead, configure the same model
and serving backend that created it, set `LEGACY_INDEX_EMBED_MODEL` to that exact
model ID, and start the app once. Remove the setting after the index has been
adopted. For example, an old BGE-M3 index needs `BAAI/bge-m3` and a matching
external endpoint; selecting `amd-vllm` or `nvidia-vllm` alone now selects
Qwen3-Embedding-4B and cannot reuse it. Do not set the legacy option if the
original model or backend is uncertain. Legacy archives without backend
metadata require the same explicit assertion before import.

For custom endpoints, changing the endpoint URL is conservatively treated as
a different embedding space. The app cannot detect weights or quantization
changed *in place* behind the same model ID and URL; use a separate data
directory and re-analyze when that happens. Copy the complete app and Qdrant
data together when relocating an existing library so the lexical and vector
indexes stay aligned.

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
| `WHISPER_MODEL` | Whisper transcription model: `base`, `small`, `medium` (default), `large-v3`, `distil-large-v3` |
| `WHISPER_DEVICE` | Device for Whisper: `cpu` by default; other devices require a compatible custom runtime |
| `WHISPER_COMPUTE_TYPE` | Whisper precision: `int8` by default; available types depend on the runtime |
| `HARDWARE_TYPE` | Hardware platform selection (nvidia/amd/apple/native/cpu/custom, or nvidia-vllm/amd-vllm) |
| `VRAM_TIER` | Model size for local backends (low/mid) |
| `VIDEO_DIR_1..5` | Up to 5 video library directories |
| `APP_DATA_DIR` | App library data and, by default, Qdrant storage (default: `./data`) |
| `QDRANT_STORAGE_PATH` | Optional separate Qdrant storage path; keep it paired with the app data |
| `LEGACY_INDEX_EMBED_MODEL` | One-time, explicit adoption of a verified legacy index or archive |
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
   - `auto` — probes accessible CUDA and VAAPI devices inside the API container, otherwise uses software decoding (the setup scripts select this)
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

   **AMD ROCm (Strix Halo):** The `docker-compose.rocm.yml` override already passes `/dev/kfd` and `/dev/dri` to `local-vss-api` — just make sure `COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml` is set in `.env` (`setup.sh` does this).

3. **Verify** — open the **Settings** UI. With `auto`, it shows the resolved backend (`cuda`, `vaapi`, or `off`). If an accelerated decode fails, the app retries in software and reports the fallback in Settings and the API logs. It keeps using software until settings are reapplied or the API restarts. If software decoding also fails, check those logs.

   An isolated Strix Halo test selected VAAPI and extracted a frame successfully. On the NVIDIA test host, the default API container had no GPU passthrough, so `auto` correctly selected software decoding.

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
                                        bge-reranker-v2-m3 (search reranking, CPU)
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
│   ├── fusion.py      # Cross-modal description fusion
│   ├── local_models.py# Model tier registry (family × VRAM)
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
├── docs/
│   └── QUERY-MECHANICS.md  # How search works (end-user guide)
├── setup.sh            # Smart setup — Linux/macOS
├── setup.ps1           # Smart setup — Windows
├── docker-compose.yml  # Full stack orchestration
├── docker-compose.rocm.yml  # AMD ROCm GPU override
├── .env.example        # All configuration options with documentation
├── LICENSE             # MIT License
└── requirements.txt    # Python dependencies
```

## License

MIT
