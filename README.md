# Local Video Search & Summarization

A self-hosted video search and summarization system powered by local AI models. Index your video library, search in natural language, and get AI-generated summaries.

## Features

- **Natural Language Search** - Ask questions like "Find the moment where the car crashes" or "Show me cooking scenes"
- **Segment-Level Search** - Not just which video, but exactly when in the video something happens
- **AI Summarization** - Automatic video summaries powered by local LLM
- **Local AI** - All processing happens on your hardware via Ollama and vLLM
- **Self-Hosted** - No cloud dependencies, no API keys needed

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