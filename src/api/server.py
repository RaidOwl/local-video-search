from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import asyncio
import os
import random
import tempfile
import uuid
import json

from ..config import get_settings
from ..local_models import resolve_local_config
from ..database import Database, Video
from ..ingest.video_scanner import find_video_files, get_video_metadata, compute_file_hash
from ..ingest.frame_extractor import FrameExtractor
from ..embed.embedding_generator import EmbeddingService
from ..vlm.caption_generator import VLMService
from ..search.query_processor import QueryProcessor
from ..search.segment_retriever import SegmentRetriever
from ..summarize.video_summarizer import SummarizationService
from ..transcribe.transcriber import TranscriptionService

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _auto_import_task
    await db.init_db()
    await embed_service.ensure_collection()
    stuck = [v for v in await db.get_all_videos() if v.status in ('processing', 'transcribing')]
    for v in stuck:
        await db.update_video(v.id, {'frame_count': 0})
        _start_analysis(v.path, v.id, context=getattr(v, 'context', '') or '')
    # Restart any retranscription that was interrupted
    stuck_audio = [v for v in await db.get_all_videos() if v.status == 'retranscribing']
    for v in stuck_audio:
        task = asyncio.create_task(_retranscribe_audio(v.id, v.path))
        _active_tasks[v.id] = task
    _auto_import_task = asyncio.create_task(_auto_import_loop())
    yield

app = FastAPI(
    title="Local Video Search & Summarization",
    description="""REST API for ingesting, analyzing, and searching local video libraries.

**Base URL:** `http://<host>:8000` (or `/api/` when accessed via the frontend)

### Quick start
1. `POST /ingest` — add a video file to the library
2. `POST /reprocess/{video_id}` — run vision + speech analysis
3. `GET /search?q=your+query` — semantic search across all analyzed videos
4. `GET /search/segments?q=your+query&video_id=...` — search within a specific video
""",
    version="1.0.0",
    openapi_tags=[
        {"name": "Search", "description": "Semantic search across video frames and transcripts."},
        {"name": "Videos", "description": "Library management — list, update, hide, and delete videos."},
        {"name": "Analysis", "description": "Ingest, analyze, transcribe, and summarize videos."},
        {"name": "Media", "description": "Stream, download, and clip video files; fetch frames and thumbnails."},
        {"name": "System", "description": "Health, settings, model status, and storage diagnostics."},
    ],
    servers=[
        {"url": "/api", "description": "Via frontend proxy (port 3000)"},
        {"url": "/", "description": "Direct API (port 8000)"},
    ],
    swagger_ui_parameters={"defaultModelsExpandDepth": -1, "tryItOutEnabled": True},
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings
settings = get_settings()

# Database
db = Database(settings.database_url)

# Settings override file (persists user changes across restarts)
SETTINGS_OVERRIDE_PATH = Path("/data/settings_override.json")

def _load_override() -> dict:
    if SETTINGS_OVERRIDE_PATH.exists():
        try:
            return json.loads(SETTINGS_OVERRIDE_PATH.read_text())
        except Exception:
            return {}
    return {}

def _effective_cfg() -> dict:
    ai_backend = settings.ai_backend

    if ai_backend == "custom":
        base = {
            "vllm_embed_url":   settings.vllm_embed_url,
            "vllm_vlm_url":     settings.vllm_vlm_url,
            "vllm_url":         settings.vllm_url,
            "qdrant_url":       settings.qdrant_url,
            "vllm_embed_model": settings.vllm_embed_model,
            "vllm_vlm_model":   settings.vllm_vlm_model,
            "vllm_model":       settings.vllm_model,
            "whisper_model":    settings.whisper_model,
            "frame_extraction_interval": settings.frame_extraction_interval,
            "embed_batch_size": settings.embed_batch_size,
            "ai_backend":       "custom",
        }
    else:
        base = resolve_local_config(
            backend=ai_backend,
            family=settings.model_family,
            tier=settings.vram_tier,
            settings=settings,
        )
        base["qdrant_url"]       = settings.qdrant_url
        base["whisper_model"]    = settings.whisper_model
        base["frame_extraction_interval"] = settings.frame_extraction_interval
        base["embed_batch_size"] = settings.embed_batch_size
        base["ai_backend"]       = ai_backend
        base["vram_tier"]        = settings.vram_tier
        base["model_family"]     = settings.model_family

    # For local backends (non-custom), never allow the override file to redirect
    # model URLs or swap model names — those are always determined by the tier.
    # Only user-tunable processing settings are overridable.
    _URL_MODEL_KEYS = {
        'vllm_embed_url', 'vllm_vlm_url', 'vllm_url', 'qdrant_url',
        'vllm_embed_model', 'vllm_vlm_model', 'vllm_model',
    }
    override = _load_override()
    if ai_backend != "custom":
        override = {k: v for k, v in override.items() if k not in _URL_MODEL_KEYS}
    base.update(override)
    base.setdefault("max_concurrent_jobs", 8)
    base.setdefault("import_blacklist", [])
    base.setdefault("auto_import_interval", 0)
    base.setdefault("auto_analyze", False)
    base.setdefault("default_context", "")
    return base


def _backend_warning(cfg: dict) -> str | None:
    """Return a warning string if the backend/platform combo is problematic."""
    import platform
    backend = cfg.get("ai_backend", "custom")
    arch = platform.machine()
    is_apple_silicon = arch in ("arm64", "aarch64") and platform.system() == "Darwin"
    if backend == "vllm" and is_apple_silicon:
        return (
            "vLLM requires an NVIDIA GPU and does not support Apple Silicon. "
            "Use AI_BACKEND=ollama-native instead — install Ollama on your Mac "
            "and it will use Metal automatically."
        )
    if backend == "ollama" and is_apple_silicon:
        return (
            "Ollama in Docker loses Metal acceleration on Apple Silicon. "
            "Consider AI_BACKEND=ollama-native for full GPU performance."
        )
    return None

_cfg = _effective_cfg()
_analysis_semaphore = asyncio.Semaphore(int(_cfg.get("max_concurrent_jobs", 8)))

# Services
embed_service = EmbeddingService(_cfg["vllm_embed_url"], _cfg["qdrant_url"], model=_cfg["vllm_embed_model"])
vlm_service = VLMService(_cfg["vllm_vlm_url"], model=_cfg["vllm_vlm_model"])
summarizer_service = SummarizationService(_cfg["vllm_url"], model=_cfg["vllm_model"])
transcription_service = TranscriptionService(model_size=_cfg["whisper_model"])
query_processor = QueryProcessor()
segment_retriever = SegmentRetriever()

# Cross-encoder re-ranker — lazy-loaded on first enhanced search
_reranker = None

# Frame extractor
frame_extractor = FrameExtractor(interval=_cfg["frame_extraction_interval"])

def _reinit_services(cfg: dict):
    global embed_service, vlm_service, summarizer_service, frame_extractor, transcription_service, _analysis_semaphore
    embed_service = EmbeddingService(cfg["vllm_embed_url"], cfg["qdrant_url"], model=cfg["vllm_embed_model"])
    vlm_service = VLMService(cfg["vllm_vlm_url"], model=cfg["vllm_vlm_model"])
    summarizer_service = SummarizationService(cfg["vllm_url"], model=cfg["vllm_model"])
    transcription_service = TranscriptionService(model_size=cfg["whisper_model"])
    frame_extractor = FrameExtractor(interval=int(cfg["frame_extraction_interval"]))
    _analysis_semaphore = asyncio.Semaphore(int(cfg.get("max_concurrent_jobs", 8)))

# Persistent frame storage directory
FRAMES_DIR = Path("/data/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


# Pydantic models
class IngestRequest(BaseModel):
    video_path: Optional[str] = None
    directory: Optional[str] = None
    context: Optional[str] = None


class IngestResponse(BaseModel):
    video_id: str
    status: str
    frames_processed: int


class SearchResponse(BaseModel):
    results: List[dict]
    query: str


class VideoResponse(BaseModel):
    id: str
    name: str
    display_name: Optional[str]
    path: str
    duration: float
    summary: Optional[str]
    tags: List[str]
    status: str
    frame_count: int
    thumbnail_path: Optional[str] = None
    context: Optional[str] = None
    analyzed_at: Optional[str] = None
    has_transcript: bool = False


class VideoUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    tags: Optional[List[str]] = None
    thumbnail_path: Optional[str] = None
    context: Optional[str] = None


class SegmentSearchResponse(BaseModel):
    segments: List[dict]
    video_id: str
    query: str


class FrameUpdateRequest(BaseModel):
    caption: str


# Track in-flight tasks by video_id so we can cancel on re-trigger
_active_tasks: dict = {}
_auto_import_task: asyncio.Task | None = None


async def _expand_query(q: str) -> list[str]:
    """Use the LLM to generate alternative phrasings of a search query.
    Returns a list of 2-3 reformulations (empty list on failure)."""
    cfg = _effective_cfg()
    url = cfg.get("vllm_url", "").rstrip("/")
    model = cfg.get("vllm_model", "")
    if not url or not model:
        return []
    prompt = (
        f'Generate 3 alternative search queries for a personal video library search. '
        f'The user is looking for: "{q}"\n\n'
        f'Write queries that use different vocabulary — synonyms, technical terms, '
        f'and different phrasings that might appear in video transcripts or captions. '
        f'Return ONLY the 3 queries, one per line, no numbering or explanation.'
    )
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{url}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 120, "temperature": 0.4},
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            lines = [ln.strip().strip('"\'') for ln in text.splitlines() if ln.strip()]
            return [ln for ln in lines if len(ln) > 4][:3]
    except Exception:
        return []


async def _rerank_segments(query: str, segments: list[dict]) -> list[dict]:
    """Re-rank candidate segments using a local cross-encoder model.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (~23 MB, runs on CPU).
    Lazy-loaded on first call; weights cached in /data/hf_cache so they
    survive container rebuilds.  Final score blends Qdrant + cross-encoder:
        0.4 * qdrant_score + 0.6 * sigmoid(cross_encoder_raw)
    Falls back to original scores on any failure.
    """
    if not segments:
        return segments

    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("Loading cross-encoder model (first enhanced search — downloading ~23 MB)...")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            print("Cross-encoder ready.")
        except Exception as e:
            print(f"Re-rank: failed to load cross-encoder: {e}")
            return segments

    try:
        import asyncio as _asyncio, math as _math
        pairs = [
            (query, s.get("caption", "").replace("[transcript] ", "").strip())
            for s in segments
        ]
        # Run synchronous predict in a thread so we don't block the event loop
        loop = _asyncio.get_running_loop()
        raw_scores = await loop.run_in_executor(None, _reranker.predict, pairs)

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + _math.exp(-x / 3.0))  # /3 softens the curve

        norm = [_sigmoid(float(s)) for s in raw_scores]
        print(f"Re-rank '{query}': top={sorted(norm, reverse=True)[:8]}")

        for seg, ns in zip(segments, norm):
            seg["score"] = 0.4 * seg["score"] + 0.6 * ns

        return segments

    except Exception as e:
        print(f"Re-rank predict failed: {e}")
        return segments

def _start_analysis(video_path: str, video_id: str, context: str = '', resume: bool = False) -> None:
    """Cancel any existing task for this video, then start a fresh one."""
    existing = _active_tasks.get(video_id)
    if existing and not existing.done():
        existing.cancel()
    task = asyncio.create_task(process_video(video_path, video_id, context=context, resume=resume))
    _active_tasks[video_id] = task

# Background tasks
async def process_video(video_path: str, video_id: str, context: str = '', resume: bool = False):
    """Process a video: extract frames, caption, transcribe audio, and index."""
    try:
        await db.update_video(video_id, {'status': 'processing'})
        async with _analysis_semaphore:
            if not resume:
                # Full restart: clear any leftover frame records and embeddings
                await db.update_video(video_id, {'frame_count': 0})
                await db.delete_frames_for_video(video_id)
                try:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    embed_service.qdrant_client.delete(
                        collection_name=embed_service.collection_name,
                        points_selector=Filter(
                            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                        ),
                    )
                except Exception:
                    pass
                existing_frame_ids: set[str] = set()
                has_existing_transcript = False
            else:
                # Resume: collect which frame IDs and transcript segments already exist
                existing_db_frames = await db.get_frames_for_video(video_id)
                existing_frame_ids = {f.id for f in existing_db_frames}
                has_existing_transcript = any(f.frame_path == '' for f in existing_db_frames)

            # Use persistent per-video frame directory
            frame_dir = FRAMES_DIR / video_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_running_loop()
            frames = await loop.run_in_executor(
                None, lambda: frame_extractor.extract_frames(
                    video_path, str(frame_dir), prefix=f"frame_{video_id}"
                )
            )

            if not frames:
                await db.update_video(video_id, {'status': 'error'})
                return

            for i, frame_path in enumerate(frames):
                frame_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_frame_{i}"))
                if frame_id in existing_frame_ids:
                    # Already captioned and indexed in a previous run — skip
                    continue
                timestamp = i * frame_extractor.interval
                caption = await vlm_service.caption_frame(frame_path, context=context)
                await embed_service.index_frame(
                    video_id=video_id, frame_path=frame_path,
                    timestamp=timestamp, caption=caption
                )
                await db.create_frame({
                    "id": frame_id, "video_id": video_id,
                    "timestamp": timestamp, "frame_path": frame_path,
                    "caption": caption, "embedding_id": frame_id
                })
                await db.update_video(video_id, {'frame_count': i + 1})

            # Set a random frame as the default thumbnail if none set
            video_check = await db.get_video(video_id)
            if frames and not video_check.thumbnail_path:
                await db.update_video(video_id, {'thumbnail_path': str(random.choice(frames))})

            if resume and has_existing_transcript:
                print(f"Resume: skipping transcription for {video_id} (already indexed)")
                await db.update_video(video_id, {'status': 'analyzed'})
                return

            await db.update_video(video_id, {'status': 'transcribing'})
            print(f"Transcribing audio for {video_id}...")
            segments = await loop.run_in_executor(
                None, transcription_service.transcribe_video, video_path
            )
            print(f"Whisper: {len(segments)} transcript segment(s) for {video_id}")

            for seg in segments:
                text = f"[transcript] {seg.text}"
                await embed_service.index_frame(
                    video_id=video_id, frame_path="",
                    timestamp=seg.start, caption=text
                )
                seg_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_transcript_{seg.start}"))
                await db.create_frame({
                    "id": seg_id, "video_id": video_id,
                    "timestamp": seg.start, "frame_path": "",
                    "caption": text, "embedding_id": seg_id
                })

            # Sliding windows: embed overlapping groups of 3 consecutive transcript
            # segments so multi-sentence topics get a single strong embedding rather
            # than being spread across individually-weak points.
            WINDOW_SIZE = 3
            WINDOW_STEP = 2  # 1-segment overlap between adjacent windows
            for i in range(0, len(segments) - 1, WINDOW_STEP):
                win = segments[i:i + WINDOW_SIZE]
                if len(win) < 2:
                    continue
                combined = " ".join(s.text.strip() for s in win if s.text.strip())
                if not combined:
                    continue
                win_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                    f"{video_id}_win_{win[0].start:.3f}_{win[-1].end:.3f}"))
                await embed_service.index_frame(
                    video_id=video_id, frame_path="",
                    timestamp=win[0].start,
                    caption=f"[transcript] {combined}",
                    _point_id=win_id,
                )

        await db.update_video(video_id, {'status': 'analyzed', 'has_transcript': 1})

    except asyncio.CancelledError:
        print(f"Analysis cancelled for {video_id}")
        await db.update_video(video_id, {'status': 'error'})
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            embed_service.qdrant_client.delete(
                collection_name=embed_service.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                ),
            )
        except Exception:
            pass
        raise
    except Exception as e:
        import traceback
        print(f"Error processing video {video_id}: {e}")
        print(traceback.format_exc())
        await db.update_video(video_id, {'status': 'error'})
    finally:
        _active_tasks.pop(video_id, None)



# API endpoints


# All mounted video library directories (lib1–lib5 set via VIDEO_DIR_1–5 in .env)
VIDEO_SCAN_DIRS = [Path(f"/data/videos/lib{i}") for i in range(1, 6)]


async def _auto_import(auto_analyze: bool = False, default_context: str = ""):
    """Scan all configured video library directories and add new files."""
    import fnmatch as _fnmatch
    blacklist: list = _effective_cfg().get("import_blacklist", [])
    for video_dir in VIDEO_SCAN_DIRS:
        if not video_dir.exists():
            continue
        for video_path in find_video_files(str(video_dir)):
            path_str = str(video_path)
            if blacklist and any(_fnmatch.fnmatch(path_str, pat) for pat in blacklist):
                continue
            if await db.get_video_by_path(path_str):
                continue
            video_id = str(uuid.uuid4())
            metadata = get_video_metadata(video_path)
            await db.create_video({
                "id": video_id, "name": video_path.stem, "path": str(video_path),
                "duration": metadata.get("duration", 0), "width": metadata.get("width", 0),
                "height": metadata.get("height", 0), "size": metadata.get("size", 0),
                "file_hash": compute_file_hash(video_path),
            })
            await db.update_video(video_id, {"status": "ready"})
            if auto_analyze:
                _start_analysis(path_str, video_id, context=default_context)


async def _auto_import_loop():
    """Run auto-import on startup, then loop periodically if auto_import_interval > 0."""
    while True:
        cfg = _effective_cfg()
        await _auto_import(
            auto_analyze=bool(cfg.get("auto_analyze", False)),
            default_context=str(cfg.get("default_context", "")),
        )
        interval = int(cfg.get("auto_import_interval", 0))
        if interval <= 0:
            break
        await asyncio.sleep(interval * 60)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy"}


@app.post("/reimport", tags=["System"])
async def reimport():
    """Scan all configured video library directories and add any new videos."""
    await _auto_import(
        auto_analyze=bool(_effective_cfg().get("auto_analyze", False)),
        default_context=str(_effective_cfg().get("default_context", "")),
    )
    videos = await db.get_all_videos()
    return {"total": len(videos)}


@app.get("/status/models", tags=["System"])
async def model_status():
    """Check reachability of each AI model endpoint."""
    import httpx
    cfg = _effective_cfg()
    async def probe(url: str, name: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{url}/v1/models")
                models = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
                return {"name": name, "url": url, "ok": r.status_code == 200, "models": models}
        except Exception:
            return {"name": name, "url": url, "ok": False, "models": []}

    results = await asyncio.gather(
        probe(cfg["vllm_embed_url"], "Embeddings"),
        probe(cfg["vllm_vlm_url"], "Vision AI"),
        probe(cfg["vllm_url"], "Summarization"),
    )
    return {"services": list(results)}


@app.get("/status/qdrant", tags=["System"])
async def qdrant_health():
    """Check Qdrant health: reachability, collection state, and per-video embedding integrity."""
    from collections import Counter
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    out: dict = {
        "reachable": False,
        "collection_exists": False,
        "total_points": 0,
        "issues": [],
        "summary": {"videos_checked": 0, "ok": 0, "missing": 0, "incomplete": 0, "orphaned_points": 0},
    }

    try:
        client = embed_service.qdrant_client
        coll = embed_service.collection_name
        out["reachable"] = True

        collections = [c.name for c in client.get_collections().collections]
        if coll not in collections:
            return out
        out["collection_exists"] = True
        out["total_points"] = client.count(collection_name=coll).count

        # Single scroll pass to build video_id -> point count map
        qdrant_counts: Counter = Counter()
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=coll,
                with_payload=["video_id"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for r in records:
                vid = r.payload.get("video_id") if r.payload else None
                if vid:
                    qdrant_counts[vid] += 1
            if offset is None:
                break

        videos = await db.get_all_videos()
        db_ids = {v.id for v in videos}
        analyzed = [v for v in videos if v.status == "analyzed"]
        out["summary"]["videos_checked"] = len(analyzed)

        for video in analyzed:
            q_count = qdrant_counts.get(video.id, 0)
            db_frames = video.frame_count or 0
            if q_count == 0:
                out["issues"].append({
                    "type": "missing",
                    "video_id": video.id,
                    "name": video.display_name or video.name,
                    "db_frames": db_frames,
                    "qdrant_points": 0,
                })
                out["summary"]["missing"] += 1
            elif q_count < db_frames:
                out["issues"].append({
                    "type": "incomplete",
                    "video_id": video.id,
                    "name": video.display_name or video.name,
                    "db_frames": db_frames,
                    "qdrant_points": q_count,
                })
                out["summary"]["incomplete"] += 1
            else:
                out["summary"]["ok"] += 1

        # Orphaned: points in Qdrant for video_ids not in DB at all
        orphaned_ids = set(qdrant_counts.keys()) - db_ids
        out["summary"]["orphaned_points"] = sum(qdrant_counts[vid] for vid in orphaned_ids)
        if orphaned_ids:
            out["issues"].append({
                "type": "orphaned",
                "video_ids": list(orphaned_ids),
                "qdrant_points": out["summary"]["orphaned_points"],
            })

    except Exception as e:
        import traceback
        out["error"] = str(e)
        out["traceback"] = traceback.format_exc()

    return out


@app.post("/status/qdrant/purge-orphans", tags=["System"])
async def qdrant_purge_orphans():
    """Delete Qdrant points for video_ids that no longer exist in the database."""
    from collections import Counter
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    client = embed_service.qdrant_client
    coll = embed_service.collection_name

    # Collect all video_ids present in Qdrant
    qdrant_video_ids: set = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=coll,
            with_payload=["video_id"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for r in records:
            vid = r.payload.get("video_id") if r.payload else None
            if vid:
                qdrant_video_ids.add(vid)
        if offset is None:
            break

    videos = await db.get_all_videos()
    db_ids = {v.id for v in videos}
    orphaned_ids = list(qdrant_video_ids - db_ids)

    if not orphaned_ids:
        return {"deleted_points": 0, "orphaned_video_ids": []}

    deleted = 0
    for vid in orphaned_ids:
        result = client.delete(
            collection_name=coll,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchAny(any=[vid]))]
            ),
        )
        if hasattr(result, "result"):
            deleted += getattr(result.result, "deleted", 0) or 0

    return {"deleted_points": deleted, "orphaned_video_ids": orphaned_ids}


@app.get("/settings", tags=["System"])
async def get_settings_endpoint():
    """Return current effective settings."""
    cfg = _effective_cfg()
    warning = _backend_warning(cfg)
    video_dirs = [str(d) for d in VIDEO_SCAN_DIRS if d.exists() and any(d.iterdir())]
    return {**cfg, "backend_warning": warning, "video_dirs": video_dirs}


class SettingsUpdateRequest(BaseModel):
    vllm_embed_url: Optional[str] = None
    vllm_vlm_url: Optional[str] = None
    vllm_url: Optional[str] = None
    qdrant_url: Optional[str] = None
    vllm_embed_model: Optional[str] = None
    vllm_vlm_model: Optional[str] = None
    vllm_model: Optional[str] = None
    whisper_model: Optional[str] = None
    frame_extraction_interval: Optional[int] = None
    embed_batch_size: Optional[int] = None
    max_concurrent_jobs: Optional[int] = None
    import_blacklist: Optional[List[str]] = None
    auto_import_interval: Optional[int] = None
    auto_analyze: Optional[bool] = None
    default_context: Optional[str] = None
    # Note: ai_backend, vram_tier, model_family are .env-only — not overridable via UI


async def _sync_blacklist():
    """Auto-hide videos matching blacklist patterns; restore those no longer matching."""
    import fnmatch as _fnmatch
    blacklist: list = _effective_cfg().get("import_blacklist", [])
    # Need all videos regardless of hidden state to sync correctly
    all_videos = await db.get_all_videos(include_all=True)
    for video in all_videos:
        matches = bool(blacklist) and any(_fnmatch.fnmatch(video.path, pat) for pat in blacklist)
        currently_blacklisted = video.hidden == 'blacklist'
        manually_hidden = video.hidden == '1'
        if matches and not manually_hidden and not currently_blacklisted:
            await db.update_video(video.id, {'hidden': 'blacklist'})
        elif not matches and currently_blacklisted:
            await db.update_video(video.id, {'hidden': '0'})


@app.put("/settings", tags=["System"])
async def update_settings_endpoint(req: SettingsUpdateRequest):
    """Persist settings overrides and reinitialize services."""
    global _auto_import_task
    override = _load_override()
    for field, val in req.model_dump(exclude_none=True).items():
        override[field] = val
    SETTINGS_OVERRIDE_PATH.write_text(json.dumps(override, indent=2))
    cfg = _effective_cfg()
    _reinit_services(cfg)
    asyncio.create_task(_sync_blacklist())
    # Restart auto-import loop if just enabled and not already running
    if cfg.get("auto_import_interval", 0) > 0 and (not _auto_import_task or _auto_import_task.done()):
        _auto_import_task = asyncio.create_task(_auto_import_loop())
    return cfg



@app.get("/browse", tags=["System"])
async def browse_directory(path: str = "/data/videos/lib1"):
    """List directories and video files within the configured library dirs."""
    base = Path("/data/videos").resolve()
    try:
        target = Path(path).resolve()
        target.relative_to(base)  # restrict to /data/videos
    except (ValueError, Exception):
        target = base

    if not target.is_dir():
        target = target.parent
        if not target.is_dir():
            target = base

    VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    entries = []
    try:
        for item in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                entries.append({"name": item.name, "path": str(item), "type": "dir"})
            elif item.suffix.lower() in VIDEO_EXTS:
                entries.append({"name": item.name, "path": str(item), "type": "video"})
    except PermissionError:
        pass

    return {
        "path": str(target),
        "parent": str(target.parent) if target.resolve() != base else None,
        "entries": entries,
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Analysis"])
async def ingest_video(request: IngestRequest):
    """Ingest a video or directory of videos into the library without auto-analyzing."""
    if request.video_path:
        # Single video
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

        existing = await db.get_video_by_path(str(video_path))
        if existing:
            return IngestResponse(video_id=existing.id, status=existing.status, frames_processed=existing.frame_count)

        video_id = str(uuid.uuid4())
        metadata = get_video_metadata(video_path)
        await db.create_video({
            "id": video_id,
            "name": video_path.stem,
            "path": str(video_path),
            "duration": metadata.get("duration", 0),
            "width": metadata.get("width", 0),
            "height": metadata.get("height", 0),
            "size": metadata.get("size", 0),
            "file_hash": compute_file_hash(video_path)
        })
        await db.update_video(video_id, {"status": "ready"})
        if request.context:
            await db.update_video(video_id, {"context": request.context})

        return IngestResponse(video_id=video_id, status="ready", frames_processed=0)

    elif request.directory:
        # Directory — add all new videos to library, skip existing ones
        videos = find_video_files(request.directory)
        results = []

        for video_path in videos:
            existing = await db.get_video_by_path(str(video_path))
            if existing:
                results.append(existing.id)
                continue

            video_id = str(uuid.uuid4())
            metadata = get_video_metadata(video_path)
            await db.create_video({
                "id": video_id,
                "name": video_path.stem,
                "path": str(video_path),
                "duration": metadata.get("duration", 0),
                "width": metadata.get("width", 0),
                "height": metadata.get("height", 0),
                "size": metadata.get("size", 0),
                "file_hash": compute_file_hash(video_path)
            })
            await db.update_video(video_id, {"status": "ready"})
            if request.context:
                await db.update_video(video_id, {"context": request.context})
            results.append(video_id)

        return IngestResponse(video_id=", ".join(results), status="ready", frames_processed=0)

    else:
        raise HTTPException(status_code=400, detail="Must specify video_path or directory")


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_videos(q: str, limit: int = 10, threshold: float = 0.50, sources: str = 'both', expand: bool = False):
    """Search for videos by natural language query."""
    processed = query_processor.process_query(q)

    # When the UI has both sources selected, always use mixed — don't auto-detect
    # intent from the query phrasing, since the user wants all source types searched.
    if sources == 'video':
        source_intent = 'visual'
    elif sources == 'audio':
        source_intent = 'speech'
    else:
        source_intent = 'mixed'

    # The keyword boost below can raise scores by up to MAX_BOOST.
    # Pass a relaxed pre-filter to Qdrant so results that qualify *after*
    # boosting aren't cut before they ever get a chance to be boosted.
    MAX_BOOST = 1.25
    qdrant_threshold = max(0.0, threshold / MAX_BOOST)

    # Build list of queries to search: cleaned + original.
    # Note: Enhanced mode (expand=True) uses LLM re-ranking for precision rather than
    # query expansion — re-ranking is more accurate and avoids a second LLM round-trip.
    queries_to_search = list(dict.fromkeys([processed["search_terms"], q]))  # dedup, order-preserved

    # Run all queries in parallel; merge by Qdrant point ID keeping the max score.
    search_params = dict(limit=limit * 8, score_threshold=qdrant_threshold, source_intent=source_intent)
    all_result_lists = await asyncio.gather(*(
        embed_service.search(query=qv, **search_params) for qv in queries_to_search
    ))
    merged: dict = {}
    for result_list in all_result_lists:
        for r in result_list:
            hit_id = r.get("id")
            if hit_id not in merged or r["score"] > merged[hit_id]["score"]:
                merged[hit_id] = r
    results = list(merged.values())

    # Keyword overlap boost with prefix matching (handles plurals/conjugations):
    # "speeds" matches "speed", "drives" matches "drive", etc.
    import re as _re
    _STOPWORDS = {
        'a','an','the','and','or','but','in','on','at','to','for','of','with',
        'i','my','me','we','you','it','is','was','are','were','be','been',
        'have','had','has','do','did','does','about','that','this','which',
        'when','where','how','what','who','their','its','by','from','up',
        'specifically','really','actually','just','so','some','any',
    }
    key_terms = [
        w.lower() for w in _re.findall(r'\b[a-zA-Z]{3,}\b', processed["search_terms"])
        if w.lower() not in _STOPWORDS
    ]
    def _boost(caption: str) -> float:
        if not key_terms:
            return 1.0
        cap_lower = caption.lower()
        cap_words = set(_re.findall(r'\b[a-z]{3,}\b', cap_lower))
        hits = sum(
            1 for t in key_terms
            if t in cap_lower or any(
                cw.startswith(t[:5]) or t.startswith(cw[:5])
                for cw in cap_words if len(cw) >= 4 and len(t) >= 4
            )
        )
        return 1.0 + (MAX_BOOST - 1.0) * hits / len(key_terms)

    for r in results:
        r["score"] = r["score"] * _boost(r.get("caption", ""))

    # Group by video, collecting all raw hits
    video_hits: dict = {}
    for r in results:
        vid = r.get("video_id")
        if vid not in video_hits:
            video_hits[vid] = {"score": r.get("score", 0), "hits": []}
        else:
            video_hits[vid]["score"] = max(video_hits[vid]["score"], r.get("score", 0))
        video_hits[vid]["hits"].append(r)

    # Fetch video details and build segments per video
    all_segments: list[dict] = []
    video_meta: dict = {}
    for vid, data in video_hits.items():
        video = await db.get_video(vid)
        if not video:
            continue
        for h in data["hits"]:
            h["video_name"] = video.name
        segments = segment_retriever.group_segments(data["hits"], vid)
        video_meta[vid] = {
            "name": video.name, "path": video.path, "duration": video.duration,
        }
        for s in segments[:25]:
            all_segments.append({
                "video_id": s.video_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "caption": s.caption,
                "score": s.score,
                "frame_path": s.frame_path,
            })

    # Cross-encoder re-ranking when enhanced mode is on
    if expand:
        all_segments.sort(key=lambda s: s["score"], reverse=True)
        all_segments = await _rerank_segments(q, all_segments)

    # Re-group by video for the response
    video_segs: dict = {}
    for seg in all_segments:
        video_segs.setdefault(seg["video_id"], []).append(seg)

    # After re-ranking the blended scores are on a different scale than raw Qdrant
    # cosine similarity, so use a fixed floor instead of the user threshold.
    seg_threshold = 0.2 if expand else threshold

    final_results = []
    for vid, segs in video_segs.items():
        meta = video_meta.get(vid)
        if not meta:
            continue
        top_segments = [s for s in segs if s["score"] >= seg_threshold]
        if not top_segments:
            continue
        best_score = max(s["score"] for s in top_segments)
        final_results.append({
            "video_id": vid,
            "name": meta["name"],
            "path": meta["path"],
            "duration": meta["duration"],
            "score": best_score,
            "matching_frames": len(top_segments),
            "segments": top_segments,
        })

    # Only return videos where the best-matching segment is above threshold
    final_results = [r for r in final_results if r["score"] >= seg_threshold]

    # Sort by score descending, cap to limit
    final_results.sort(key=lambda x: x["score"], reverse=True)
    final_results = final_results[:limit]

    return SearchResponse(results=final_results, query=q)


@app.get("/search/segments", response_model=SegmentSearchResponse, tags=["Search"])
async def search_segments(video_id: str, q: str, limit: int = 20):
    """Search for specific segments within a video."""
    processed = query_processor.process_query(q)
    
    results = await embed_service.search(
        query=processed["search_terms"],
        limit=limit,
        video_id=video_id,
        source_intent=processed["source_intent"]
    )
    
    # Get video info
    video = await db.get_video(video_id)
    for r in results:
        r["video_name"] = video.name if video else ""
    
    # Group into segments
    segments = segment_retriever.group_segments(results, video_id)
    
    return SegmentSearchResponse(
        segments=[
            {
                "video_id": s.video_id,
                "video_name": s.video_name,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "caption": s.caption,
                "score": s.score,
                "frame_path": s.frame_path
            }
            for s in segments
        ],
        video_id=video_id,
        query=q
    )


def _video_to_response(v) -> VideoResponse:
    import json as _json
    tags = []
    try:
        tags = _json.loads(v.tags or '[]')
    except Exception:
        pass
    _status = getattr(v, 'status', 'processing')
    _updated_at = getattr(v, 'updated_at', None)
    _analyzed_at = _updated_at.isoformat() if _updated_at and _status in ('analyzed', 'retranscribing') else None
    return VideoResponse(
        id=v.id,
        name=v.name,
        display_name=v.display_name,
        path=v.path,
        duration=v.duration,
        summary=v.summary,
        tags=tags,
        status=_status,
        frame_count=getattr(v, 'frame_count', 0) or 0,
        thumbnail_path=getattr(v, 'thumbnail_path', None),
        context=getattr(v, 'context', None),
        analyzed_at=_analyzed_at,
        has_transcript=bool(getattr(v, 'has_transcript', 0)),
    )


@app.get("/videos", response_model=List[VideoResponse], tags=["Videos"])
async def list_videos(hidden: bool = False):
    """List videos. Pass ?hidden=true to list hidden videos instead."""
    videos = await db.get_all_videos(hidden=hidden)
    return [_video_to_response(v) for v in videos]


@app.get("/videos/{video_id}", response_model=VideoResponse, tags=["Videos"])
async def get_video(video_id: str):
    """Get details for a specific video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return _video_to_response(video)


@app.patch("/videos/{video_id}", response_model=VideoResponse, tags=["Videos"])
async def update_video(video_id: str, req: VideoUpdateRequest):
    """Update display name and/or tags for a video."""
    import json as _json
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    updates = {}
    if req.display_name is not None:
        updates['display_name'] = req.display_name
    if req.tags is not None:
        updates['tags'] = _json.dumps(req.tags)
    if req.thumbnail_path is not None:
        updates['thumbnail_path'] = req.thumbnail_path
    if req.context is not None:
        updates['context'] = req.context
    if updates:
        await db.update_video(video_id, updates)
    video = await db.get_video(video_id)
    return _video_to_response(video)


@app.delete("/videos/{video_id}", status_code=204, tags=["Videos"])
async def delete_video(video_id: str):
    """Hide a video from the library (soft delete). Use POST /videos/{id}/unhide to restore."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    await db.update_video(video_id, {'hidden': '1'})


@app.post("/videos/{video_id}/unhide", status_code=204, tags=["Videos"])
async def unhide_video(video_id: str):
    """Restore a hidden video back into the library."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    await db.update_video(video_id, {'hidden': '0'})


@app.get("/videos/{video_id}/frames", tags=["Videos"])
async def list_video_frames(video_id: str):
    """List all visual frames for a video (excludes transcript entries)."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    frames = await db.get_frames_for_video(video_id)
    visual = [
        {"id": f.id, "timestamp": f.timestamp, "caption": f.caption, "path": f.frame_path}
        for f in frames
        if f.frame_path and not (f.caption or '').startswith('[transcript]')
    ]
    return {"frames": visual, "video_id": video_id}


@app.get("/videos/{video_id}/thumbnail", tags=["Media"])
async def get_video_thumbnail(video_id: str):
    """Return thumbnail image for a video, falling back to extracted or on-demand frame."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    # 1. Explicit thumbnail already set
    if video.thumbnail_path:
        p = Path(video.thumbnail_path)
        if p.exists():
            return FileResponse(str(p), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})
    # 2. Middle frame from a completed analysis
    frames = await db.get_frames_for_video(video_id)
    visual = [f for f in frames if f.frame_path and not (f.caption or '').startswith('[transcript]')]
    if visual:
        frame = visual[len(visual) // 2]
        p = Path(frame.frame_path)
        if p.exists():
            return FileResponse(str(p), media_type="image/jpeg")
    # 3. On-demand preview: extract a single frame at 10% into the video via ffmpeg
    video_path = Path(video.path)
    if video_path.exists() and getattr(video, 'duration', 0) > 0:
        frame_dir = FRAMES_DIR / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        preview_path = str(frame_dir / "preview.jpg")
        seek = video.duration * 0.1
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, frame_extractor.extract_frame_at, str(video_path), seek, preview_path
        )
        if result and Path(result).exists():
            await db.update_video(video_id, {'thumbnail_path': result})
            return FileResponse(result, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No thumbnail available")


@app.get("/videos/{video_id}/transcript", tags=["Analysis"])
async def get_video_transcript(video_id: str):
    """Return all Whisper transcript segments for a video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    frames = await db.get_frames_for_video(video_id)
    segments = [
        {"id": f.id, "timestamp": f.timestamp, "text": (f.caption or '').replace('[transcript] ', '', 1)}
        for f in sorted(frames, key=lambda x: x.timestamp)
        if (f.caption or '').startswith('[transcript]')
    ]
    return {"segments": segments, "video_id": video_id, "count": len(segments)}


@app.patch("/videos/{video_id}/frames/{frame_id}", tags=["Analysis"])
async def update_frame_caption(video_id: str, frame_id: str, req: FrameUpdateRequest):
    """Update a frame's caption and re-embed in the background."""
    frame = await db.get_frame(frame_id)
    if not frame or frame.video_id != video_id:
        raise HTTPException(status_code=404, detail="Frame not found")
    is_transcript = (frame.caption or '').startswith('[transcript]')
    new_caption = req.caption.strip()
    if is_transcript and not new_caption.startswith('[transcript]'):
        new_caption = '[transcript] ' + new_caption
    await db.update_frame(frame_id, {'caption': new_caption})
    # Re-embed in background so the search index reflects the edit
    if frame.embedding_id:
        asyncio.create_task(embed_service.index_frame(
            video_id, frame.frame_path, frame.timestamp, new_caption,
            _point_id=frame.embedding_id
        ))
    return {"id": frame_id, "caption": req.caption.strip(), "ok": True}


@app.get("/frames/{video_id}/{filename}", tags=["Media"])
async def serve_frame(video_id: str, filename: str):
    """Serve a frame image file."""
    # Sanitize to prevent path traversal
    safe_name = Path(filename).name
    frame_path = FRAMES_DIR / video_id / safe_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.get("/videos/{video_id}/stream", tags=["Media"])
async def stream_video(video_id: str):
    """Stream the original video file with range request support for seeking."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    # Prevent path traversal — only serve files under /data/videos
    try:
        video_path.resolve().relative_to(Path("/data/videos").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    media_types = {
        ".mp4": "video/mp4", ".m4v": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    media_type = media_types.get(video_path.suffix.lower(), "video/mp4")
    return FileResponse(
        str(video_path),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/videos/{video_id}/download", tags=["Media"])
async def download_video(video_id: str):
    """Download the original video file."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    try:
        video_path.resolve().relative_to(Path("/data/videos").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(
        str(video_path),
        filename=video_path.name,
        headers={"Content-Disposition": f'attachment; filename="{video_path.name}"'},
    )


@app.get("/videos/{video_id}/clip", tags=["Media"])
async def download_clip(video_id: str, start: float, end: float, background_tasks: BackgroundTasks):
    """Extract and download a clip using ffmpeg stream-copy (no re-encode)."""
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be greater than start")
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    try:
        video_path.resolve().relative_to(Path("/data/videos").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    suffix = video_path.suffix or '.mp4'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        clip_path = f.name
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', str(video_path),
        '-t', str(end - start),
        '-c', 'copy',
        clip_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        background_tasks.add_task(os.unlink, clip_path)
        raise HTTPException(status_code=500, detail="Clip extraction timed out")
    if proc.returncode != 0:
        background_tasks.add_task(os.unlink, clip_path)
        raise HTTPException(status_code=500, detail="Clip extraction failed")
    clip_name = f"{video_path.stem}_{int(start)}-{int(end)}{suffix}"
    background_tasks.add_task(os.unlink, clip_path)
    return FileResponse(clip_path, filename=clip_name,
                        headers={"Content-Disposition": f'attachment; filename="{clip_name}"'})


class AnalyzeRequest(BaseModel):
    context: Optional[str] = None
    resume: bool = False

ReprocessRequest = AnalyzeRequest  # backward-compat alias


async def _run_analyze(video_id: str, req: AnalyzeRequest) -> dict:
    """Shared logic: start full analysis (or resume a partial one)."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    updates: dict = {'status': 'processing'}
    if not req.resume:
        updates['summary'] = None
    if req.context is not None:
        updates['context'] = req.context
    await db.update_video(video_id, updates)
    context = req.context or getattr(video, 'context', '') or ''
    _start_analysis(video.path, video_id, context=context, resume=req.resume)
    return {"status": "processing", "video_id": video_id}


@app.post("/analyze/{video_id}", tags=["Analysis"])
async def analyze_video(video_id: str, req: AnalyzeRequest = AnalyzeRequest()):
    """Start (or restart) full vision + speech analysis for a video.

    Works for any video status: `ready`, `analyzed`, `error`, etc.
    Clears any existing frames/embeddings and re-runs from scratch.
    Optionally supply a `context` string to guide caption generation.
    """
    return await _run_analyze(video_id, req)


@app.post("/reprocess/{video_id}", tags=["Analysis"])
async def reprocess_video(video_id: str, req: AnalyzeRequest = AnalyzeRequest()):
    """Alias for `POST /analyze/{video_id}` — kept for backward compatibility."""
    return await _run_analyze(video_id, req)


async def _retranscribe_audio(video_id: str, video_path: str):
    """Replace only the transcript segments for a video, leaving visual frames intact."""
    try:
        # Remove existing transcript frames from Qdrant
        frames = await db.get_frames_for_video(video_id)
        transcript_frames = [f for f in frames if f.frame_path == "" and f.embedding_id]
        if transcript_frames:
            from qdrant_client.models import HasIdCondition
            try:
                embed_service.qdrant_client.delete(
                    collection_name=embed_service.collection_name,
                    points_selector=HasIdCondition(has_id=[f.embedding_id for f in transcript_frames]),
                )
            except Exception:
                pass
        # Remove from DB
        await db.delete_transcript_frames_for_video(video_id)

        # Re-transcribe
        loop = asyncio.get_running_loop()
        segments = await loop.run_in_executor(
            None, transcription_service.transcribe_video, video_path
        )
        print(f"Retranscribe: {len(segments)} segment(s) for {video_id}")

        for seg in segments:
            text = f"[transcript] {seg.text}"
            await embed_service.index_frame(
                video_id=video_id, frame_path="", timestamp=seg.start, caption=text
            )
            seg_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_transcript_{seg.start}"))
            await db.create_frame({
                "id": seg_id, "video_id": video_id,
                "timestamp": seg.start, "frame_path": "",
                "caption": text, "embedding_id": seg_id
            })

        # Sliding windows
        WINDOW_SIZE = 3
        WINDOW_STEP = 2
        for i in range(0, len(segments) - 1, WINDOW_STEP):
            win = segments[i:i + WINDOW_SIZE]
            if len(win) < 2:
                continue
            combined = " ".join(s.text.strip() for s in win if s.text.strip())
            if not combined:
                continue
            win_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                f"{video_id}_win_{win[0].start:.3f}_{win[-1].end:.3f}"))
            await embed_service.index_frame(
                video_id=video_id, frame_path="", timestamp=win[0].start,
                caption=f"[transcript] {combined}", _point_id=win_id,
            )

        await db.update_video(video_id, {'status': 'analyzed', 'has_transcript': 1})
    except asyncio.CancelledError:
        await db.update_video(video_id, {'status': 'analyzed'})
        raise
    except Exception as e:
        print(f"Retranscription error for {video_id}: {e}")
        await db.update_video(video_id, {'status': 'error'})
    finally:
        _active_tasks.pop(video_id, None)


@app.post("/retranscribe/{video_id}", tags=["Analysis"])
async def retranscribe_video(video_id: str):
    """Re-run Whisper transcription only, leaving visual frames intact."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status == 'processing':
        raise HTTPException(status_code=409, detail="Full analysis in progress — stop it first")
    # Cancel any existing (possibly orphaned) retranscription task
    existing = _active_tasks.pop(video_id, None)
    if existing and not existing.done():
        existing.cancel()
    await db.update_video(video_id, {'status': 'retranscribing'})
    task = asyncio.create_task(_retranscribe_audio(video_id, video.path))
    _active_tasks[video_id] = task
    return {"status": "retranscribing", "video_id": video_id}


@app.post("/cancel/{video_id}", tags=["Analysis"])
async def cancel_analysis(video_id: str):
    """Cancel an in-progress analysis or retranscription for a video."""
    video = await db.get_video(video_id)
    if not video or video.status not in ('processing', 'transcribing', 'retranscribing'):
        raise HTTPException(status_code=409, detail="No active analysis for this video")
    reset_status = 'error' if video.status in ('processing', 'transcribing') else 'analyzed'
    task = _active_tasks.pop(video_id, None)
    if task and not task.done():
        task.cancel()
    else:
        # Orphaned status (container restart killed the task) — reset directly
        await db.update_video(video_id, {'status': reset_status})
    return {"status": "cancelled", "video_id": video_id}


@app.post("/summarize/{video_id}", tags=["Analysis"])
async def summarize_video(video_id: str, background_tasks: BackgroundTasks):
    """Generate or regenerate a summary for a video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    async def generate_summary():
        try:
            captions = await db.get_captions_for_video(video_id)
            summary = await summarizer_service.summarize_video(captions, video.name)
            await db.update_video_summary(video_id, summary)
        except Exception as e:
            print(f"Summary generation failed for {video_id}: {e}")
    
    background_tasks.add_task(generate_summary)
    
    return {"status": "generating", "video_id": video_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)