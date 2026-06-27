from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import asyncio
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
    await db.init_db()
    await embed_service.ensure_collection()
    stuck = [v for v in await db.get_all_videos() if v.status == 'processing']
    for v in stuck:
        await db.update_video(v.id, {'frame_count': 0})
        _start_analysis(v.path, v.id, context=getattr(v, 'context', '') or '')
    asyncio.create_task(_auto_import())
    yield

app = FastAPI(title="Local Video Search & Summarization", lifespan=lifespan)

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


class VideoUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    tags: Optional[List[str]] = None
    thumbnail_path: Optional[str] = None
    context: Optional[str] = None


class SegmentSearchResponse(BaseModel):
    segments: List[dict]
    video_id: str
    query: str


# Track in-flight tasks by video_id so we can cancel on re-trigger
_active_tasks: dict = {}


def _start_analysis(video_path: str, video_id: str, context: str = '') -> None:
    """Cancel any existing task for this video, then start a fresh one."""
    existing = _active_tasks.get(video_id)
    if existing and not existing.done():
        existing.cancel()
    task = asyncio.create_task(process_video(video_path, video_id, context=context))
    _active_tasks[video_id] = task

# Background tasks
async def process_video(video_path: str, video_id: str, context: str = ''):
    """Process a video: extract frames, caption, transcribe audio, and index."""
    try:
        await db.update_video(video_id, {'status': 'processing', 'frame_count': 0})
        async with _analysis_semaphore:
            # Clear any leftover frame records and embeddings from a previous (failed/partial) run
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
                timestamp = i * frame_extractor.interval
                caption = await vlm_service.caption_frame(frame_path, context=context)
                await embed_service.index_frame(
                    video_id=video_id, frame_path=frame_path,
                    timestamp=timestamp, caption=caption
                )
                frame_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_frame_{i}"))
                await db.create_frame({
                    "id": frame_id, "video_id": video_id,
                    "timestamp": timestamp, "frame_path": frame_path,
                    "caption": caption, "embedding_id": frame_id
                })
                await db.update_video(video_id, {'frame_count': i + 1})

            # Set first frame as default thumbnail if none set
            video_check = await db.get_video(video_id)
            if frames and not video_check.thumbnail_path:
                await db.update_video(video_id, {'thumbnail_path': str(frames[0])})

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

            await db.update_video(video_id, {'status': 'analyzed'})

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


async def _auto_import():
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


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reimport")
async def reimport():
    """Scan all configured video library directories and add any new videos."""
    await _auto_import()
    videos = await db.get_all_videos()
    return {"total": len(videos)}


@app.get("/status/models")
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
        probe(cfg["vllm_vlm_url"], "Vision (VLM)"),
        probe(cfg["vllm_url"], "Summarizer"),
    )
    return {"services": list(results)}


@app.get("/status/qdrant")
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


@app.post("/status/qdrant/purge-orphans")
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


@app.get("/settings")
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


@app.put("/settings")
async def update_settings_endpoint(req: SettingsUpdateRequest):
    """Persist settings overrides and reinitialize services."""
    override = _load_override()
    for field, val in req.model_dump(exclude_none=True).items():
        override[field] = val
    SETTINGS_OVERRIDE_PATH.write_text(json.dumps(override, indent=2))
    cfg = _effective_cfg()
    _reinit_services(cfg)
    asyncio.create_task(_sync_blacklist())
    return cfg



@app.get("/browse")
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


@app.post("/ingest", response_model=IngestResponse)
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


@app.get("/search", response_model=SearchResponse)
async def search_videos(q: str, limit: int = 10, threshold: float = 0.50, sources: str = 'both'):
    """Search for videos by natural language query."""
    processed = query_processor.process_query(q)
    # UI source filter overrides auto-detected intent
    if sources == 'video':
        processed['source_intent'] = 'visual'
    elif sources == 'audio':
        processed['source_intent'] = 'speech'

    # Fetch more raw results so we have enough to group and show segments
    results = await embed_service.search(
        query=processed["search_terms"],
        limit=limit * 5,
        score_threshold=threshold,
        source_intent=processed["source_intent"]
    )

    # Group by video, collecting all raw hits
    video_hits: dict = {}
    for r in results:
        vid = r.get("video_id")
        if vid not in video_hits:
            video_hits[vid] = {"score": r.get("score", 0), "hits": []}
        else:
            video_hits[vid]["score"] = max(video_hits[vid]["score"], r.get("score", 0))
        video_hits[vid]["hits"].append(r)

    # Fetch video details and build top-N segments per video
    final_results = []
    for vid, data in video_hits.items():
        video = await db.get_video(vid)
        if not video:
            continue
        for h in data["hits"]:
            h["video_name"] = video.name
        segments = segment_retriever.group_segments(data["hits"], vid)
        top_segments = [
            {
                "video_id": s.video_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "caption": s.caption,
                "score": s.score,
                "frame_path": s.frame_path,
            }
            for s in segments[:5]
        ]
        final_results.append({
            "video_id": vid,
            "name": video.name,
            "path": video.path,
            "duration": video.duration,
            "score": data["score"],
            "matching_frames": len(segments),
            "segments": top_segments,
        })

    # Only return videos where the best-matching segment is above threshold
    final_results = [r for r in final_results if r["score"] >= threshold]

    # Sort by score descending, cap to limit
    final_results.sort(key=lambda x: x["score"], reverse=True)
    final_results = final_results[:limit]

    return SearchResponse(results=final_results, query=q)


@app.get("/search/segments", response_model=SegmentSearchResponse)
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
    return VideoResponse(
        id=v.id,
        name=v.name,
        display_name=v.display_name,
        path=v.path,
        duration=v.duration,
        summary=v.summary,
        tags=tags,
        status=getattr(v, 'status', 'processing'),
        frame_count=getattr(v, 'frame_count', 0) or 0,
        thumbnail_path=getattr(v, 'thumbnail_path', None),
        context=getattr(v, 'context', None),
    )


@app.get("/videos", response_model=List[VideoResponse])
async def list_videos(hidden: bool = False):
    """List videos. Pass ?hidden=true to list hidden videos instead."""
    videos = await db.get_all_videos(hidden=hidden)
    return [_video_to_response(v) for v in videos]


@app.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(video_id: str):
    """Get details for a specific video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return _video_to_response(video)


@app.patch("/videos/{video_id}", response_model=VideoResponse)
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


@app.delete("/videos/{video_id}", status_code=204)
async def delete_video(video_id: str):
    """Hide a video from the library (soft delete). Use POST /videos/{id}/unhide to restore."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    await db.update_video(video_id, {'hidden': '1'})


@app.post("/videos/{video_id}/unhide", status_code=204)
async def unhide_video(video_id: str):
    """Restore a hidden video back into the library."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    await db.update_video(video_id, {'hidden': '0'})


@app.get("/videos/{video_id}/frames")
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


@app.get("/videos/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: str):
    """Return thumbnail image for a video, falling back to the middle extracted frame."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.thumbnail_path:
        p = Path(video.thumbnail_path)
        if p.exists():
            return FileResponse(str(p), media_type="image/jpeg")
    frames = await db.get_frames_for_video(video_id)
    visual = [f for f in frames if f.frame_path and not (f.caption or '').startswith('[transcript]')]
    if visual:
        frame = visual[len(visual) // 2]
        p = Path(frame.frame_path)
        if p.exists():
            return FileResponse(str(p), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No thumbnail available")


@app.get("/videos/{video_id}/transcript")
async def get_video_transcript(video_id: str):
    """Return all Whisper transcript segments for a video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    frames = await db.get_frames_for_video(video_id)
    segments = [
        {"timestamp": f.timestamp, "text": (f.caption or '').replace('[transcript] ', '', 1)}
        for f in sorted(frames, key=lambda x: x.timestamp)
        if (f.caption or '').startswith('[transcript]')
    ]
    return {"segments": segments, "video_id": video_id, "count": len(segments)}


@app.get("/frames/{video_id}/{filename}")
async def serve_frame(video_id: str, filename: str):
    """Serve a frame image file."""
    # Sanitize to prevent path traversal
    safe_name = Path(filename).name
    frame_path = FRAMES_DIR / video_id / safe_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.get("/videos/{video_id}/stream")
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


class ReprocessRequest(BaseModel):
    context: Optional[str] = None


@app.post("/reprocess/{video_id}")
async def reprocess_video(video_id: str, req: ReprocessRequest = ReprocessRequest()):
    """Reset and re-run full analysis for a video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    # Clear existing Qdrant vectors
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        embed_service.qdrant_client.delete(
            collection_name=embed_service.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
        )
    except Exception:
        pass
    updates = {'frame_count': 0, 'status': 'processing', 'summary': None}
    if req.context is not None:
        updates['context'] = req.context
    await db.update_video(video_id, updates)
    context = req.context or getattr(video, 'context', '') or ''
    _start_analysis(video.path, video_id, context=context)
    return {"status": "processing", "video_id": video_id}


@app.post("/cancel/{video_id}")
async def cancel_analysis(video_id: str):
    """Cancel an in-progress analysis for a video."""
    task = _active_tasks.get(video_id)
    if not task or task.done():
        raise HTTPException(status_code=409, detail="No active analysis for this video")
    task.cancel()
    return {"status": "cancelled", "video_id": video_id}


@app.post("/summarize/{video_id}")
async def summarize_video(video_id: str, background_tasks: BackgroundTasks):
    """Generate or regenerate a summary for a video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    async def generate_summary():
        captions = await db.get_captions_for_video(video_id)
        summary = await summarizer_service.summarize_video(captions, video.name)
        await db.update_video_summary(video_id, summary)
    
    background_tasks.add_task(generate_summary)
    
    return {"status": "generating", "video_id": video_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)