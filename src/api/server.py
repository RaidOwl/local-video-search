from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timezone
import asyncio
import os
import random
import re
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
from ..search.metadata_filter import build_filters, video_matches, describe_metadata
from ..summarize.video_summarizer import SummarizationService
from ..transcribe.transcriber import TranscriptionService
from ..fusion import FusionService
from ..search.llm_rerank import LLMReranker

# Stopwords stripped before computing the keyword-overlap boost in /search.
_SEARCH_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'i', 'my', 'me', 'we', 'you', 'it', 'is', 'was', 'are', 'were', 'be', 'been',
    'have', 'had', 'has', 'do', 'did', 'does', 'about', 'that', 'this', 'which',
    'when', 'where', 'how', 'what', 'who', 'their', 'its', 'by', 'from', 'up',
    'specifically', 'really', 'actually', 'just', 'so', 'some', 'any',
}

# Words that must never be chosen as a distinctive cross-modal "anchor" (the rare
# locating noun like robot/museum/kvm), even when they are statistically rare in
# the caption corpus. VLM captions describe actions with a narrow vocabulary
# ("gesturing", "speaking") and almost never write "talking", so a user's generic
# verb like "talking" looks rarer than a real object like "robot" and would
# hijack the anchor. These are the "how/where someone is" words — they describe
# the SCENE, not the THING to locate — so they are excluded from anchoring (but
# still used everywhere else in search). Stored as base/gerund/3rd-person/past
# variants since users type any form.
_ANCHOR_STOPLIST = {
    'talk', 'talks', 'talking', 'talked', 'speak', 'speaks', 'speaking', 'spoke',
    'say', 'says', 'saying', 'said', 'discuss', 'discusses', 'discussing',
    'explain', 'explains', 'explaining', 'tell', 'tells', 'telling', 'told',
    'show', 'shows', 'showing', 'showed', 'look', 'looks', 'looking', 'looked',
    'watch', 'watches', 'watching', 'see', 'sees', 'seeing', 'seen',
    'stand', 'stands', 'standing', 'stood', 'sit', 'sits', 'sitting', 'sat',
    'walk', 'walks', 'walking', 'walked', 'hold', 'holds', 'holding', 'held',
    'use', 'uses', 'using', 'used', 'wear', 'wears', 'wearing', 'wore',
    'play', 'plays', 'playing', 'played', 'work', 'works', 'working', 'worked',
    'run', 'runs', 'running', 'move', 'moves', 'moving', 'moved',
    'point', 'points', 'pointing', 'gesture', 'gestures', 'gesturing',
    'make', 'makes', 'making', 'made', 'doing', 'going', 'getting', 'being',
    'person', 'people', 'man', 'men', 'woman', 'women', 'guy', 'guys',
    'someone', 'somebody', 'thing', 'things', 'something', 'stuff',
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _auto_import_task
    await db.init_db()
    await embed_service.ensure_collection()
    stuck = [v for v in await db.get_all_videos() if v.status in ('processing', 'transcribing', 'fusing')]
    for v in stuck:
        await db.update_video(v.id, {'frame_count': 0})
        _start_analysis(v.path, v.id, context=getattr(v, 'context', '') or '')
    # Restart any retranscription that was interrupted
    stuck_audio = [v for v in await db.get_all_videos() if v.status == 'retranscribing']
    for v in stuck_audio:
        task = asyncio.create_task(_retranscribe_audio(v.id, v.path))
        _active_tasks[v.id] = task
    _auto_import_task = asyncio.create_task(_auto_import_loop())
    # Warm the reranker after a short delay (see _warm_reranker) so the API can
    # pass its healthcheck before the GIL-heavy model load stalls the event loop.
    asyncio.create_task(_warm_reranker())
    # Backfill file metadata for videos imported before those columns existed.
    asyncio.create_task(_backfill_metadata())
    # Tag Qdrant points with is_transcript so the Video/Audio source filter works.
    asyncio.create_task(_backfill_source_flags())
    # Correct has_frames/has_transcript for videos analyzed before the has_frames
    # column existed (its migration blanket-defaulted every row to 1).
    asyncio.create_task(_backfill_modality_flags())
    # Populate the lexical (BM25/FTS5) index from existing Qdrant payloads so
    # hybrid search works on a library indexed before this feature.
    asyncio.create_task(_backfill_segment_fts())
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
            "embed_url":   settings.embed_url,
            "vlm_url":     settings.vlm_url,
            "llm_url":     settings.llm_url,
            "qdrant_url":       settings.qdrant_url,
            "embed_model": settings.embed_model,
            "vlm_model":   settings.vlm_model,
            "llm_model":       settings.llm_model,
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
        'embed_url', 'vlm_url', 'llm_url', 'qdrant_url',
        'embed_model', 'vlm_model', 'llm_model',
    }
    override = _load_override()
    if ai_backend != "custom":
        override = {k: v for k, v in override.items() if k not in _URL_MODEL_KEYS}
    base.update(override)
    base.setdefault("max_concurrent_jobs", 8)
    base.setdefault("transcription_concurrency", 2)
    # FFmpeg hardware acceleration — "auto" probes the best backend on the host.
    base.setdefault("ffmpeg_hwaccel", settings.ffmpeg_hwaccel)
    # Whisper decode tuning: greedy (beam 1) + no previous-text conditioning is the
    # fast default; both are override-tunable from the Settings file.
    base.setdefault("whisper_beam_size", settings.whisper_beam_size)
    base.setdefault("whisper_condition_on_previous", settings.whisper_condition_on_previous)
    # LLM compositional precision re-rank of the search shortlist (final precision
    # stage). On by default; override-tunable to disable if LLM latency is unwanted.
    base.setdefault("llm_rerank", True)
    # CPU thread cap for ffmpeg: 0 = auto (half cores), keeps host responsive.
    cores = os.cpu_count() or 4
    base["cpu_cores"] = cores
    threads = int(base.get("ffmpeg_threads", settings.ffmpeg_threads) or 0)
    if threads <= 0:
        threads = max(1, cores // 2)
    base["ffmpeg_threads"] = max(1, min(threads, cores))
    # Automation: .env values seed the defaults; the override file (Settings UI) wins.
    if "import_blacklist" not in base:
        base["import_blacklist"] = [p.strip() for p in settings.import_blacklist.split(",") if p.strip()]
    base.setdefault("auto_import_interval", settings.auto_import_interval)
    base.setdefault("auto_analyze", settings.auto_analyze)
    base.setdefault("default_context", settings.default_context)
    # Normalize auto_analyze: legacy bool overrides -> mode string; validate.
    aa = base.get("auto_analyze")
    if isinstance(aa, bool):
        aa = "both" if aa else "off"
    if aa not in ("off", "audio", "video", "both"):
        aa = "off"
    base["auto_analyze"] = aa
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
            "Use HARDWARE_TYPE=apple instead — install Ollama on your Mac "
            "and it will use Metal automatically."
        )
    if backend == "ollama" and is_apple_silicon:
        return (
            "Ollama in Docker loses Metal acceleration on Apple Silicon. "
            "Use HARDWARE_TYPE=apple for full GPU performance."
        )
    return None

_cfg = _effective_cfg()
_analysis_semaphore = asyncio.Semaphore(int(_cfg.get("max_concurrent_jobs", 8)))
# Whisper runs on CPU and already uses all cores per job, so concurrent
# transcriptions oversubscribe the CPU and slow each other down. Cap it
# separately from vision so frame analysis can stay wide while audio stays narrow.
_transcribe_semaphore = asyncio.Semaphore(max(1, int(_cfg.get("transcription_concurrency", 2))))

# Services
embed_service = EmbeddingService(_cfg["embed_url"], _cfg["qdrant_url"], model=_cfg["embed_model"], text_index=db)
vlm_service = VLMService(_cfg["vlm_url"], model=_cfg["vlm_model"])
summarizer_service = SummarizationService(_cfg["llm_url"], model=_cfg["llm_model"])
# Fusion (3rd analysis pass) reuses the summarizer's text LLM endpoint.
fusion_service = FusionService(_cfg["llm_url"], model=_cfg["llm_model"])
# Compositional precision re-ranker (final search-precision stage). Reuses the
# text LLM endpoint; gated on per search by `llm_rerank` config + query shape.
llm_reranker = LLMReranker(_cfg["llm_url"], model=_cfg["llm_model"])
_llm_rerank_on = bool(_cfg.get("llm_rerank", True))
transcription_service = TranscriptionService(
    model_size=_cfg["whisper_model"],
    beam_size=int(_cfg.get("whisper_beam_size", 1)),
    condition_on_previous_text=bool(_cfg.get("whisper_condition_on_previous", False)),
)
query_processor = QueryProcessor()
segment_retriever = SegmentRetriever()

# Cross-encoder re-ranker — warmed at startup, always applied for ranking precision
_reranker = None

# Frame extractor
frame_extractor = FrameExtractor(
    interval=_cfg["frame_extraction_interval"],
    threads=int(_cfg.get("ffmpeg_threads", 0)),
    hwaccel=_cfg.get("ffmpeg_hwaccel", "off"),
)

def _reinit_services(cfg: dict):
    global embed_service, vlm_service, summarizer_service, fusion_service, llm_reranker, _llm_rerank_on, frame_extractor, transcription_service, _analysis_semaphore, _transcribe_semaphore
    embed_service = EmbeddingService(cfg["embed_url"], cfg["qdrant_url"], model=cfg["embed_model"], text_index=db)
    vlm_service = VLMService(cfg["vlm_url"], model=cfg["vlm_model"])
    summarizer_service = SummarizationService(cfg["llm_url"], model=cfg["llm_model"])
    fusion_service = FusionService(cfg["llm_url"], model=cfg["llm_model"])
    llm_reranker = LLMReranker(cfg["llm_url"], model=cfg["llm_model"])
    _llm_rerank_on = bool(cfg.get("llm_rerank", True))
    transcription_service = TranscriptionService(
        model_size=cfg["whisper_model"],
        beam_size=int(cfg.get("whisper_beam_size", 1)),
        condition_on_previous_text=bool(cfg.get("whisper_condition_on_previous", False)),
    )
    frame_extractor = FrameExtractor(
        interval=int(cfg["frame_extraction_interval"]),
        threads=int(cfg.get("ffmpeg_threads", 0)),
        hwaccel=cfg.get("ffmpeg_hwaccel", "off"),
    )
    _analysis_semaphore = asyncio.Semaphore(int(cfg.get("max_concurrent_jobs", 8)))
    _transcribe_semaphore = asyncio.Semaphore(max(1, int(cfg.get("transcription_concurrency", 2))))

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
    has_frames: bool = True
    transcribe_progress: float = 0.0
    fuse_progress: float = 0.0


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


def _load_reranker():
    """Lazily load the bge-reranker-base cross-encoder (~1.1 GB, CPU).

    Returns the model, or None if loading fails.  Weights are cached in
    /data/hf_cache so they survive container rebuilds.
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("Loading reranker model (BAAI/bge-reranker-base, ~1.1 GB)...")
            _reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
            print("Reranker ready.")
        except Exception as e:
            print(f"Reranker: failed to load: {e}")
            return None
    return _reranker


async def _warm_reranker() -> None:
    """Pre-load the reranker shortly after startup so the first search isn't slow.

    The load is delayed so the container can pass its healthcheck first. Building
    the cross-encoder does enough pure-Python (transformers/torch) work to hold
    the GIL, which starves the asyncio event loop even from a thread executor —
    so while it runs, the trivial ``/health`` endpoint stalls. On a fresh deploy
    the first run also downloads ~1.1 GB, which can take minutes. Warming
    immediately would keep ``/health`` unresponsive long enough to exceed the
    healthcheck ``start_period``, marking the API unhealthy and aborting any
    ``up`` where the frontend waits on ``service_healthy``. Delaying lets the API
    report healthy first; the brief warmup stall afterward is harmless because
    dependents have already started.
    """
    try:
        await asyncio.sleep(30)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_reranker)
    except Exception as e:
        print(f"Reranker warmup failed: {e}")


async def _rerank_segments(query, segments: list[dict]) -> list[dict]:
    """Re-rank candidate segments with a local cross-encoder and assign each an
    absolute relevance score.

    ``query`` may be a single string or a list of query formulations. When given
    several formulations, each segment keeps the *maximum* relevance across them
    (max-pooling) — this lets a terse query like "dog" be judged both literally
    and as "a video of a dog" without the templated form ever lowering a score
    that the natural form judged higher.

    Architecture: bi-encoder (BGE-M3 / Qdrant) retrieves candidates for recall;
    this cross-encoder (BAAI/bge-reranker-base, ~1.1 GB, CPU) judges true
    query↔caption relevance for precision.  bge-reranker is the same family as
    the BGE-M3 embeddings and is trained for natural-language semantic relevance,
    so it handles descriptive VLM captions and nuances like gender far better
    than passage-ranking models such as ms-marco-MiniLM.  The cross-encoder is
    the *sole* authority on the final score — we deliberately do NOT blend in the
    Qdrant cosine, because blending re-injects the bi-encoder's mistakes.

    The model's raw logit is calibrated: positive ⇒ relevant, negative ⇒ not.
    A plain sigmoid maps it to a stable 0–1 relevance probability that is
    consistent across queries (the same clip scores ~the same every search) and
    is NOT renormalized so the top hit is never artificially forced to 100%.

    Lazy-loaded on first call; weights cached in /data/hf_cache so they survive
    container rebuilds.  Falls back to original Qdrant scores on any failure.
    """
    if not segments:
        return segments

    if _load_reranker() is None:
        return segments

    try:
        import asyncio as _asyncio, math as _math
        queries = [query] if isinstance(query, str) else [q for q in query if q]
        captions = [
            s.get("caption", "").replace("[transcript] ", "").strip()
            for s in segments
        ]
        # Run synchronous predict in a thread so we don't block the event loop.
        # With multiple query formulations, keep the max raw logit per segment.
        loop = _asyncio.get_running_loop()
        best_raw = [float("-inf")] * len(segments)
        for qv in queries:
            pairs = [(qv, cap) for cap in captions]
            raw = await loop.run_in_executor(None, _reranker.predict, pairs)
            for i, r in enumerate(raw):
                rf = float(r)
                if rf > best_raw[i]:
                    best_raw[i] = rf

        # Short-segment damp: a 1-2 word caption/transcript line (e.g. "Micro
        # Center.") can ride a single domain-adjacent word to a moderate positive
        # logit (~0.6) even when it isn't a real answer — with almost no context
        # the cross-encoder can't distinguish a true match from topical
        # adjacency. Scale down *positive* logits for very short segments so weak
        # adjacency falls below the relevance threshold while genuine matches
        # (logit ~1.0) stay comfortably above it. Factor: 0→0.4, 1→0.55, 2→0.70,
        # 3→0.85, ≥4→1.0 (no change).
        import re as _re
        def _content_words(text: str) -> int:
            return len([w for w in _re.findall(r"[A-Za-z0-9']+", text) if len(w) > 1])
        for i, cap in enumerate(captions):
            if best_raw[i] > 0:
                n = _content_words(cap)
                if n < 4:
                    best_raw[i] *= min(1.0, 0.4 + 0.15 * n)

        def _sigmoid(x: float) -> float:
            # bge-reranker logits are calibrated (>0 relevant, <0 not) but small
            # in magnitude on descriptive captions, so a plain sigmoid bunches
            # everything near 0.5.  A temperature spreads the useful range so a
            # strong match lands ~0.65-0.8 (reads "near exact") while weak matches
            # stay below ~0.55, keeping the relevance threshold meaningful.
            x = max(-30.0, min(30.0, x * 1.5))
            return 1.0 / (1.0 + _math.exp(-x))

        for seg, raw in zip(segments, best_raw):
            seg["score"] = _sigmoid(raw)

        segments.sort(key=lambda s: s["score"], reverse=True)
        print(f"Re-rank {queries}: top={[round(s['score'], 3) for s in segments[:8]]}")
        return segments

    except Exception as e:
        print(f"Re-rank predict failed: {e}")
        return segments

def _start_analysis(video_path: str, video_id: str, context: str = '', resume: bool = False,
                    do_visual: bool = True, do_audio: bool = True, do_fuse: bool = True) -> None:
    """Cancel any existing task for this video, then start a fresh one."""
    existing = _active_tasks.get(video_id)
    if existing and not existing.done():
        existing.cancel()
    task = asyncio.create_task(process_video(
        video_path, video_id, context=context, resume=resume,
        do_visual=do_visual, do_audio=do_audio, do_fuse=do_fuse,
    ))
    _active_tasks[video_id] = task


def _make_transcribe_progress_cb(video_id: str, loop):
    """Build a thread-safe callback that records transcription progress (0..1) on
    the video row. Runs from the Whisper worker thread, so DB writes are
    scheduled back onto the event loop, throttled to ~1% steps."""
    state = {'last': 0.0}

    def _cb(done: float, total: float):
        frac = max(0.0, min(1.0, done / total)) if total else 0.0
        if frac - state['last'] >= 0.01 or frac >= 1.0:
            state['last'] = frac
            asyncio.run_coroutine_threadsafe(
                db.update_video(video_id, {'transcribe_progress': frac}), loop
            )

    return _cb


def _delete_video_points(video_id: str, is_transcript: Optional[bool] = None,
                         fused: Optional[bool] = None) -> None:
    """Delete a video's Qdrant points. ``is_transcript=None`` removes all of them;
    True removes only transcript points; False removes only visual points.
    ``fused=True`` restricts the deletion to fused 3rd-pass points."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        must = [FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        if is_transcript is not None:
            must.append(FieldCondition(key="is_transcript", match=MatchValue(value=is_transcript)))
        if fused is not None:
            must.append(FieldCondition(key="fused", match=MatchValue(value=fused)))
        embed_service.qdrant_client.delete(
            collection_name=embed_service.collection_name,
            points_selector=Filter(must=must),
        )
    except Exception:
        pass


async def _purge_points(video_id: str, is_transcript: Optional[bool] = None,
                        fused: Optional[bool] = None) -> None:
    """Delete a video's points from BOTH the dense (Qdrant) and lexical (FTS5)
    indexes, keeping the two recall arms in sync. Same filter semantics as
    _delete_video_points."""
    _delete_video_points(video_id, is_transcript=is_transcript, fused=fused)
    try:
        await db.delete_segment_text(video_id, is_transcript=is_transcript, fused=fused)
    except Exception as e:
        print(f"[purge] lexical delete failed: {e}")


# Background tasks
async def process_video(video_path: str, video_id: str, context: str = '', resume: bool = False,
                        do_visual: bool = True, do_audio: bool = True, do_fuse: bool = True):
    """Process a video: extract frames, caption, transcribe audio, and index.

    ``do_visual`` / ``do_audio`` select which modalities to analyze (default both).
    Audio-only or video-only runs still reach the 'analyzed' status; the missing
    modality is flagged on the video (has_frames / has_transcript) so the UI can
    show the appropriate "no video" / "no audio" indicator.
    """
    if not do_visual and not do_audio and not do_fuse:
        do_visual = do_audio = True  # nothing selected: fall back to full analysis
    try:
        await db.update_video(video_id, {'status': 'processing'})
        async with _analysis_semaphore:
            loop = asyncio.get_running_loop()
            if not resume:
                # Start over: clear ONLY the modalities we're about to redo, so an
                # audio-only re-run never wipes existing visual frames and vice versa.
                if do_visual:
                    await db.update_video(video_id, {'frame_count': 0})
                    await db.delete_visual_frames_for_video(video_id)
                    await _purge_points(video_id, is_transcript=False)
                if do_audio:
                    await db.delete_transcript_frames_for_video(video_id)
                    await _purge_points(video_id, is_transcript=True)

            # Collect what already exists (for resume gap-filling and skip logic)
            existing_db_frames = await db.get_frames_for_video(video_id)
            existing_frame_ids = {f.id for f in existing_db_frames}
            has_existing_transcript = any(f.frame_path == '' for f in existing_db_frames)

            # ---- Visual: extract + caption frames ----
            if do_visual:
                frame_dir = FRAMES_DIR / video_id
                frame_dir.mkdir(parents=True, exist_ok=True)
                frames = await loop.run_in_executor(
                    None, lambda: frame_extractor.extract_frames(
                        video_path, str(frame_dir), prefix=f"frame_{video_id}"
                    )
                )
                if not frames and not do_audio:
                    # Visual-only run with no extractable frames is a hard failure.
                    await db.update_video(video_id, {'status': 'error'})
                    return
                for i, (frame_path, timestamp) in enumerate(frames or []):
                    frame_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_frame_{i}"))
                    if frame_id in existing_frame_ids:
                        # Already captioned and indexed in a previous run — skip
                        continue
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
                    await db.update_video(video_id, {'thumbnail_path': str(random.choice(frames)[0])})

            # ---- Audio: transcribe ----
            if do_audio and not (resume and has_existing_transcript):
                await db.update_video(video_id, {'status': 'transcribing', 'transcribe_progress': 0.0})
                print(f"Transcribing audio for {video_id}...")
                _progress_cb = _make_transcribe_progress_cb(video_id, loop)
                async with _transcribe_semaphore:
                    segments = await loop.run_in_executor(
                        None,
                        lambda: transcription_service.transcribe_video(video_path, progress_callback=_progress_cb),
                    )
                print(f"Whisper: {len(segments)} transcript segment(s) for {video_id}")
                for seg in segments:
                    text = f"[transcript] {seg.text}"
                    seg_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_transcript_{seg.start}"))
                    await embed_service.index_frame(
                        video_id=video_id, frame_path="",
                        timestamp=seg.start, caption=text, _point_id=seg_id,
                    )
                    await db.create_frame({
                        "id": seg_id, "video_id": video_id,
                        "timestamp": seg.start, "frame_path": "",
                        "caption": text, "embedding_id": seg_id
                    })
            elif do_audio:
                print(f"Resume: skipping transcription for {video_id} (already indexed)")

            # ---- Fusion: 3rd pass — fuse per-window visual + spoken evidence ----
            # The VLM scored each frame in isolation and Whisper each utterance, so
            # neither sees a moment split across channels. Here we group both into
            # ~20s windows and let the LLM fuse each into one factual description,
            # indexed alongside (never replacing) the raw points so cross-modal
            # moments rank correctly. Rebuilt every run from the current frame set.
            # Gated on do_fuse and only does work when both modalities have data
            # (build_windows needs a (shown) AND a (spoken) line per window), so a
            # fuse-only run works off the existing DB frames + transcript.
            if do_fuse:
                fusion_frames = await db.get_frames_for_video(video_id)
                windows = fusion_service.build_windows(fusion_frames)
                if windows:
                    await db.update_video(video_id, {'status': 'fusing', 'fuse_progress': 0.0})
                    await _purge_points(video_id, fused=True)   # clear any prior fused points
                    for i, w in enumerate(windows):
                        desc = await fusion_service.fuse(context, w["lines"])
                        if desc:
                            fid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_fused_{i}"))
                            await embed_service.index_frame(
                                video_id=video_id, frame_path=w["rep_fp"],
                                timestamp=w["lo"], caption=desc, _point_id=fid, fused=True,
                            )
                        await db.update_video(video_id, {'fuse_progress': (i + 1) / len(windows)})

        # Derive the final modality flags from what's actually indexed, so the
        # "no video" / "no audio" badges always reflect reality.
        final_frames = await db.get_frames_for_video(video_id)
        has_frames = 1 if any(f.frame_path for f in final_frames) else 0
        has_transcript = 1 if any(not f.frame_path for f in final_frames) else 0
        await db.update_video(video_id, {
            'status': 'analyzed', 'has_transcript': has_transcript, 'has_frames': has_frames,
        })

    except asyncio.CancelledError:
        print(f"Analysis cancelled for {video_id}")
        await db.update_video(video_id, {'status': 'error'})
        # Only clear the modalities this run was (re)building, so cancelling an
        # audio-only re-run doesn't discard the existing visual index.
        if do_visual and do_audio:
            await _purge_points(video_id, is_transcript=None)
        elif do_visual:
            await _purge_points(video_id, is_transcript=False)
        elif do_audio:
            await _purge_points(video_id, is_transcript=True)
        elif do_fuse:
            await _purge_points(video_id, fused=True)
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


def _build_video_record(video_id: str, video_path: Path, metadata: dict) -> dict:
    """Assemble a videos-table row from probed ffprobe/filesystem metadata."""
    return {
        "id": video_id,
        "name": video_path.stem,
        "path": str(video_path),
        "duration": metadata.get("duration", 0),
        "width": metadata.get("width", 0),
        "height": metadata.get("height", 0),
        "size": metadata.get("size", 0),
        "fps": metadata.get("fps"),
        "video_codec": metadata.get("video_codec"),
        "audio_codec": metadata.get("audio_codec"),
        "container": metadata.get("container"),
        "extension": metadata.get("extension"),
        "bitrate": metadata.get("bitrate"),
        "resolution_label": metadata.get("resolution_label"),
        "recorded_at": metadata.get("recorded_at"),
        "file_modified_at": metadata.get("file_modified_at"),
        "file_hash": compute_file_hash(video_path),
    }


async def _auto_import(auto_analyze: str = "off", default_context: str = ""):
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
            await db.create_video(_build_video_record(video_id, video_path, metadata))
            await db.update_video(video_id, {"status": "ready"})
            if auto_analyze and auto_analyze != "off":
                _start_analysis(
                    path_str, video_id, context=default_context,
                    do_visual=auto_analyze in ("video", "both"),
                    do_audio=auto_analyze in ("audio", "both"),
                )


async def _auto_import_loop():
    """Run auto-import on startup, then loop periodically if auto_import_interval > 0."""
    while True:
        cfg = _effective_cfg()
        await _auto_import(
            auto_analyze=str(cfg.get("auto_analyze", "off")),
            default_context=str(cfg.get("default_context", "")),
        )
        interval = int(cfg.get("auto_import_interval", 0))
        if interval <= 0:
            break
        await asyncio.sleep(interval * 60)


async def _backfill_metadata():
    """Populate the richer metadata fields (fps, codec, dates, ...) for videos
    imported before those columns existed. Re-probes only rows missing them."""
    try:
        videos = await db.get_all_videos(include_all=True)
    except Exception:
        return
    updated = 0
    for v in videos:
        # fps is the marker for "this row predates the metadata fields".
        if getattr(v, "fps", None) is not None:
            continue
        vpath = Path(v.path)
        updates: dict = {"extension": vpath.suffix.lower().lstrip('.')}
        if vpath.exists():
            try:
                meta = await asyncio.to_thread(get_video_metadata, vpath)
            except Exception:
                meta = {}
            for key in ("duration", "width", "height", "size", "fps", "video_codec",
                        "audio_codec", "container", "extension", "bitrate",
                        "resolution_label", "recorded_at", "file_modified_at"):
                val = meta.get(key)
                if val is not None and val != "":
                    updates[key] = val
        # Ensure fps is set (even to 0.0) so the row isn't re-probed every boot.
        updates.setdefault("fps", 0.0)
        try:
            await db.update_video(v.id, updates)
            updated += 1
        except Exception:
            pass
    if updated:
        print(f"[metadata] backfilled {updated} video(s) with file metadata", flush=True)


async def _backfill_modality_flags():
    """Derive has_frames / has_transcript from the actual indexed frames.

    The has_frames column was added with a blanket DEFAULT 1, so videos that were
    analyzed audio-only before this feature (no visual frames) were wrongly marked
    as having video — hiding the "no video" indicator. This recomputes both flags
    for every analyzed video from real frame data and fixes any mismatch. Cheap
    single grouped query; only writes rows that are actually wrong.
    """
    try:
        counts = await db.frame_modality_counts()
        videos = await db.get_all_videos(include_all=True)
    except Exception:
        return
    fixed = 0
    for v in videos:
        if getattr(v, 'status', None) != 'analyzed':
            continue
        vis, tr = counts.get(v.id, (0, 0))
        want_frames = 1 if vis > 0 else 0
        want_tr = 1 if tr > 0 else 0
        cur_frames = 1 if getattr(v, 'has_frames', 1) else 0
        cur_tr = 1 if getattr(v, 'has_transcript', 0) else 0
        if want_frames != cur_frames or want_tr != cur_tr:
            try:
                await db.update_video(v.id, {'has_frames': want_frames, 'has_transcript': want_tr})
                fixed += 1
            except Exception:
                pass
    if fixed:
        print(f"[modality] corrected has_frames/has_transcript for {fixed} video(s)", flush=True)


async def _backfill_segment_fts():
    """Populate the lexical (BM25/FTS5) index from existing Qdrant payloads.

    The hybrid-recall sparse arm needs one FTS row per point. New points are
    mirrored live by index_frame, but a library indexed before this feature has
    none — so on first startup we scroll every Qdrant point and bulk-insert its
    caption. Cheap (no embedding/GPU) and idempotent: skipped once the index is
    populated.
    """
    try:
        if await db.count_segment_text() > 0:
            return  # already built
        loop = asyncio.get_running_loop()
        points = await loop.run_in_executor(None, lambda: list(embed_service.scroll_all_points()))
        rows = []
        for p in points:
            pl = p.get("payload") or {}
            cap = pl.get("caption") or ""
            rows.append({
                "point_id": p["id"], "video_id": pl.get("video_id") or "",
                "caption": cap,
                "is_transcript": bool(pl.get("is_transcript")) or cap.startswith("[transcript]"),
                "fused": bool(pl.get("fused")),
                "frame_path": pl.get("frame_path") or "", "timestamp": pl.get("timestamp") or 0.0,
            })
        BATCH = 1000
        total = 0
        for i in range(0, len(rows), BATCH):
            total += await db.index_segments_bulk(rows[i:i + BATCH])
        if total:
            print(f"[segment_fts] backfilled {total} point(s) into the lexical index", flush=True)
    except Exception as e:
        print(f"[segment_fts] backfill failed: {e}", flush=True)


def _do_source_backfill():
    """Tag every Qdrant point with an is_transcript flag and index it so search
    can filter by source (Video/Audio toggles) server-side. Idempotent: returns
    immediately once the payload index exists."""
    client = embed_service.qdrant_client
    coll = embed_service.collection_name
    info = client.get_collection(coll)
    schema = getattr(info, "payload_schema", {}) or {}
    if "is_transcript" in schema:
        return
    from qdrant_client.models import PayloadSchemaType
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=coll, with_payload=True, with_vectors=False,
            limit=512, offset=offset,
        )
        if not points:
            break
        t_ids = [p.id for p in points if ((p.payload or {}).get("caption", "") or "").startswith("[transcript]")]
        v_ids = [p.id for p in points if not ((p.payload or {}).get("caption", "") or "").startswith("[transcript]")]
        if t_ids:
            client.set_payload(collection_name=coll, payload={"is_transcript": True}, points=t_ids)
        if v_ids:
            client.set_payload(collection_name=coll, payload={"is_transcript": False}, points=v_ids)
        if offset is None:
            break
    client.create_payload_index(
        collection_name=coll, field_name="is_transcript", field_schema=PayloadSchemaType.BOOL,
    )


async def _backfill_source_flags():
    """Run the one-time is_transcript backfill off the event loop."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _do_source_backfill)
        print("[source] is_transcript backfill complete", flush=True)
    except Exception as e:
        print(f"[source] is_transcript backfill failed: {e}", flush=True)


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


# ── Library export / import ──────────────────────────────────────────────────
# A single .zip carries all *analysis* work (DB rows + Qdrant vectors + frame
# thumbnails) so an analyzed library can be backed up or moved to another setup
# mapped to the same video files WITHOUT re-running vision/transcription/embedding.
# Source video files are NOT included. Import overwrites any matching video
# (by id or file_hash) and inserts the rest.

EXPORT_SCHEMA_VERSION = 1
_VIDEO_COLS = [
    'id', 'name', 'display_name', 'path', 'duration', 'width', 'height', 'size',
    'fps', 'video_codec', 'audio_codec', 'container', 'extension', 'bitrate',
    'resolution_label', 'recorded_at', 'file_modified_at', 'file_hash', 'summary',
    'tags', 'status', 'frame_count', 'thumbnail_path', 'context',
    'transcribe_progress', 'created_at', 'updated_at', 'hidden', 'has_transcript',
    'has_frames',
]
_FRAME_COLS = ['id', 'video_id', 'timestamp', 'frame_path', 'caption', 'embedding_id', 'created_at']
_DATETIME_VIDEO_COLS = {'recorded_at', 'file_modified_at', 'created_at', 'updated_at'}
_DATETIME_FRAME_COLS = {'created_at'}


def _row_to_dict(row, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        val = getattr(row, c, None)
        if hasattr(val, 'isoformat'):
            val = val.isoformat()
        out[c] = val
    return out


def _parse_dt(val):
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except Exception:
        return None


def _dict_to_row(d: dict, cols: list[str], dt_cols: set) -> dict:
    row = {c: d.get(c) for c in cols if c in d}
    for k in dt_cols:
        if k in row:
            row[k] = _parse_dt(row[k])
    return row


@app.get("/export", tags=["System"])
async def export_library(include_images: bool = True):
    """Export the entire analyzed library as a single .zip (DB rows + Qdrant
    vectors + optional frame thumbnails). Does not include source video files."""
    videos = await db.get_all_videos(include_all=True)
    loop = asyncio.get_running_loop()

    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "embed_model": _effective_cfg().get("embed_model"),
        "vector_size": 1024,
        "distance": "Cosine",
        "include_images": include_images,
        "videos": [_row_to_dict(v, _VIDEO_COLS) for v in videos],
        "frames": [],
    }

    point_lines: list[str] = []
    for v in videos:
        frames = await db.get_frames_for_video(v.id)
        manifest["frames"].extend(_row_to_dict(f, _FRAME_COLS) for f in frames)
        pts = await loop.run_in_executor(None, embed_service.export_points, v.id)
        point_lines.extend(json.dumps(p) for p in pts)

    tmp = tempfile.NamedTemporaryFile(prefix="vss_export_", suffix=".zip", dir="/data", delete=False)
    tmp_path = tmp.name
    tmp.close()

    def _write_zip():
        import zipfile
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("points.jsonl", "\n".join(point_lines))
            if include_images:
                for v in videos:
                    vdir = FRAMES_DIR / v.id
                    if vdir.exists():
                        for img in vdir.iterdir():
                            if img.is_file():
                                zf.write(img, arcname=f"frames/{v.id}/{img.name}")

    await loop.run_in_executor(None, _write_zip)
    fname = f"vss-library-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    return FileResponse(
        tmp_path, filename=fname, media_type="application/zip",
        background=BackgroundTask(lambda: os.path.exists(tmp_path) and os.remove(tmp_path)),
    )


@app.post("/import", tags=["System"])
async def import_library(file: UploadFile = File(...)):
    """Import a library .zip produced by /export. Overwrites any video that
    matches by id or file_hash; inserts the rest. Idempotent (re-overwrites)."""
    import zipfile
    import io
    import shutil

    data = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid .zip archive.")
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except KeyError:
        raise HTTPException(status_code=400, detail="Archive is missing manifest.json.")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read manifest.json.")

    # Vectors are only meaningful if produced by the same embedding model + size.
    if manifest.get("vector_size") != 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Vector size mismatch: archive has {manifest.get('vector_size')}, expected 1024.",
        )
    local_model = _effective_cfg().get("embed_model")
    arch_model = manifest.get("embed_model")
    if local_model and arch_model and arch_model != local_model:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding model mismatch: archive '{arch_model}' vs local '{local_model}'. "
                   "The vectors would be invalid — re-export from a matching setup.",
        )

    # Group points and frames by video id.
    points_by_video: dict[str, list] = {}
    try:
        for line in zf.read("points.jsonl").decode().splitlines():
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            vid = (p.get("payload") or {}).get("video_id")
            if vid:
                points_by_video.setdefault(vid, []).append(p)
    except KeyError:
        pass

    frames_by_video: dict[str, list] = {}
    for f in manifest.get("frames", []):
        frames_by_video.setdefault(f.get("video_id"), []).append(f)

    include_images = bool(manifest.get("include_images"))
    loop = asyncio.get_running_loop()
    imported = 0

    # Ids carried in this archive. A hash-match whose id is also in the archive is
    # a distinct incoming video that will be overwritten by its own iteration, so
    # we must NOT collapse it here (that would lose duplicate-content videos).
    incoming_ids = {vd.get("id") for vd in manifest.get("videos", []) if vd.get("id")}

    for vd in manifest.get("videos", []):
        vid = vd.get("id")
        if not vid:
            continue

        # Find local records this import should overwrite: same id OR same file_hash.
        to_delete: set[str] = set()
        existing = await db.get_video(vid)
        if existing:
            to_delete.add(existing.id)
        match = await db.get_video_by_hash(vd.get("file_hash"))
        if match and match.id not in incoming_ids:
            to_delete.add(match.id)

        # Path safeguard: if the exported path doesn't exist locally but a matched
        # local row points at a real file, keep that valid local path.
        new_path = vd.get("path")
        if not new_path or not Path(new_path).exists():
            for mid in to_delete:
                mv = await db.get_video(mid)
                if mv and mv.path and Path(mv.path).exists():
                    new_path = mv.path
                    break

        # Remove matched rows + their vectors, plus any stale points under the
        # incoming id, so the overwrite is clean and idempotent.
        for mid in to_delete:
            await db.delete_video(mid)
            await loop.run_in_executor(None, embed_service.delete_video_points, mid)
            await db.delete_segment_text(mid)
        await loop.run_in_executor(None, embed_service.delete_video_points, vid)
        await db.delete_segment_text(vid)

        # Insert the imported records verbatim (ids preserved).
        row = _dict_to_row(vd, _VIDEO_COLS, _DATETIME_VIDEO_COLS)
        if new_path:
            row["path"] = new_path
        await db.create_video(row)

        frame_rows = [_dict_to_row(f, _FRAME_COLS, _DATETIME_FRAME_COLS)
                      for f in frames_by_video.get(vid, [])]
        await db.create_frames(frame_rows)

        await loop.run_in_executor(None, embed_service.import_points, points_by_video.get(vid, []))
        # Mirror the imported captions into the lexical index for hybrid recall.
        _fts_rows = []
        for _p in points_by_video.get(vid, []):
            _pl = _p.get("payload") or {}
            _cap = _pl.get("caption") or ""
            _fts_rows.append({
                "point_id": _p["id"], "video_id": _pl.get("video_id") or vid,
                "caption": _cap,
                "is_transcript": bool(_pl.get("is_transcript")) or _cap.startswith("[transcript]"),
                "fused": bool(_pl.get("fused")),
                "frame_path": _pl.get("frame_path") or "", "timestamp": _pl.get("timestamp") or 0.0,
            })
        await db.index_segments_bulk(_fts_rows)

        if include_images:
            for name in zf.namelist():
                if name.startswith(f"frames/{vid}/") and not name.endswith("/"):
                    target = FRAMES_DIR / vid / Path(name).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

        imported += 1

    # Ensure the is_transcript payload index exists (no-op if already present);
    # required for the Video/Audio source filter on a fresh target machine.
    await loop.run_in_executor(None, _do_source_backfill)

    total = len(await db.get_all_videos(include_all=True))
    return {"imported": imported, "total": total}



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
        probe(cfg["embed_url"], "Embeddings"),
        probe(cfg["vlm_url"], "Vision AI"),
        probe(cfg["llm_url"], "Summarization"),
    )
    return {"services": list(results)}


@app.get("/status/hardware", tags=["System"])
async def hardware_status():
    """Report detected hardware configuration and AI backend."""
    cfg = _effective_cfg()
    import platform, subprocess, json

    hw_info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware_type": cfg.get("hardware_type", "custom"),
        "ai_backend": cfg.get("ai_backend", "custom"),
        "model_family": cfg.get("model_family", "qwen"),
        "vram_tier": cfg.get("vram_tier", "mid"),
        "resolved_model": cfg.get("llm_model"),
        "resolved_embed_model": cfg.get("embed_model"),
        "embed_url": cfg.get("embed_url"),
        "vlm_url": cfg.get("vlm_url"),
        "llm_url": cfg.get("llm_url"),
    }

    # GPU detection
    try:
        if cfg.get("hardware_type") == "nvidia":
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    name, mem = line.split(", ")
                    gpus.append({"name": name.strip(), "memory": mem.strip()})
                hw_info["gpus"] = gpus
    except Exception:
        hw_info["gpus"] = "nvidia-smi unavailable"

    try:
        if cfg.get("hardware_type") == "amd":
            import os
            hw_info["rocm_devices"] = {
                "kfd": os.path.exists("/dev/kfd"),
                "dri": os.path.exists("/dev/dri/renderD128"),
            }
    except Exception:
        pass

    return hw_info


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
    from ..ingest.frame_extractor import _resolve_hwaccel
    cfg = _effective_cfg()
    warning = _backend_warning(cfg)
    video_dirs = [str(d) for d in VIDEO_SCAN_DIRS if d.exists() and any(d.iterdir())]
    hw = cfg.get("ffmpeg_hwaccel", "off")
    cfg["ffmpeg_hwaccel_resolved"] = _resolve_hwaccel(hw)
    return {**cfg, "backend_warning": warning, "video_dirs": video_dirs}


class SettingsUpdateRequest(BaseModel):
    embed_url: Optional[str] = None
    vlm_url: Optional[str] = None
    llm_url: Optional[str] = None
    qdrant_url: Optional[str] = None
    embed_model: Optional[str] = None
    vlm_model: Optional[str] = None
    llm_model: Optional[str] = None
    whisper_model: Optional[str] = None
    frame_extraction_interval: Optional[int] = None
    embed_batch_size: Optional[int] = None
    max_concurrent_jobs: Optional[int] = None
    ffmpeg_threads: Optional[int] = None
    ffmpeg_hwaccel: Optional[str] = None
    import_blacklist: Optional[List[str]] = None
    auto_import_interval: Optional[int] = None
    auto_analyze: Optional[str] = None  # off | audio | video | both
    default_context: Optional[str] = None
    # Note: hardware_type, vram_tier, model_family are .env-only — not overridable via UI


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
        await db.create_video(_build_video_record(video_id, video_path, metadata))
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
            await db.create_video(_build_video_record(video_id, video_path, metadata))
            await db.update_video(video_id, {"status": "ready"})
            if request.context:
                await db.update_video(video_id, {"context": request.context})
            results.append(video_id)

        return IngestResponse(video_id=", ".join(results), status="ready", frames_processed=0)

    else:
        raise HTTPException(status_code=400, detail="Must specify video_path or directory")


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_videos(
    q: str = "",
    limit: int = 10,
    threshold: float = 0.50,
    sources: str = 'both',
    media_types: str = "",
    resolution: str = "",
    date_from: str = "",
    date_to: str = "",
):
    """Search for videos by natural language query, with optional metadata filters.

    Every search runs the full retrieve-then-rerank pipeline: BGE-M3 / Qdrant
    retrieves candidates (recall), then the bge-reranker cross-encoder assigns
    the final relevance score (precision).

    The optional ``media_types`` / ``resolution`` / ``date_from`` / ``date_to``
    parameters are explicit, user-supplied filters (from the search filter popup)
    applied as a hard AND over the results. They are never inferred from ``q``.
    """
    filters = build_filters(media_types, resolution, date_from, date_to)

    # Filters-only (no text query): browse the library by metadata, newest first.
    if not q.strip():
        if filters:
            return await _filter_browse(filters, limit)
        return SearchResponse(results=[], query=q)

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
    MAX_BOOST = 1.25

    # Retrieval is the *recall* stage: the cross-encoder reranker below is the
    # sole authority on the final relevance score, and the user's relevance
    # threshold is applied *after* reranking. So we must NOT pre-filter Qdrant
    # with the user's threshold — a short query like "dog" has low bi-encoder
    # cosine against a verbose, scene-heavy caption ("a person on a couch ... a
    # dog lies beside them ...") and would be cut before the reranker, which
    # rates it ~0.78, ever sees it. Use NO absolute floor and rely purely on
    # top-k (limit * 8) to return the best candidates; let the reranker filter.
    RECALL_FLOOR = 0.0
    qdrant_threshold = RECALL_FLOOR

    # Build list of queries to search: cleaned + original.
    # The cross-encoder reranker below handles precision; we cast a wide net here.
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
    # Tokens are length >= 2 and must contain a letter, so high-signal model names
    # like "m5", "m1", "4k", "rtx" (and "pc"/"tv"/"ai") survive while bare numbers
    # ("16", "2024") are dropped. A 3+ letter-only rule used to silently discard
    # "m5", which is exactly the kind of concise query the search box teaches.
    key_terms = [
        w.lower() for w in re.findall(r'\b[A-Za-z0-9]{2,}\b', processed["search_terms"])
        if w.lower() not in _SEARCH_STOPWORDS and re.search(r'[A-Za-z]', w)
    ]

    # --- Hybrid recall: fuse dense (bi-encoder) + sparse (BM25/FTS5) candidates ---
    # A two-stage retrieve->rerank pipeline can only rerank what the first stage
    # recalls, and a fixed dense top-k (limit*8) buries strongly-lexical matches as
    # the library grows (e.g. a caption rich in the query's exact terms that the
    # bi-encoder still ranks past position 80). Add a BM25 arm over the same
    # captions and combine the two ranked lists with Reciprocal Rank Fusion, so a
    # hit that ranks at the top of EITHER signal reaches the reranker. Purely
    # additive: on any error or empty lexical result we keep the dense-only pool.
    #
    # The sparse arm's depth is NOT tied to the (small) display limit the way the
    # dense arm is: BM25 over SQLite is cheap (no GPU/embedding), and the whole
    # point of the lexical arm is to reach matches the dense top-k buries, so a
    # shallow lexical pool would defeat it. Use a fixed floor; RRF's 1/(k+rank)
    # keeps deep lexical hits from polluting the top while still letting the
    # reranker (the real precision authority) judge them.
    LEXICAL_RECALL = max(limit * 8, 200)
    try:
        lexical_hits = await db.search_segment_text(
            key_terms, limit=LEXICAL_RECALL, source_intent=source_intent)
    except Exception as e:
        print(f"[search] lexical recall failed: {e}")
        lexical_hits = []

    if lexical_hits:
        RRF_K = 60
        dense_ranked = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
        fused_scores: dict = {}
        hit_by_id: dict = {}
        for ranked_list in (dense_ranked, lexical_hits):
            for rank, r in enumerate(ranked_list):
                hid = r.get("id")
                if hid is None:
                    continue
                hit_by_id.setdefault(hid, r)
                fused_scores[hid] = fused_scores.get(hid, 0.0) + 1.0 / (RRF_K + rank + 1)
        results = []
        for hid, score in fused_scores.items():
            hit = hit_by_id[hid]
            hit["score"] = score
            results.append(hit)

    def _boost(caption: str) -> float:
        if not key_terms:
            return 1.0
        cap_lower = caption.lower()
        cap_words = set(re.findall(r'\b[a-z]{3,}\b', cap_lower))
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
    video_objs: dict = await db.get_videos(list(video_hits.keys()))
    for vid, data in video_hits.items():
        video = video_objs.get(vid)
        if not video:
            continue
        for h in data["hits"]:
            h["video_name"] = video.display_name or video.name
        segments = segment_retriever.group_segments(data["hits"], vid)
        video_meta[vid] = {
            "name": video.name, "display_name": video.display_name, "path": video.path, "duration": video.duration,
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

    # Literal keyword pass: also pull every clip that LITERALLY mentions the query
    # words straight from the DB, independent of the semantic recall pool. The
    # cross-encoder rejects terse product queries ("m5 macbook") to the ~0.50 noise
    # band, so a purely semantic pool misses many genuine mentions; this lexical
    # pass guarantees keyword recall. These hits get reranked below like everything
    # else, then bypass the relevance floor (see broad_keyword).
    #
    # It runs when EITHER the tier is loosest, OR the query has 2+ key terms. A
    # multi-word literal match (e.g. "m5" AND "macbook" both present in the caption)
    # is precise enough to show at ANY relevance tier — otherwise we'd teach users
    # to search concisely and then penalize them for it. Single-word literal matches
    # stay gated to the loosest tier so stricter tiers remain semantically clean.
    multi_literal = len(key_terms) >= 2
    broad_keyword = bool(key_terms) and (threshold <= 0.50 or multi_literal)

    # Distinctive-term lexical recall. The cross-encoder reranker is blind to a
    # clearly-relevant transcript when the query's real anchor word is absent from
    # a more common synonym: for "ai helping doctors" it ranks "...AI voice
    # assistant" (has the literal token "ai") ABOVE "...help nurses and doctors"
    # (the actual answer, but with no "ai" token). A query word that is RARE across
    # the WHOLE library is high-signal intent, so pull every clip that literally
    # contains one and let it bypass the reranker's mistake. Such a term matches
    # only a handful of frames by definition, so this never floods the results.
    distinctive_terms: list[str] = []
    distinctive_frames: list = []
    if key_terms:
        _rare_cap = max(6, int(0.003 * await db.count_frames()))
        for t in key_terms:
            if len(t) < 4:
                continue  # skip terse/common tokens (ai, pc, two, 4k, ...)
            hits = await db.search_captions([t], limit=_rare_cap + 1)
            if 0 < len(hits) <= _rare_cap:
                distinctive_terms.append(t)
                distinctive_frames.extend(hits)

    if broad_keyword or distinctive_frames:
        # Existing semantic segments as (video_id, [start, end]) intervals, split by
        # modality so a lexical frame already covered by a SAME-modality grouped
        # segment doesn't surface twice — but a spoken line is NOT suppressed by an
        # overlapping VISUAL segment. In Both mode the visual frames around a moment
        # get grouped into a span that would otherwise swallow the very transcript
        # line the distinctive-term recall exists to surface.
        vis_spans: dict = {}
        txt_spans: dict = {}
        for s in all_segments:
            (vis_spans if s.get("frame_path") else txt_spans).setdefault(
                s["video_id"], []).append((s["start_time"], s["end_time"]))
        # Multi-term literal matches (broad_keyword) plus single rare-term matches,
        # deduped by frame id so a clip caught by both passes is added once.
        lexical_map: dict = {}
        if broad_keyword:
            for f in await db.search_captions(key_terms, limit=300):
                lexical_map[f.id] = f
        for f in distinctive_frames:
            lexical_map[f.id] = f
        lexical = list(lexical_map.values())
        missing_ids = [f.video_id for f in lexical if f.video_id not in video_meta]
        if missing_ids:
            extra = await db.get_videos(missing_ids)
            video_objs.update(extra)
            for vid, v in extra.items():
                video_meta[vid] = {"name": v.name, "display_name": v.display_name, "path": v.path, "duration": v.duration}
        for f in lexical:
            if f.video_id not in video_meta:
                continue  # video row missing/deleted
            # Honor the modality filter: a visual-only search must not surface a
            # transcript caption (empty frame_path) and vice versa, matching the
            # is_transcript filter applied to the semantic recall pool above.
            is_transcript = not (f.frame_path or "")
            if source_intent == 'visual' and is_transcript:
                continue
            if source_intent == 'speech' and not is_transcript:
                continue
            ts = f.timestamp
            spans = txt_spans if is_transcript else vis_spans
            if any(a <= ts <= b for a, b in spans.get(f.video_id, ())):
                continue  # already represented by a same-modality grouped segment
            all_segments.append({
                "video_id": f.video_id,
                "start_time": ts,
                "end_time": ts,
                "caption": f.caption or "",
                "score": 0.0,
                "frame_path": f.frame_path or "",
            })
            spans.setdefault(f.video_id, []).append((ts, ts))

    # Drop degenerate transcript fragments — a stray 1-2 word chunk like a lone
    # "one" (the tail of "...see you in the next one") carries no meaning yet the
    # cross-encoder can score it in the noise band and surface it as a bogus hit.
    # Visual captions are always descriptive, so this only filters tiny transcripts.
    def _degenerate(seg: dict) -> bool:
        if seg.get("frame_path"):
            return False
        text = (seg.get("caption") or "").replace("[transcript]", "", 1)
        return len(text.split()) < 3

    all_segments = [s for s in all_segments if not _degenerate(s)]

    # Cap the cross-encoder workload: reranking is the latency bottleneck (CPU on
    # some deployments). Keep the strongest recall candidates so a broad keyword
    # search can't feed thousands of pairs through the model. Lexical hits carry
    # recall score 0.0, so they sort last and are trimmed first once the cap is hit.
    RERANK_CAP = 400
    if len(all_segments) > RERANK_CAP:
        all_segments.sort(key=lambda s: s["score"], reverse=True)
        all_segments = all_segments[:RERANK_CAP]

    # Cross-encoder re-ranking — always applied.
    # Qdrant + keyword boost above served only to pick candidates (recall);
    # the cross-encoder now assigns the final absolute relevance score (precision).
    # Feed the cleaned query (filler words like "find the segment where" stripped)
    # so the reranker judges the actual subject, not conversational scaffolding.
    rerank_query = processed.get("search_terms") or q

    # bge-reranker returns a neutral ~0.5 score for bare-noun queries — it expects
    # a descriptive phrase. A terse visual search like "dog" scores ~0.51 against a
    # caption it should match, while "a video of a dog" scores ~0.78. So for SHORT
    # VISUAL queries (1-2 content words) we ALSO rerank with the query framed as
    # "a video of X" and keep the max per segment (see _rerank_segments). Max-
    # pooling means the templated form can only rescue a terse query, never lower
    # a score the natural form judged higher (e.g. "black dog" natural 0.65 vs
    # templated 0.51). We skip this for longer/already-descriptive queries (they
    # rerank well naturally) and for speech/mixed (it hurts transcript matches).
    rerank_queries = [rerank_query]
    if source_intent == 'visual':
        _content = [
            w for w in re.findall(r'\b[a-zA-Z]+\b', rerank_query.lower())
            if w not in _SEARCH_STOPWORDS
        ]
        if 1 <= len(_content) <= 2:
            rerank_queries.append(f"a video of {rerank_query.strip()}")

    # Co-temporal cross-modal fusion (Both mode only). A single moment's meaning
    # is split across channels: the visual caption says WHAT IS SHOWN, the
    # transcript says WHAT IS SAID. A query like "an interview about AI in
    # museums" needs both, yet each caption holds only half, so neither clears the
    # reranker on its own. Enrich each candidate's RERANK TEXT (never its displayed
    # caption) with the OTHER modality's words from the same time window, so the
    # cross-encoder judges the whole moment instead of just one half of it.
    all_segments = await _rerank_segments(rerank_queries, all_segments)

    # Cross-modal anchoring (Both mode). A concise query's RARE word is its real
    # intent: "an interview about ai in museums" is really "find where MUSEUM is
    # said over an interview shot" — the user wouldn't have typed "museum" unless
    # it mattered. But semantic recall routinely drops that rare word (a single
    # spoken "museum" is a faint signal next to a video saying "AI" 100 times), so
    # a stray strong "AI" line outranks the real clip. We therefore (1) decide
    # which query words are DISTINCTIVE from the whole LIBRARY (not the recalled
    # pool, which already dropped them), then (2) cross-reference each candidate
    # moment against the full caption timeline — both what's SHOWN and what's SAID
    # in its time window. A moment that literally contains every distinctive word
    # is confirmed by cross-reference (high confidence — "fairly sure this is it");
    # a moment missing them is held to the loosest tier so it only surfaces when
    # the user explicitly asks for "Nearly everything".
    if source_intent == 'mixed' and all_segments and key_terms:
        AGREE_WINDOW = 30.0    # seconds within which evidence must line up

        def _stem(w: str) -> str:
            return w[:-1] if len(w) > 3 and w.endswith("s") else w

        # Distinctive anchors = query terms that NAME a thing to locate and are
        # RARE across the whole library. Two filters, both required:
        #   1) not a generic "scene" word (talk/stand/person/...) — those describe
        #      HOW someone is, not WHAT to find, and VLM captions word them so
        #      oddly ("gesturing", never "talking") that a user's verb can look
        #      rarer than a real object and hijack the anchor.
        #   2) rare library-wide — a term in a small fraction of frames is a
        #      deliberate locating noun (robot, museum, kvm); a ubiquitous one
        #      (ai, man, person) carries no locating power and must not gate.
        # Library document frequency (not the recalled pool, which already dropped
        # the rare word) is what makes a term distinctive.
        total_frames = await db.count_frames()
        rare_cap = max(40, int(0.04 * total_frames))   # term in <= ~4% of frames is rare
        anchor_terms: list = []
        for t in dict.fromkeys(_stem(x) for x in key_terms):
            if t in _ANCHOR_STOPLIST or len(t) < 3:
                continue
            hits = await db.search_captions([t], limit=rare_cap + 1)
            if 1 <= len(hits) <= rare_cap:
                anchor_terms.append(t)

        if anchor_terms:
            # Full per-video caption timeline (both channels) so a co-temporal
            # mention that was NOT itself recalled still counts toward coverage.
            timelines: dict = {}
            for _vid in {s["video_id"] for s in all_segments}:
                entries = []
                for _f in await db.get_frames_for_video(_vid):
                    _cap = (_f.caption or "").replace("[transcript] ", "").strip().lower()
                    if _cap:
                        entries.append((_f.timestamp, _cap))
                entries.sort()
                timelines[_vid] = entries

            PARTIAL_CEIL = 0.73    # any anchor missing -> at most "Mostly matches"
            for s in all_segments:
                lo, hi = s["start_time"] - AGREE_WINDOW, s["end_time"] + AGREE_WINDOW
                moment = " ".join(
                    c for ts, c in timelines.get(s["video_id"], []) if lo <= ts <= hi
                )
                covered = sum(1 for t in anchor_terms if t in moment)
                coverage = covered / len(anchor_terms)
                if coverage >= 1.0:
                    # Distinctive words cross-referenced nearby: don't cap. Promote
                    # to exact ONLY on direct evidence — every query word literally
                    # in THIS segment's own caption (the man-talking-to-hologram
                    # frame), not merely scattered across a wide window. A wide
                    # window finds "wood" desks + "accents" boxes apart and would
                    # fake an exact; demanding all words in one caption blocks that
                    # while still floating the genuine moment to the top.
                    own = (s.get("caption") or "").replace("[transcript] ", "").lower()
                    own_words = set(re.findall(r'\b[a-z]{2,}\b', own))
                    if all(t in own_words or t in own for t in key_terms):
                        s["score"] = 1.0 - (1.0 - s["score"]) * (1.0 - 0.74)
                elif coverage <= 0.0:
                    # The defining word is nowhere in this moment's window. An
                    # "Exact match" MUST literally contain every distinctive term,
                    # so this can never be exact no matter how confident the
                    # cross-encoder is on phrase structure alone (it happily rates
                    # "talk to each other" 0.79 for "talking to a hologram"). Cap
                    # to "Mostly matches" — still visible (covers a VLM synonym like
                    # "humanoid machine" for "robot"), never deceptively exact.
                    s["score"] = min(s["score"], PARTIAL_CEIL)
                else:
                    # Partial: some distinctive words present, not all. Same rule —
                    # only a full literal cross-reference earns the exact tier, so
                    # cap to "Mostly matches" regardless of reranker confidence.
                    s["score"] = min(s["score"], PARTIAL_CEIL)

    # LLM compositional precision re-rank — the FINAL precision authority.
    # The cross-encoder rewards token co-occurrence regardless of whether the
    # query's attributes bind to the SAME object: for "mini pc with wooden accent"
    # it scores a scene with an unrelated AI Mini PC (on a price tag) and separate
    # "wooden desks" in the background as highly as the true "a black mini PC with
    # a wooden base". No lexical/proximity rule fixes this (a false positive can
    # have "Mac Mini" and "wooden table" in the same sentence, yet the wood is the
    # table, not the computer). Only semantic reasoning distinguishes an attribute
    # bound to the subject from background furniture, so we ask the text LLM to
    # score the top shortlist for whether a SINGLE subject matches the WHOLE query,
    # and blend that compositional confidence in. Bounded cost (one call over the
    # top-K), gated to multi-term/compositional queries, and safe: a failed or
    # silent judgment leaves the cross-encoder ranking unchanged.
    if _llm_rerank_on and llm_reranker is not None and len(key_terms) >= 2 and all_segments:
        all_segments.sort(key=lambda s: s["score"], reverse=True)
        _shortlist = all_segments[:20]
        _caps = [(s.get("caption") or "").replace("[transcript] ", "").strip()
                 for s in _shortlist]
        _llm_scores = await llm_reranker.score(rerank_query, _caps)
        if _llm_scores:
            # Additive blend centered on the LLM's 0.5 neutral point: a confirmed
            # binding promotes (up to +0.375), a clear non-binding match demotes
            # (down to -0.375), an unsure/silent judgment leaves the score as-is.
            # The cross-encoder remains the baseline; the LLM only shifts it.
            LLM_RERANK_WEIGHT = 0.75
            for _s, _ls in zip(_shortlist, _llm_scores):
                if _ls is None:
                    continue
                _s["score"] = max(0.0, min(1.0, _s["score"] + LLM_RERANK_WEIGHT * (_ls - 0.5)))
            all_segments.sort(key=lambda s: s["score"], reverse=True)
            print(f"LLM-rerank '{rerank_query}': top={[round(s['score'], 3) for s in all_segments[:8]]}")

    # Re-group by video for the response
    video_segs: dict = {}
    for seg in all_segments:
        video_segs.setdefault(seg["video_id"], []).append(seg)

    # The reranker's sigmoid maps a logit of 0 ("not relevant") to raw 0.50, so
    # scores in the ~0.50-0.51 band are noise the model is actively rejecting —
    # NOT weak matches. Never show those, even if the user drags the slider to
    # the bottom (raw ~0.37 at the default 50% position). Clamp the effective
    # threshold to this floor so broad searches surface genuine-but-weak matches
    # (>=~0.52) without leaking the neutral-band filler that otherwise displays
    # as a misleading ~65% via the UI score curve.
    RELEVANCE_FLOOR = 0.52
    seg_threshold = max(threshold, RELEVANCE_FLOOR)

    # We ALSO keep any segment that literally contains every query keyword, even if
    # the reranker scored it in the noise band (see broad_keyword above). This makes
    # a literal mention count as a match regardless of how terse the query was.
    def _kw_match(caption: str) -> bool:
        cap = (caption or "").lower()
        return all(t in cap for t in key_terms)

    # A clip that literally contains a library-rare query word (see
    # distinctive_terms) is a confident match the reranker may have underscored
    # only because a commoner synonym carried the literal token instead.
    def _distinctive_match(caption: str) -> bool:
        if not distinctive_terms:
            return False
        cap = (caption or "").lower()
        return any(t in cap for t in distinctive_terms)

    final_results = []
    for vid, segs in video_segs.items():
        meta = video_meta.get(vid)
        if not meta:
            continue
        top_segments = []
        for s in segs:
            if s["score"] >= seg_threshold:
                top_segments.append(s)
            elif (broad_keyword and _kw_match(s.get("caption", ""))) or _distinctive_match(s.get("caption", "")):
                # A clip that literally contains the query's keywords is shown AT
                # the selected bar rather than its unreliable terse-query rerank
                # score, so concise queries aren't penalized with low-relevance
                # dots. BUT a literal word hit is NOT a high-confidence semantic
                # match, so it must never qualify as "Exact match". Cap the
                # presented score just below the Exact tier (0.74): at looser
                # tiers the match still surfaces at the bar, but when the user
                # asks for "Exact match" it falls below the threshold and drops
                # out — so the strictest setting stays purely semantic.
                LEXICAL_DISPLAY_CEIL = 0.73   # one notch under the 0.74 Exact tier
                bumped = min(max(s["score"], seg_threshold), LEXICAL_DISPLAY_CEIL)
                if bumped >= seg_threshold:
                    s["score"] = bumped
                    top_segments.append(s)
        if not top_segments:
            continue
        top_segments.sort(key=lambda s: s["score"], reverse=True)
        best_score = max(s["score"] for s in top_segments)
        final_results.append({
            "video_id": vid,
            "name": meta["display_name"] or meta["name"],
            "path": meta["path"],
            "duration": meta["duration"],
            "score": best_score,
            "matching_frames": len(top_segments),
            "segments": top_segments,
        })

    # Only return videos where the best-matching segment is above threshold. In
    # the loosest tier, keep videos kept solely via a literal keyword match too.
    final_results = [
        r for r in final_results
        if r["score"] >= seg_threshold or broad_keyword or distinctive_terms
    ]

    # Apply explicit, user-supplied metadata filters as a hard AND over results,
    # reusing the Video rows already fetched above (no full-library scan). Hidden
    # and blacklisted videos are excluded, matching the library browse behavior.
    if filters:
        final_results = [
            r for r in final_results
            if (v := video_objs.get(r["video_id"])) is not None
            and (v.hidden or '0') not in ('1', 'blacklist')
            and video_matches(v, filters)
        ]

    # Sort by score descending, cap to limit
    final_results.sort(key=lambda x: x["score"], reverse=True)
    final_results = final_results[:limit]

    # ±1 transcript context: a matched chunk like "the chassis and plug them in."
    # is too terse to read on its own. Attach the adjacent spoken chunks so the UI
    # can show (dimmed) the sentence the match actually sits in. Visual captions are
    # already descriptive, so this is transcript-only; cache the per-video timeline.
    _tl_cache: dict = {}
    _vis_cache: dict = {}
    for r in final_results:
        for s in r["segments"]:
            if s.get("frame_path"):
                continue
            vid = r["video_id"]
            if vid not in _tl_cache:
                all_frames = await db.get_frames_for_video(vid)
                tl = [f for f in all_frames if not f.frame_path]
                tl.sort(key=lambda f: f.timestamp)
                _tl_cache[vid] = tl
                vis = [f for f in all_frames if f.frame_path]
                vis.sort(key=lambda f: f.timestamp)
                _vis_cache[vid] = vis
            tl = _tl_cache[vid]
            # Borrow the nearest visual frame at-or-before the spoken timestamp as a
            # thumbnail. Frames are extracted on scene cuts, so the frame just before
            # is in the same shot (no cut yet) and visually represents the moment;
            # a later frame may be past a cut. Caption still carries [transcript], so
            # the UI keeps the mic icon. Fall back to the first frame after if none.
            vis = _vis_cache[vid]
            if vis:
                t = s["start_time"]
                before = [f for f in vis if f.timestamp <= t]
                s["frame_path"] = (before[-1] if before else vis[0]).frame_path
            if not tl:
                continue
            idx = min(range(len(tl)), key=lambda i: abs(tl[i].timestamp - s["start_time"]))
            if idx > 0:
                s["context_before"] = (tl[idx - 1].caption or "").replace("[transcript] ", "")
            if idx < len(tl) - 1:
                s["context_after"] = (tl[idx + 1].caption or "").replace("[transcript] ", "")

    return SearchResponse(results=final_results, query=q)


async def _metadata_segment(video) -> Optional[dict]:
    """Synthesize a representative segment for a filter-browse match: the video's
    first indexed frame plus a caption describing its technical metadata."""
    caption = describe_metadata(video) or video.name
    frame_path = ""
    try:
        frames = await db.get_frames_for_video(video.id)
        first = next((f for f in frames if f.frame_path), None)
        if first:
            frame_path = first.frame_path
    except Exception:
        pass
    return {
        "video_id": video.id,
        "start_time": 0.0,
        "end_time": 0.0,
        "caption": caption,
        "score": 1.0,
        "frame_path": frame_path,
    }


async def _filter_browse(filters: dict, limit: int) -> "SearchResponse":
    """Return library videos matching the explicit metadata filters (no text
    query), newest first."""
    matches = [v for v in await db.get_all_videos() if video_matches(v, filters)]
    matches.sort(
        key=lambda v: (v.recorded_at or v.file_modified_at or v.created_at or datetime.min),
        reverse=True,
    )
    results = []
    for v in matches[:limit]:
        seg = await _metadata_segment(v)
        results.append({
            "video_id": v.id,
            "name": v.name,
            "path": v.path,
            "duration": v.duration or 0,
            "score": 1.0,
            "matching_frames": 1 if seg and seg["frame_path"] else 0,
            "segments": [seg] if seg else [],
        })
    return SearchResponse(results=results, query="")


@app.get("/metadata/facets", tags=["Search"])
async def metadata_facets():
    """Available metadata filter options for the search filter popup."""
    videos = await db.get_all_videos()
    exts = sorted({(v.extension or "").lower() for v in videos if v.extension})
    # Fixed minimum-resolution ladder (matches the popup dropdown).
    resolutions = ["720p", "1080p", "1440p", "4k"]
    return {"media_types": exts, "resolutions": resolutions}


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
        r["video_name"] = (video.display_name or video.name) if video else ""
    
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
        has_frames=bool(getattr(v, 'has_frames', 1)),
        transcribe_progress=float(getattr(v, 'transcribe_progress', 0.0) or 0.0),
        fuse_progress=float(getattr(v, 'fuse_progress', 0.0) or 0.0),
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


@app.get("/videos/{video_id}/fused", tags=["Analysis"])
async def get_video_fused(video_id: str):
    """Return the fused 3rd-pass window descriptions for a video.

    Fused points live only in Qdrant (payload ``fused=True``), so this reads
    them directly from the collection rather than the DB frame table.
    """
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    segments = []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        points, _ = embed_service.qdrant_client.scroll(
            collection_name=embed_service.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="video_id", match=MatchValue(value=video_id)),
                FieldCondition(key="fused", match=MatchValue(value=True)),
            ]),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            segments.append({
                "id": str(p.id),
                "timestamp": pl.get("timestamp", 0.0),
                "caption": pl.get("caption", ""),
                "path": pl.get("frame_path") or None,
            })
        segments.sort(key=lambda s: s["timestamp"])
    except Exception:
        pass
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
    visual: bool = True   # caption video frames
    audio: bool = True    # transcribe audio
    fuse: bool = True     # 3rd-pass cross-modal fusion (needs both modalities' data)

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
    _start_analysis(video.path, video_id, context=context, resume=req.resume,
                    do_visual=req.visual, do_audio=req.audio, do_fuse=req.fuse)
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
        # Remove ALL existing transcript points for this video from Qdrant by
        # payload filter — this covers per-segment points AND any legacy sliding-
        # window points regardless of how their ids were assigned (deleting by
        # the DB's embedding_id missed points whose Qdrant id was derived
        # differently). Then clear the transcript rows from the DB.
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            embed_service.qdrant_client.delete(
                collection_name=embed_service.collection_name,
                points_selector=Filter(must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id)),
                    FieldCondition(key="is_transcript", match=MatchValue(value=True)),
                ]),
            )
        except Exception as e:
            print(f"Retranscribe: failed clearing old transcript points for {video_id}: {e}")
        await db.delete_transcript_frames_for_video(video_id)

        # Re-transcribe
        loop = asyncio.get_running_loop()
        await db.update_video(video_id, {'transcribe_progress': 0.0})
        _progress_cb = _make_transcribe_progress_cb(video_id, loop)
        async with _transcribe_semaphore:
            segments = await loop.run_in_executor(
                None,
                lambda: transcription_service.transcribe_video(video_path, progress_callback=_progress_cb),
            )
        print(f"Retranscribe: {len(segments)} segment(s) for {video_id}")

        for seg in segments:
            text = f"[transcript] {seg.text}"
            seg_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_transcript_{seg.start}"))
            await embed_service.index_frame(
                video_id=video_id, frame_path="", timestamp=seg.start,
                caption=text, _point_id=seg_id,
            )
            await db.create_frame({
                "id": seg_id, "video_id": video_id,
                "timestamp": seg.start, "frame_path": "",
                "caption": text, "embedding_id": seg_id
            })

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