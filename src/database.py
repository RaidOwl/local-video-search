from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone
from typing import Optional

Base = declarative_base()


class Video(Base):
    __tablename__ = 'videos'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    path = Column(String, nullable=False)
    duration = Column(Float, default=0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    size = Column(Integer, default=0)
    fps = Column(Float, nullable=True)  # frames per second
    video_codec = Column(String, nullable=True)  # e.g. h264, hevc, av1
    audio_codec = Column(String, nullable=True)  # e.g. aac, opus
    container = Column(String, nullable=True)  # ffprobe format_name
    extension = Column(String, nullable=True)  # file extension w/o dot, e.g. mp4
    bitrate = Column(Integer, nullable=True)  # bits per second
    resolution_label = Column(String, nullable=True)  # 4K, 1080p, 720p, ...
    recorded_at = Column(DateTime, nullable=True)  # creation time from file metadata
    file_modified_at = Column(DateTime, nullable=True)  # filesystem mtime
    file_hash = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    tags = Column(Text, nullable=True, default='[]')  # JSON array
    status = Column(String, nullable=False, default='processing')  # processing | analyzed | error
    frame_count = Column(Integer, default=0)
    thumbnail_path = Column(String, nullable=True)
    context = Column(Text, nullable=True)  # user-supplied context used during analysis
    transcribe_progress = Column(Float, nullable=True, default=0.0)  # 0..1 during transcription
    fuse_progress = Column(Float, nullable=True, default=0.0)  # 0..1 during fusion (3rd pass)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    hidden = Column(String, nullable=True, default='0')  # '1' = hidden from library
    has_transcript = Column(Integer, nullable=True, default=0)  # 1 = audio transcription indexed
    has_frames = Column(Integer, nullable=True, default=1)  # 1 = visual frames captioned & indexed
    frames = relationship("Frame", back_populates="video", cascade="all, delete-orphan")


class Frame(Base):
    __tablename__ = 'frames'
    
    id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey('videos.id'), nullable=False)
    timestamp = Column(Float, nullable=False)
    frame_path = Column(String, nullable=False)
    caption = Column(Text, nullable=True)
    embedding_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    video = relationship("Video", back_populates="frames")


class Database:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.session_maker = async_sessionmaker(self.engine, class_=AsyncSession)
    
    async def init_db(self):
        """Initialize the database tables."""
        from sqlalchemy import text
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Migrate: add columns introduced after initial schema
            for col in ['context TEXT', 'thumbnail_path TEXT', 'hidden TEXT DEFAULT 0', 'has_transcript INTEGER DEFAULT 0',
                        'has_frames INTEGER DEFAULT 1',
                        'fps REAL', 'video_codec TEXT', 'audio_codec TEXT', 'container TEXT',
                        'extension TEXT', 'bitrate INTEGER', 'resolution_label TEXT',
                        'recorded_at DATETIME', 'file_modified_at DATETIME', 'transcribe_progress REAL DEFAULT 0',
                        'fuse_progress REAL DEFAULT 0']:
                col_name = col.split()[0]
                try:
                    await conn.execute(text(f'ALTER TABLE videos ADD COLUMN {col}'))
                except Exception:
                    pass  # already exists
            # Lexical (BM25) full-text index mirroring every Qdrant point, used for
            # hybrid recall (dense + sparse) in /search. One row per point id —
            # visual, transcript, and fused captions alike — so a strong lexical
            # match the bi-encoder buries can still reach the reranker.
            await conn.execute(text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts USING fts5("
                "point_id UNINDEXED, video_id UNINDEXED, is_transcript UNINDEXED, "
                "fused UNINDEXED, frame_path UNINDEXED, timestamp UNINDEXED, caption)"
            ))
    
    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return self.session_maker()
    
    # Video operations
    async def create_video(self, video_data: dict) -> Video:
        """Create a new video record."""
        async with self.session_maker() as session:
            video = Video(**video_data)
            session.add(video)
            await session.commit()
            await session.refresh(video)
            return video
    
    async def get_video(self, video_id: str) -> Optional[Video]:
        """Get a video by ID."""
        async with self.session_maker() as session:
            result = await session.get(Video, video_id)
            if result:
                # Expire to load relationships
                await session.refresh(result)
            return result

    async def get_videos(self, video_ids: list[str]) -> dict[str, Video]:
        """Batch-fetch videos by ID in a single query, returned keyed by ID.

        Used by the search path to avoid an N+1 of per-video ``get_video`` calls
        (each of which opened a session and did a redundant refresh).
        """
        ids = [vid for vid in dict.fromkeys(video_ids) if vid]
        if not ids:
            return {}
        async with self.session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(select(Video).where(Video.id.in_(ids)))
            return {v.id: v for v in result.scalars().all()}

    async def get_video_by_path(self, path: str) -> Optional[Video]:
        """Get a video by file path."""
        async with self.session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(select(Video).where(Video.path == path))
            return result.scalars().first()

    async def get_video_by_hash(self, file_hash: str) -> Optional[Video]:
        """Get a video by its content hash (used to match files across machines)."""
        if not file_hash:
            return None
        async with self.session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(select(Video).where(Video.file_hash == file_hash))
            return result.scalars().first()

    async def get_all_videos(self, hidden: bool = False, include_all: bool = False) -> list[Video]:
        """Get all videos. By default excludes hidden videos.
        hidden=True  → only manually-hidden ('1')
        include_all  → every row regardless of hidden state
        """
        async with self.session_maker() as session:
            from sqlalchemy import select
            q = select(Video).order_by(Video.created_at.desc())
            if include_all:
                pass
            elif hidden:
                q = q.where(Video.hidden == '1')
            else:
                # Exclude both manually hidden ('1') and blacklist-hidden ('blacklist')
                q = q.where((Video.hidden == None) | (Video.hidden.notin_(['1', 'blacklist'])))
            result = await session.execute(q)
            return list(result.scalars().all())
    
    async def update_video_summary(self, video_id: str, summary: str):
        """Update the summary for a video."""
        async with self.session_maker() as session:
            video = await session.get(Video, video_id)
            if video:
                video.summary = summary
                await session.commit()

    async def update_video(self, video_id: str, updates: dict):
        """Update arbitrary fields on a video."""
        async with self.session_maker() as session:
            video = await session.get(Video, video_id)
            if video:
                for key, value in updates.items():
                    setattr(video, key, value)
                await session.commit()
    
    # Frame operations
    async def create_frame(self, frame_data: dict) -> Frame:
        """Create a new frame record."""
        async with self.session_maker() as session:
            frame = Frame(**frame_data)
            session.add(frame)
            await session.commit()
            await session.refresh(frame)
            return frame

    async def create_frames(self, frames: list[dict]):
        """Bulk-insert frame records (used by library import)."""
        if not frames:
            return
        async with self.session_maker() as session:
            session.add_all([Frame(**f) for f in frames])
            await session.commit()
    
    async def get_frames_for_video(self, video_id: str) -> list[Frame]:
        """Get all frames for a video."""
        async with self.session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Frame)
                .where(Frame.video_id == video_id)
                .order_by(Frame.timestamp)
            )
            return list(result.scalars().all())

    async def count_frames(self) -> int:
        """Total number of indexed frames (visual + transcript) library-wide.

        Used by cross-modal search to size the 'distinctive term' rarity cap
        relative to the corpus rather than a hard-coded constant.
        """
        async with self.session_maker() as session:
            from sqlalchemy import select, func
            result = await session.execute(select(func.count()).select_from(Frame))
            return int(result.scalar() or 0)

    async def frame_modality_counts(self) -> dict:
        """Per-video (visual_frame_count, transcript_frame_count) in a single query.

        Visual frames have a non-empty frame_path; transcript frames have ''.
        Used by the startup backfill to derive has_frames / has_transcript.
        """
        async with self.session_maker() as session:
            from sqlalchemy import select, func, case
            stmt = (
                select(
                    Frame.video_id,
                    func.sum(case((Frame.frame_path != "", 1), else_=0)),
                    func.sum(case((Frame.frame_path == "", 1), else_=0)),
                )
                .group_by(Frame.video_id)
            )
            result = await session.execute(stmt)
            return {vid: (int(v or 0), int(t or 0)) for vid, v, t in result.all()}

    async def search_captions(self, terms: list[str], limit: int = 300) -> list[Frame]:
        """Lexical fallback: frames whose caption contains ALL of ``terms``.

        Used by the loosest search tier so "Nearly everything" can surface every
        clip that literally mentions the query words, independent of the semantic
        recall pool. Case-insensitive substring match (AND across terms).
        """
        terms = [t for t in (terms or []) if t]
        if not terms:
            return []
        async with self.session_maker() as session:
            from sqlalchemy import select, func
            stmt = select(Frame)
            for t in terms:
                # Escape LIKE wildcards so a stray % / _ in a term is matched
                # literally rather than acting as a wildcard.
                esc = t.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                stmt = stmt.where(func.lower(Frame.caption).like(f"%{esc}%", escape="\\"))
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ---- Lexical (BM25 / FTS5) segment index for hybrid recall ----

    @staticmethod
    def _fts_match(terms: list[str]) -> str:
        """Build an FTS5 MATCH expression: OR of prefix tokens.

        Prefix (`wood*`) bridges simple inflections the unicode61 tokenizer can't
        (query "wood" -> caption "wooden"), and OR (not AND) means a partial term
        overlap still retrieves + ranks the row by BM25 rather than dropping it.
        """
        import re as _re
        toks = []
        for t in (terms or []):
            t = _re.sub(r'[^a-z0-9]', '', (t or '').lower())
            if t:
                toks.append(f'"{t}"*')
        return " OR ".join(toks)

    async def index_segment_text(self, point_id: str, video_id: str, caption: str,
                                 is_transcript: bool = False, fused: bool = False,
                                 frame_path: str = "", timestamp: float = 0.0) -> None:
        """Upsert one point's caption into the lexical index (delete-then-insert)."""
        from sqlalchemy import text
        async with self.session_maker() as session:
            await session.execute(text("DELETE FROM segment_fts WHERE point_id = :pid"),
                                  {"pid": point_id})
            await session.execute(text(
                "INSERT INTO segment_fts(point_id, video_id, is_transcript, fused, "
                "frame_path, timestamp, caption) VALUES (:pid, :vid, :istr, :fused, "
                ":fp, :ts, :cap)"), {
                    "pid": point_id, "vid": video_id,
                    "istr": "1" if is_transcript else "0",
                    "fused": "1" if fused else "0",
                    "fp": frame_path or "", "ts": float(timestamp or 0.0),
                    "cap": caption or "",
                })
            await session.commit()

    async def index_segments_bulk(self, rows: list[dict]) -> int:
        """Insert many segment rows in a single transaction (used by the backfill)."""
        from sqlalchemy import text
        if not rows:
            return 0
        async with self.session_maker() as session:
            for r in rows:
                await session.execute(text(
                    "INSERT INTO segment_fts(point_id, video_id, is_transcript, fused, "
                    "frame_path, timestamp, caption) VALUES (:pid, :vid, :istr, :fused, "
                    ":fp, :ts, :cap)"), {
                        "pid": r["point_id"], "vid": r.get("video_id") or "",
                        "istr": "1" if r.get("is_transcript") else "0",
                        "fused": "1" if r.get("fused") else "0",
                        "fp": r.get("frame_path") or "", "ts": float(r.get("timestamp") or 0.0),
                        "cap": r.get("caption") or "",
                    })
            await session.commit()
        return len(rows)

    async def delete_segment_text(self, video_id: str, is_transcript: Optional[bool] = None,
                                  fused: Optional[bool] = None) -> None:
        """Remove a video's lexical rows, mirroring _delete_video_points' filters."""
        from sqlalchemy import text
        clauses = ["video_id = :vid"]
        params: dict = {"vid": video_id}
        if is_transcript is not None:
            clauses.append("is_transcript = :istr")
            params["istr"] = "1" if is_transcript else "0"
        if fused is not None:
            clauses.append("fused = :fused")
            params["fused"] = "1" if fused else "0"
        async with self.session_maker() as session:
            await session.execute(
                text(f"DELETE FROM segment_fts WHERE {' AND '.join(clauses)}"), params)
            await session.commit()

    async def count_segment_text(self) -> int:
        """Number of rows in the lexical index (0 => backfill needed)."""
        from sqlalchemy import text
        async with self.session_maker() as session:
            res = await session.execute(text("SELECT count(*) FROM segment_fts"))
            return int(res.scalar() or 0)

    async def search_segment_text(self, terms: list[str], limit: int = 80,
                                  source_intent: str = 'mixed',
                                  video_id: str = None) -> list[dict]:
        """BM25-ranked lexical recall over captions (the sparse arm of hybrid search).

        Returns hit dicts shaped like the dense recall hits so the two can be
        reciprocal-rank-fused into one candidate pool. Lower bm25() is better, so
        results are returned best-first.
        """
        match = self._fts_match(terms)
        if not match:
            return []
        from sqlalchemy import text
        clauses = ["segment_fts MATCH :m"]
        params: dict = {"m": match, "lim": limit}
        if video_id:
            clauses.append("video_id = :vid")
            params["vid"] = video_id
        if source_intent == 'visual':
            clauses.append("is_transcript = '0'")
        elif source_intent == 'speech':
            clauses.append("is_transcript = '1'")
        sql = ("SELECT point_id, video_id, is_transcript, fused, frame_path, timestamp, "
               "caption, bm25(segment_fts) AS rank FROM segment_fts "
               f"WHERE {' AND '.join(clauses)} ORDER BY rank LIMIT :lim")
        async with self.session_maker() as session:
            res = await session.execute(text(sql), params)
            out: list[dict] = []
            for row in res.mappings().all():
                out.append({
                    "id": row["point_id"],
                    "video_id": row["video_id"],
                    "frame_path": row["frame_path"] or "",
                    "timestamp": row["timestamp"],
                    "caption": row["caption"] or "",
                    "is_transcript": row["is_transcript"] == "1",
                    "fused": row["fused"] == "1",
                    "score": 0.0,
                })
            return out

    async def get_frame(self, frame_id: str) -> Frame | None:
        """Get a single frame by ID."""
        async with self.session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(select(Frame).where(Frame.id == frame_id))
            return result.scalar_one_or_none()

    async def update_frame(self, frame_id: str, data: dict) -> Frame | None:
        """Update a frame record."""
        async with self.session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(select(Frame).where(Frame.id == frame_id))
            frame = result.scalar_one_or_none()
            if not frame:
                return None
            for k, v in data.items():
                setattr(frame, k, v)
            await session.commit()
            await session.refresh(frame)
            return frame

    async def get_captions_for_video(self, video_id: str) -> list[dict]:
        """Get all captions for a video."""
        frames = await self.get_frames_for_video(video_id)
        return [
            {"timestamp": f.timestamp, "caption": f.caption or ""}
            for f in frames
        ]

    async def delete_video(self, video_id: str):
        """Delete a video and all its frames."""
        async with self.session_maker() as session:
            from sqlalchemy import select, delete as sa_delete
            await session.execute(sa_delete(Frame).where(Frame.video_id == video_id))
            await session.execute(sa_delete(Video).where(Video.id == video_id))
            await session.commit()

    async def delete_frames_for_video(self, video_id: str):
        """Delete all frame records for a video (used before re-processing)."""
        async with self.session_maker() as session:
            from sqlalchemy import delete as sa_delete
            await session.execute(sa_delete(Frame).where(Frame.video_id == video_id))
            await session.commit()

    async def delete_transcript_frames_for_video(self, video_id: str):
        """Delete only transcript frame records (frame_path='') for a video."""
        async with self.session_maker() as session:
            from sqlalchemy import delete as sa_delete
            await session.execute(
                sa_delete(Frame)
                .where(Frame.video_id == video_id)
                .where(Frame.frame_path == "")
            )
            await session.commit()

    async def delete_visual_frames_for_video(self, video_id: str):
        """Delete only visual frame records (frame_path != '') for a video."""
        async with self.session_maker() as session:
            from sqlalchemy import delete as sa_delete
            await session.execute(
                sa_delete(Frame)
                .where(Frame.video_id == video_id)
                .where(Frame.frame_path != "")
            )
            await session.commit()