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
    file_hash = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    tags = Column(Text, nullable=True, default='[]')  # JSON array
    status = Column(String, nullable=False, default='processing')  # processing | analyzed | error
    frame_count = Column(Integer, default=0)
    thumbnail_path = Column(String, nullable=True)
    context = Column(Text, nullable=True)  # user-supplied context used during analysis
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    hidden = Column(String, nullable=True, default='0')  # '1' = hidden from library
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
            for col in ['context TEXT', 'thumbnail_path TEXT', 'hidden TEXT DEFAULT 0']:
                col_name = col.split()[0]
                try:
                    await conn.execute(text(f'ALTER TABLE videos ADD COLUMN {col}'))
                except Exception:
                    pass  # already exists
    
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
    
    async def get_video_by_path(self, path: str) -> Optional[Video]:
        """Get a video by file path."""
        async with self.session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(select(Video).where(Video.path == path))
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