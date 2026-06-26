import subprocess
import os
from pathlib import Path
from typing import List, Optional
import asyncio


class FrameExtractor:
    """Extract frames from videos using FFmpeg."""
    
    def __init__(self, interval: int = 5):
        """
        Args:
            interval: Seconds between extracted frames
        """
        self.interval = interval
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        prefix: str = "frame"
    ) -> List[str]:
        """
        Extract frames from a video at regular intervals.
        
        Returns:
            List of extracted frame file paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video duration
        duration = self._get_duration(video_path)
        if duration <= 0:
            return []
        
        # Calculate timestamps for frame extraction
        timestamps = []
        current_time = 0.0
        while current_time < duration:
            timestamps.append(current_time)
            current_time += self.interval
        
        frames = []
        for i, timestamp in enumerate(timestamps):
            output_path = output_dir / f"{prefix}_{i:04d}.jpg"
            
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(timestamp),
                '-i', str(video_path),
                '-vframes', '1',
                '-q:v', '2',
                '-vf', 'scale=448:-1',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and output_path.exists():
                frames.append(str(output_path))
        
        return frames
    
    def extract_frame_at(
        self,
        video_path: str,
        timestamp: float,
        output_path: str
    ) -> Optional[str]:
        """Extract a single frame at a specific timestamp."""
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-q:v', '2',
            '-vf', 'scale=448:-1',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and Path(output_path).exists():
            return output_path
        return None
    
    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return float(data.get('format', {}).get('duration', 0))
        except Exception:
            pass
        return 0.0


async def extract_frames_async(
    video_path: str,
    output_dir: str,
    interval: int = 5
) -> List[str]:
    """Async wrapper for frame extraction."""
    loop = asyncio.get_event_loop()
    extractor = FrameExtractor(interval=interval)
    return await loop.run_in_executor(
        None,
        extractor.extract_frames,
        video_path,
        output_dir
    )