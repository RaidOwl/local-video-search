import os
from pathlib import Path
from typing import List, Optional
import hashlib
import asyncio


VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.m4v', '.flv'}


def find_video_files(directory: str) -> List[Path]:
    """Recursively find all video files in a directory."""
    video_dir = Path(directory)
    if not video_dir.exists():
        return []
    
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(video_dir.rglob(f'*{ext}'))
    return sorted(videos)


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


async def scan_directory(directory: str, known_hashes: dict[str, str] = None) -> dict:
    """
    Scan directory for videos and return info about new/changed files.
    
    Returns:
        {
            "new": [Path, ...],
            "changed": [Path, ...],
            "removed": [Path, ...],
            "unchanged": [Path, ...]
        }
    """
    known_hashes = known_hashes or {}
    current_files = find_video_files(directory)
    current_hashes = {}
    
    result = {"new": [], "changed": [], "removed": [], "unchanged": []}
    
    # Check each current file
    for filepath in current_files:
        file_hash = compute_file_hash(filepath)
        current_hashes[str(filepath)] = file_hash
        
        if str(filepath) not in known_hashes:
            result["new"].append(filepath)
        elif known_hashes[str(filepath)] != file_hash:
            result["changed"].append(filepath)
        else:
            result["unchanged"].append(filepath)
    
    # Find removed files
    for old_path in known_hashes:
        if old_path not in current_hashes:
            result["removed"].append(Path(old_path))
    
    return result, current_hashes


def get_video_metadata(filepath: Path) -> dict:
    """Extract basic metadata from a video file."""
    import subprocess
    
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(filepath)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {"error": result.stderr}
        
        import json
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})
        
        return {
            "path": str(filepath),
            "name": filepath.stem,
            "format": data.get('format', {}).get('format_name', 'unknown'),
            "duration": float(data.get('format', {}).get('duration', 0)),
            "size": int(data.get('format', {}).get('size', 0)),
            "width": video_stream.get('width', 0),
            "height": video_stream.get('height', 0),
        }
    except Exception as e:
        return {"path": str(filepath), "error": str(e)}