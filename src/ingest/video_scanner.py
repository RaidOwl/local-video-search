from pathlib import Path
from typing import List
import hashlib


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


def _parse_frame_rate(value: str) -> float:
    """Parse an ffprobe frame-rate string like '30000/1001' or '30/1' into fps."""
    if not value or value in ('0/0', '0'):
        return 0.0
    try:
        if '/' in value:
            num, den = value.split('/', 1)
            den_f = float(den)
            return round(float(num) / den_f, 3) if den_f else 0.0
        return round(float(value), 3)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _resolution_label(width: int, height: int) -> str:
    """Map a pixel resolution to a common label (4K, 1080p, 720p, ...)."""
    h = max(int(height or 0), 0)
    if h <= 0:
        return ""
    if h >= 2000:
        return "4K"
    if h >= 1300:
        return "1440p"
    if h >= 1000:
        return "1080p"
    if h >= 660:
        return "720p"
    if h >= 460:
        return "480p"
    if h >= 340:
        return "360p"
    return f"{h}p"


def _parse_creation_time(tags: dict):
    """Parse an ffprobe creation_time tag into a datetime, or None."""
    if not tags:
        return None
    raw = tags.get('creation_time') or tags.get('com.apple.quicktime.creationdate')
    if not raw:
        return None
    from datetime import datetime
    text = str(raw).strip().replace('Z', '+00:00')
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def get_video_metadata(filepath: Path) -> dict:
    """Extract metadata from a video file via ffprobe plus filesystem stat."""
    import subprocess
    from datetime import datetime, timezone

    extension = filepath.suffix.lower().lstrip('.')
    try:
        file_modified_at = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
    except OSError:
        file_modified_at = None

    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(filepath)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {
                "path": str(filepath), "name": filepath.stem,
                "extension": extension, "file_modified_at": file_modified_at,
                "error": result.stderr,
            }
        
        import json
        data = json.loads(result.stdout)
        fmt = data.get('format', {})
        
        # Find video / audio streams
        video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})
        audio_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), {})

        width = video_stream.get('width', 0) or 0
        height = video_stream.get('height', 0) or 0
        fps = _parse_frame_rate(video_stream.get('avg_frame_rate', '')) \
            or _parse_frame_rate(video_stream.get('r_frame_rate', ''))

        # Prefer the embedded creation time; fall back to the file's mtime.
        recorded_at = _parse_creation_time(fmt.get('tags', {})) \
            or _parse_creation_time(video_stream.get('tags', {}))

        try:
            bitrate = int(fmt.get('bit_rate', 0) or 0)
        except (ValueError, TypeError):
            bitrate = 0

        return {
            "path": str(filepath),
            "name": filepath.stem,
            "format": fmt.get('format_name', 'unknown'),
            "container": fmt.get('format_name', ''),
            "extension": extension,
            "duration": float(fmt.get('duration', 0) or 0),
            "size": int(fmt.get('size', 0) or 0),
            "width": width,
            "height": height,
            "fps": fps,
            "video_codec": video_stream.get('codec_name', '') or '',
            "audio_codec": audio_stream.get('codec_name', '') or '',
            "bitrate": bitrate,
            "resolution_label": _resolution_label(width, height),
            "recorded_at": recorded_at,
            "file_modified_at": file_modified_at,
        }
    except Exception as e:
        return {
            "path": str(filepath), "name": filepath.stem,
            "extension": extension, "file_modified_at": file_modified_at,
            "error": str(e),
        }