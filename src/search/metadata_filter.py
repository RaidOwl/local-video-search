"""Explicit, opt-in video-metadata filters.

Filters are supplied deliberately by the user (via the search-page filter popup)
as structured values — never inferred from free text — so they can never misfire
on a normal content search. This module builds a filter spec from request params
and matches it against rows in the ``videos`` table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


# Minimum-resolution dropdown value -> minimum pixel height.
_RESOLUTION_MIN_HEIGHT = {
    '720p': 660,
    '1080p': 1000,
    '1440p': 1300,
    '4k': 2000,
}


def _parse_date(value: str, end_of_day: bool = False) -> Optional[datetime]:
    """Parse a YYYY-MM-DD string into a UTC datetime, or None."""
    if not value:
        return None
    try:
        d = datetime.strptime(value.strip(), '%Y-%m-%d')
    except ValueError:
        return None
    if end_of_day:
        d = d.replace(hour=23, minute=59, second=59)
    return d.replace(tzinfo=timezone.utc)


def build_filters(media_types: str = "", resolution: str = "",
                  date_from: str = "", date_to: str = "") -> dict:
    """Assemble a filter spec from raw request params. Empty values are omitted."""
    filters: dict = {}

    exts = [e.strip().lower().lstrip('.') for e in (media_types or "").split(',') if e.strip()]
    if exts:
        filters['extensions'] = exts

    min_h = _RESOLUTION_MIN_HEIGHT.get((resolution or "").strip().lower())
    if min_h:
        filters['res_min_h'] = min_h

    df = _parse_date(date_from)
    if df:
        filters['date_from'] = df
    dt = _parse_date(date_to, end_of_day=True)
    if dt:
        filters['date_to'] = dt

    return filters


def effective_date(video) -> Optional[datetime]:
    """The date a video was created: embedded recording time, else file mtime."""
    return (getattr(video, 'recorded_at', None)
            or getattr(video, 'file_modified_at', None)
            or getattr(video, 'created_at', None))


def _as_aware(d: Optional[datetime]) -> Optional[datetime]:
    if d is not None and d.tzinfo is None:
        return d.replace(tzinfo=timezone.utc)
    return d


def video_matches(video, filters: dict) -> bool:
    """Return True if ``video`` satisfies every constraint in ``filters`` (AND)."""
    exts = filters.get('extensions')
    if exts:
        # Match on the real file suffix only — the ffprobe container name is a
        # shared "mov,mp4,m4a,..." for QuickTime files and would cross-match.
        vext = (getattr(video, 'extension', '') or '').lower()
        if vext not in exts:
            return False

    if 'res_min_h' in filters:
        if (getattr(video, 'height', 0) or 0) < filters['res_min_h']:
            return False

    if 'date_from' in filters or 'date_to' in filters:
        d = _as_aware(effective_date(video))
        if d is None:
            return False
        if 'date_from' in filters and d < filters['date_from']:
            return False
        if 'date_to' in filters and d > filters['date_to']:
            return False

    return True


def describe_metadata(video) -> str:
    """Human-readable one-line metadata summary (used as a browse-result caption)."""
    parts: list[str] = []
    ext = (getattr(video, 'extension', '') or '').upper()
    if ext:
        parts.append(ext)
    label = getattr(video, 'resolution_label', '') or ''
    if label:
        parts.append(label)
    fps = getattr(video, 'fps', None)
    if fps:
        parts.append(f"{fps:g}fps")
    codec = getattr(video, 'video_codec', '') or ''
    if codec:
        parts.append(codec)
    d = effective_date(video)
    if d:
        parts.append(d.strftime('%Y-%m-%d'))
    return ' · '.join(parts)
