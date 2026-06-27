from typing import List
from dataclasses import dataclass


@dataclass
class SegmentResult:
    """A segment result from a video search."""
    video_id: str
    video_name: str
    start_time: float
    end_time: float
    caption: str
    score: float
    frame_path: str


class SegmentRetriever:
    """Retrieve and rank video segments based on search results."""
    
    def __init__(self, time_window: float = 5.0):
        """
        Args:
            time_window: Seconds to expand around matching frames
        """
        self.time_window = time_window
    
    def group_segments(
        self,
        search_results: List[dict],
        video_id: str = None
    ) -> List[SegmentResult]:
        """
        Group nearby frames into segments.
        
        Args:
            search_results: List of frame search results with timestamps
            video_id: Optional filter for specific video
        
        Returns:
            List of grouped segments
        """
        if video_id:
            results = [r for r in search_results if r.get('video_id') == video_id]
        else:
            results = search_results
        
        if not results:
            return []
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda x: x.get('timestamp', 0))
        
        # Group nearby frames
        segments = []
        current_segment = None
        
        for frame in sorted_results:
            timestamp = frame.get('timestamp', 0)
            
            if current_segment is None:
                current_segment = {
                    'frames': [frame],
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'video_id': frame.get('video_id'),
                    'video_name': frame.get('video_name', ''),
                    'captions': [frame.get('caption', '')],
                    'scores': [frame.get('score', 0)],
                    'frame_paths': [frame.get('frame_path', '')]
                }
            elif timestamp - current_segment['end_time'] <= self.time_window:
                # Extend current segment
                current_segment['frames'].append(frame)
                current_segment['end_time'] = timestamp
                current_segment['captions'].append(frame.get('caption', ''))
                current_segment['scores'].append(frame.get('score', 0))
                current_segment['frame_paths'].append(frame.get('frame_path', ''))
            else:
                # Save current and start new segment
                segments.append(self._create_segment(current_segment))
                current_segment = {
                    'frames': [frame],
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'video_id': frame.get('video_id'),
                    'video_name': frame.get('video_name', ''),
                    'captions': [frame.get('caption', '')],
                    'scores': [frame.get('score', 0)],
                    'frame_paths': [frame.get('frame_path', '')]
                }
        
        # Don't forget the last segment
        if current_segment:
            segments.append(self._create_segment(current_segment))

        # Return highest-scoring segments first
        segments.sort(key=lambda s: s.score, reverse=True)

        return segments
    
    def _create_segment(self, data: dict) -> SegmentResult:
        """Create a SegmentResult from grouped frame data."""
        # Use the highest-scoring frame as representative
        scores = data['scores']
        best_idx = scores.index(max(scores)) if scores else 0
        
        return SegmentResult(
            video_id=data['video_id'],
            video_name=data['video_name'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            caption=data['captions'][best_idx] if data['captions'] else '',
            score=max(scores) if scores else 0,
            frame_path=data['frame_paths'][best_idx] if data['frame_paths'] else ''
        )
    
