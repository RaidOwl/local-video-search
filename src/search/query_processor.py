from typing import List, Optional
import re


class QueryProcessor:
    """Process natural language queries for video search."""
    
    def __init__(self):
        # Common patterns for segment-level queries
        self.segment_patterns = [
            r'when\s+does\s+(.+?)\s+happen',
            r'when\s+did\s+(.+?)\s+happen',
            r'find\s+(?:the\s+)?(?:moment|scene|part)\s+where\s+(.+)',
            r'show\s+(?:me\s+)?(?:where\s+)?(?:when\s+)?(.+)',
            r'(?:the\s+)?(?:moment|scene|part)\s+(?:where\s+)?(?:of\s+)?(.+)',
        ]
    
    def is_segment_query(self, query: str) -> bool:
        """Check if query is asking about a specific segment within a video."""
        query_lower = query.lower()
        for pattern in self.segment_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def extract_search_terms(self, query: str) -> str:
        """Extract the core search terms from a query."""
        # Remove question words and common prefixes
        cleaned = query.lower()
        prefixes_to_remove = [
            r'^find\s+',
            r'^search\s+for\s+',
            r'^show\s+me\s+',
            r'^where\s+is\s+',
            r'^where\s+are\s+',
            r'^when\s+(?:does|did)\s+\w+\s+happen\s+',
            r'^find\s+(?:the\s+)?(?:moment|scene|part)\s+where\s+',
        ]
        
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned)
        
        return cleaned.strip()
    
    def process_query(self, query: str) -> dict:
        """
        Process a query and return structured information.
        
        Returns:
            {
                "is_segment_query": bool,
                "search_terms": str,
                "video_id": Optional[str] - if specified in query
            }
        """
        is_segment = self.is_segment_query(query)
        search_terms = self.extract_search_terms(query)
        
        # Check if video is specified in query
        video_id = None
        video_match = re.search(r'in\s+video\s+(\w+)', query.lower())
        if video_match:
            video_id = video_match.group(1)
        
        return {
            "is_segment_query": is_segment,
            "search_terms": search_terms,
            "video_id": video_id,
            "original_query": query
        }