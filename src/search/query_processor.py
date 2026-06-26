from typing import List, Optional
import re


SPEECH_PATTERNS = re.compile(
    r'\b(say|said|saying|says|told|tell|talk|talked|talking|talks|'
    r'mention|mentioned|spoke|speak|speaking|spoken|'
    r'words?|voice|heard|hear|narrat|expla|describ|discuss)\b',
    re.IGNORECASE
)

VISUAL_PATTERNS = re.compile(
    r'\b(show|showed|shown|appear|appeared|visible|see|seen|look|'
    r'on[- ]screen|display|shown|footage|frame|image|visual|camera)\b',
    re.IGNORECASE
)

# Words to strip from the query before embedding so the topic is cleaner
INTENT_STRIP = re.compile(
    r'\b(when (did|does|do) (i|they|he|she|we|the user|someone)|'
    r'where did (i|they|he|she|the user)|'
    r'(i |they |he |she |the user )?(say|said|talk(ed)? about|mention(ed)?|spoke about|'
    r'discuss(ed)?|hear(d)?|explain(ed)?)|'
    r'show(ed)? (me )?(where|when)|'
    r'appear(ed)? (on ?screen|in (the )?video))\b',
    re.IGNORECASE
)


class QueryProcessor:
    """Process natural language queries for video search."""

    def __init__(self):
        self.segment_patterns = [
            r'when\s+does\s+(.+?)\s+happen',
            r'when\s+did\s+(.+?)\s+happen',
            r'find\s+(?:the\s+)?(?:moment|scene|part)\s+where\s+(.+)',
            r'show\s+(?:me\s+)?(?:where\s+)?(?:when\s+)?(.+)',
            r'(?:the\s+)?(?:moment|scene|part)\s+(?:where\s+)?(?:of\s+)?(.+)',
        ]

    def is_segment_query(self, query: str) -> bool:
        query_lower = query.lower()
        for pattern in self.segment_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def detect_source_intent(self, query: str) -> str:
        """Return 'speech', 'visual', or 'mixed' based on query phrasing."""
        has_speech = bool(SPEECH_PATTERNS.search(query))
        has_visual = bool(VISUAL_PATTERNS.search(query))
        if has_speech and not has_visual:
            return 'speech'
        if has_visual and not has_speech:
            return 'visual'
        return 'mixed'

    def extract_search_terms(self, query: str) -> str:
        """Strip intent phrasing and return the core topic for embedding."""
        cleaned = INTENT_STRIP.sub(' ', query)
        # Also remove leading question words
        cleaned = re.sub(
            r'^(find|search for|show me|where is|where are|when does|when did)\s+',
            '', cleaned.strip(), flags=re.IGNORECASE
        )
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(' ?.,!')
        return cleaned if cleaned else query

    def process_query(self, query: str) -> dict:
        search_terms = self.extract_search_terms(query)
        source_intent = self.detect_source_intent(query)
        video_id = None
        video_match = re.search(r'in\s+video\s+(\w+)', query.lower())
        if video_match:
            video_id = video_match.group(1)

        return {
            "is_segment_query": self.is_segment_query(query),
            "search_terms": search_terms,
            "source_intent": source_intent,
            "video_id": video_id,
            "original_query": query
        }