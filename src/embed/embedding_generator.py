import httpx
from typing import List
import uuid
import re


def _clean_for_embed(text: str) -> str:
    """Strip markdown formatting so verbose captions embed cleanly against short queries."""
    # Remove headers (### Heading)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove bold/italic (**text**, *text*, __text__)
    text = re.sub(r'[*_]{1,3}([^*_\n]+)[*_]{1,3}', r'\1', text)
    # Remove bullet points
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class EmbeddingGenerator:
    """Generate embeddings using vLLM OpenAI-compatible API with bge-m3."""
    
    def __init__(self, vllm_url: str, model: str = "BAAI/bge-m3"):
        self.vllm_url = vllm_url.rstrip('/')
        self.model = model
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.vllm_url}/v1/embeddings",
                json={"model": self.model, "input": text}
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
    
    async def embed_batch(self, items: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.vllm_url}/v1/embeddings",
                json={"model": self.model, "input": items}
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


class EmbeddingService:
    """Service for managing embeddings and Qdrant storage."""
    
    def __init__(
        self,
        vllm_url: str,
        qdrant_url: str,
        collection_name: str = "video_frames",
        model: str = "BAAI/bge-m3"
    ):
        self.generator = EmbeddingGenerator(vllm_url, model=model)
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self._qdrant_client = None
    
    @property
    def qdrant_client(self):
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(url=self.qdrant_url)
        return self._qdrant_client
    
    async def ensure_collection(self, vector_size: int = 1024):
        """Ensure the collection exists in Qdrant."""
        from qdrant_client.models import Distance, VectorParams
        
        client = self.qdrant_client
        collections = [c.name for c in client.get_collections().collections]
        
        if self.collection_name not in collections:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
    
    async def index_frame(
        self,
        video_id: str,
        frame_path: str,
        timestamp: float,
        caption: str = None
    ):
        """Index a single frame with its embedding."""
        # Embed the caption text; fall back to frame path if no caption
        text_to_embed = caption if caption else frame_path
        embedding = await self.generator.embed_text(_clean_for_embed(text_to_embed))
        
        if not embedding:
            return None
        
        from qdrant_client.models import PointStruct
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{timestamp}"))
        
        payload = {
            "video_id": video_id,
            "frame_path": frame_path,
            "timestamp": timestamp,
            "caption": caption or ""
        }
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )]
        )
        
        return point_id
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        video_id: str = None,
        score_threshold: float = 0.45,
        source_intent: str = 'mixed'
    ) -> List[dict]:
        """Search for similar frames, optionally filtered by source type."""
        embedding = await self.generator.embed_text(_clean_for_embed(query))

        if not embedding:
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        filter_condition = None
        if video_id:
            filter_condition = Filter(
                must=[FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id)
                )]
            )

        # Fetch extra results so post-filtering by source type doesn't starve us
        fetch_limit = limit if source_intent == 'mixed' else limit * 3

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=filter_condition,
            limit=fetch_limit,
            score_threshold=score_threshold
        )

        hits = [
            {
                "id": hit.id,
                "score": hit.score,
                "video_id": hit.payload.get("video_id"),
                "frame_path": hit.payload.get("frame_path"),
                "timestamp": hit.payload.get("timestamp"),
                "caption": hit.payload.get("caption", "")
            }
            for hit in results
        ]

        # Post-filter by source intent
        if source_intent == 'speech':
            hits = [h for h in hits if h["caption"].startswith('[transcript]')]
        elif source_intent == 'visual':
            hits = [h for h in hits if not h["caption"].startswith('[transcript]')]

        return hits[:limit]