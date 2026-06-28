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
        caption: str = None,
        _point_id: str = None,
    ):
        """Index a single frame with its embedding."""
        # Embed the caption text; fall back to frame path if no caption
        text_to_embed = caption if caption else frame_path
        embedding = await self.generator.embed_text(_clean_for_embed(text_to_embed))

        if not embedding:
            return None

        from qdrant_client.models import PointStruct

        point_id = _point_id or str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{timestamp}"))

        payload = {
            "video_id": video_id,
            "frame_path": frame_path,
            "timestamp": timestamp,
            "caption": caption or "",
            # Transcript segments are indexed with a "[transcript] " caption
            # prefix; tag them so search can filter by source server-side.
            "is_transcript": bool(caption and caption.startswith('[transcript]')),
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
    
    def export_points(self, video_id: str) -> List[dict]:
        """Scroll every Qdrant point for a video (with vectors) for library export."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        out: List[dict] = []
        offset = None
        flt = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
        while True:
            points, offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                with_payload=True, with_vectors=True,
                limit=256, offset=offset,
            )
            for p in points:
                out.append({"id": p.id, "vector": p.vector, "payload": p.payload})
            if offset is None:
                break
        return out

    def import_points(self, points: List[dict]):
        """Batch-upsert raw points (id + vector + payload) from a library export."""
        if not points:
            return
        from qdrant_client.models import PointStruct
        structs = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload") or {})
            for p in points if p.get("vector")
        ]
        BATCH = 256
        for i in range(0, len(structs), BATCH):
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=structs[i:i + BATCH]
            )

    def delete_video_points(self, video_id: str):
        """Remove all Qdrant points for a video (used before an overwrite-import)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
        )

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

        # Filter by source intent *inside* the query so the candidate pool isn't
        # dominated by the unwanted source. Short queries (e.g. "dog") align more
        # strongly with short transcript text than verbose visual captions, so
        # post-filtering after a small over-fetch used to starve visual recall.
        must = []
        if video_id:
            must.append(FieldCondition(key="video_id", match=MatchValue(value=video_id)))
        if source_intent == 'visual':
            must.append(FieldCondition(key="is_transcript", match=MatchValue(value=False)))
        elif source_intent == 'speech':
            must.append(FieldCondition(key="is_transcript", match=MatchValue(value=True)))
        filter_condition = Filter(must=must) if must else None

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=filter_condition,
            limit=limit,
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

        # Belt-and-suspenders: also post-filter by caption prefix in case any
        # legacy points predate the is_transcript flag (before backfill runs).
        if source_intent == 'speech':
            hits = [h for h in hits if h["caption"].startswith('[transcript]')]
        elif source_intent == 'visual':
            hits = [h for h in hits if not h["caption"].startswith('[transcript]')]

        return hits[:limit]