import asyncio
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import HTTPException, UploadFile
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import Settings
from src.embed.embedding_generator import EmbeddingIndexError, EmbeddingService
from src.local_models import resolve_local_config


def make_service(client, *, model="embed-model", backend="ollama", url="http://embedder", legacy=""):
    service = EmbeddingService(url, "http://unused", model=model, backend=backend, legacy_index_model=legacy)
    service._qdrant_client = client
    service.generator.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])
    return service


def make_archive(manifest, points):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("points.jsonl", "\n".join(json.dumps(point) for point in points))
    buffer.seek(0)
    return UploadFile(filename="library.zip", file=buffer)


class ModelConfigurationTests(unittest.TestCase):
    def test_compose_settings_in_env_do_not_break_api_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("HARDWARE_TYPE=cpu\nVIDEO_DIR_1=/videos\nAPP_DATA_DIR=/library\n")
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=env_file)
            self.assertEqual(settings.ai_backend, "ollama")

    def test_backend_mapping_and_instruct_tags(self):
        for hardware, backend in {
            "nvidia": "ollama", "amd": "ollama", "cpu": "ollama",
            "apple": "ollama-native", "native": "ollama-native",
            "nvidia-vllm": "vllm", "amd-vllm": "vllm", "custom": "custom",
        }.items():
            with self.subTest(hardware=hardware):
                self.assertEqual(Settings(_env_file=None, hardware_type=hardware).ai_backend, backend)
        for backend in ("ollama", "ollama-native"):
            for tier in ("low", "mid"):
                cfg = resolve_local_config(backend, "qwen", tier)
                self.assertEqual(cfg["embed_model"], "qwen3-embedding:4b")
                self.assertEqual(cfg["vlm_model"], f"qwen3-vl:{'4b' if tier == 'low' else '8b'}-instruct")


class EmbeddingIndexTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.service = make_service(self.client)

    def tearDown(self):
        self.client.close()

    async def index_one(self):
        return await self.service.index_frame("video", "frame.jpg", 1.0, "red bicycle")

    async def test_fresh_index_records_identity_and_searches(self):
        point_id = await self.index_one()
        metadata = self.client.retrieve(self.service.metadata_collection, ids=[0], with_payload=True)
        self.assertEqual(metadata[0].payload["identity"], self.service.index_identity)
        hits = await self.service.search("bicycle", score_threshold=0)
        self.assertEqual([hit["id"] for hit in hits], [point_id])

    async def test_same_dimension_different_model_or_backend_is_blocked(self):
        await self.index_one()
        for other in (make_service(self.client, model="other"), make_service(self.client, backend="vllm")):
            with self.subTest(identity=other.index_identity):
                with self.assertRaises(EmbeddingIndexError):
                    await other.search("bicycle")
                with self.assertRaises(EmbeddingIndexError):
                    await other.index_frame("video", "other.jpg", 2, "blue bicycle")
                other.generator.embed_text.assert_not_awaited()
        self.assertEqual(self.client.count(self.service.collection_name).count, 1)

    async def test_dimension_mismatch_is_rejected_even_for_same_identity(self):
        await self.index_one()
        self.service.generator.embed_text = AsyncMock(return_value=[1.0, 0.0])
        with self.assertRaisesRegex(EmbeddingIndexError, "3 dimensions"):
            await self.service.search("bicycle")
        with self.assertRaises(EmbeddingIndexError):
            await self.service.index_frame("video", "other.jpg", 2, "blue bicycle")
        self.assertEqual(self.client.count(self.service.collection_name).count, 1)

    async def test_unknown_legacy_index_requires_explicit_matching_adoption(self):
        self.client.create_collection(self.service.collection_name, vectors_config=VectorParams(size=3, distance=Distance.COSINE))
        self.client.upsert(self.service.collection_name, points=[PointStruct(id=1, vector=[1, 0, 0])])
        with self.assertRaisesRegex(EmbeddingIndexError, "no recorded model"):
            self.service.assert_index_compatible()
        with self.assertRaises(EmbeddingIndexError):
            make_service(self.client, legacy="wrong-model").assert_index_compatible()
        adopted = make_service(self.client, legacy="embed-model")
        adopted.assert_index_compatible()
        self.assertEqual(self.client.count(self.service.collection_name).count, 1)
        make_service(self.client).assert_index_compatible()
        with self.assertRaises(EmbeddingIndexError):
            make_service(self.client, backend="vllm", legacy="embed-model").assert_index_compatible()

    async def test_empty_collection_can_be_resized_and_reassigned(self):
        await self.service.ensure_collection(vector_size=2)
        self.service._write_index_identity({c.name for c in self.client.get_collections().collections})
        other = make_service(self.client, model="new-model")
        await other.index_frame("video", "frame.jpg", 1, "bicycle")
        self.assertEqual(self.client.get_collection(other.collection_name).config.params.vectors.size, 3)
        identity = self.client.retrieve(other.metadata_collection, ids=[0], with_payload=True)[0].payload["identity"]
        self.assertEqual(identity, other.index_identity)

    async def test_custom_endpoint_identity_and_ollama_host_equivalence(self):
        custom = make_service(self.client, backend="custom", url="http://first")
        await custom.index_frame("video", "frame.jpg", 1, "bicycle")
        with self.assertRaises(EmbeddingIndexError):
            make_service(self.client, backend="custom", url="http://second").assert_index_compatible()
        other_client = QdrantClient(":memory:")
        try:
            await make_service(other_client, backend="ollama-native").index_frame("v", "", 0, "text")
            make_service(other_client, backend="ollama").assert_index_compatible()
        finally:
            other_client.close()


class IndexApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from src.api import server
        self.server = server
        self.client = QdrantClient(":memory:")
        self.original = make_service(self.client)
        await self.original.index_frame("video", "frame.jpg", 1, "bicycle")
        self.incompatible = make_service(self.client, model="different-model")
        self.db = AsyncMock()
        self.db.get_video.return_value = SimpleNamespace(id="video", path="/video.mp4", status="analyzed", context="")
        self.db.get_frame.return_value = SimpleNamespace(video_id="video", caption="old", embedding_id="point")
        self.db.get_all_videos.return_value = []
        self.service_patch = patch.object(server, "embed_service", self.incompatible)
        self.db_patch = patch.object(server, "db", self.db)
        self.service_patch.start()
        self.db_patch.start()

    async def asyncTearDown(self):
        self.db_patch.stop()
        self.service_patch.stop()
        self.client.close()

    async def test_reprocessing_retranscription_and_edits_do_not_delete_data(self):
        server = self.server
        with self.assertRaises(EmbeddingIndexError):
            await server._run_analyze("video", server.AnalyzeRequest())
        with self.assertRaises(EmbeddingIndexError):
            await server.retranscribe_video("video")
        with self.assertRaises(EmbeddingIndexError):
            await server.update_frame_caption("video", "frame", server.FrameUpdateRequest(caption="new"))
        self.db.update_video.assert_not_awaited()
        self.db.update_frame.assert_not_awaited()
        self.assertEqual(self.client.count("video_frames").count, 1)

    async def test_settings_remain_accessible_if_qdrant_is_unavailable(self):
        error = ResponseHandlingException(OSError("Qdrant offline"))
        with patch.object(self.incompatible, "assert_index_compatible", side_effect=error):
            settings = await self.server.get_settings_endpoint()
        self.assertIn("Cannot verify the embedding index", settings["index_warning"])

    async def test_missing_nvidia_tool_has_a_typed_hardware_warning(self):
        with patch.object(self.server.settings, "hardware_type", "nvidia"), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            hardware = await self.server.hardware_status()
        self.assertIn("nvidia-smi unavailable", hardware["gpu_error"])
        self.assertNotIn("gpus", hardware)

    async def test_background_jobs_guard_before_purge(self):
        server = self.server
        with patch.object(server, "_wait_for_ai_services", new=AsyncMock(return_value=True)):
            await server.process_video("/video.mp4", "video")
        await server._retranscribe_audio("video", "/video.mp4")
        self.db.delete_visual_frames_for_video.assert_not_awaited()
        self.db.delete_transcript_frames_for_video.assert_not_awaited()
        self.db.delete_segment_text.assert_not_awaited()
        self.assertEqual(self.client.count("video_frames").count, 1)

    async def test_cancelling_while_waiting_does_not_purge_old_points(self):
        server = self.server
        started = asyncio.Event()

        async def waiting(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        with patch.object(server, "_wait_for_ai_services", new=waiting):
            task = asyncio.create_task(server.process_video("/video.mp4", "video"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.db.delete_segment_text.assert_not_awaited()
        self.assertEqual(self.client.count("video_frames").count, 1)

    async def test_import_rejects_mismatches_before_overwriting(self):
        server = self.server
        point = {"id": str(uuid.uuid4()), "vector": [1, 0, 0], "payload": {"video_id": "video"}}
        manifest = {"vector_size": 3, "embed_model": "different-model",
                    "embedding_identity": self.incompatible.index_identity,
                    "videos": [{"id": "video"}], "frames": []}
        with self.assertRaises(EmbeddingIndexError):
            await server.import_library(make_archive(manifest, [point]))
        self.db.delete_video.assert_not_awaited()
        self.assertEqual(self.client.count("video_frames").count, 1)

        with patch.object(server, "embed_service", self.original):
            manifest["embed_model"] = self.original.generator.model
            with self.assertRaises(HTTPException) as error:
                await server.import_library(make_archive(manifest, [point]))
            self.assertEqual(error.exception.status_code, 400)
            manifest["embedding_identity"] = self.original.index_identity
            point["vector"] = [1, 0]
            with self.assertRaises(HTTPException):
                await server.import_library(make_archive(manifest, [point]))
        self.db.delete_video.assert_not_awaited()
        self.assertEqual(self.client.count("video_frames").count, 1)

    async def test_import_rejects_unexpected_model_dimension_before_overwriting(self):
        server = self.server
        point = {"id": str(uuid.uuid4()), "vector": [1, 0], "payload": {"video_id": "video"}}
        manifest = {"vector_size": 2, "embed_model": self.original.generator.model,
                    "embedding_identity": self.original.index_identity,
                    "videos": [{"id": "video"}], "frames": []}
        with patch.object(server, "embed_service", self.original):
            with self.assertRaisesRegex(HTTPException, "produces 3"):
                await server.import_library(make_archive(manifest, [point]))
        self.db.delete_video.assert_not_awaited()
        self.assertEqual(self.client.count("video_frames").count, 1)

    async def test_matching_archive_imports(self):
        server = self.server
        point_id = str(uuid.uuid4())
        point = {"id": point_id, "vector": [1, 0, 0], "payload": {"video_id": "video", "caption": "new"}}
        manifest = {"vector_size": 3, "embed_model": self.original.generator.model,
                    "embedding_identity": self.original.index_identity,
                    "videos": [{"id": "video", "name": "video", "path": "/video.mp4"}], "frames": []}
        self.db.get_video_by_hash.return_value = None
        with patch.object(server, "embed_service", self.original), patch.object(server, "_do_source_backfill"):
            result = await server.import_library(make_archive(manifest, [point]))
        self.assertEqual(result["imported"], 1)
        self.assertEqual(self.client.count("video_frames").count, 1)
        self.assertEqual(str(self.client.retrieve("video_frames", ids=[point_id])[0].id), point_id)

    async def test_legacy_archive_requires_an_explicit_model_assertion(self):
        server = self.server
        point = {"id": str(uuid.uuid4()), "vector": [1, 0, 0], "payload": {"video_id": "video"}}
        manifest = {"vector_size": 3, "embed_model": self.original.generator.model,
                    "videos": [{"id": "video", "name": "video", "path": "/video.mp4"}], "frames": []}
        self.db.get_video_by_hash.return_value = None
        with patch.object(server, "embed_service", self.original):
            with self.assertRaisesRegex(HTTPException, "no embedding backend identity"):
                await server.import_library(make_archive(manifest, [point]))
            self.db.delete_video.assert_not_awaited()
            with patch.object(server.settings, "legacy_index_embed_model", "embed-model"), patch.object(server, "_do_source_backfill"):
                result = await server.import_library(make_archive(manifest, [point]))
                self.assertEqual(result["imported"], 1)

    async def test_http_routes_report_index_conflicts(self):
        server = self.server
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for method, path in (("GET", "/search?q=bicycle"), ("POST", "/reprocess/video")):
                response = await client.request(method, path)
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["code"], "embedding_index_incompatible")

    async def test_model_status_and_readiness_require_the_configured_model(self):
        server = self.server
        cfg = {"embed_url": "http://embedder", "embed_model": "expected",
               "vlm_url": "http://vision", "vlm_model": "vision",
               "llm_url": "http://vision", "llm_model": "vision"}
        real_async_client = httpx.AsyncClient

        def client_factory(**kwargs):
            def handle(request):
                models = ["different"] if request.url.host == "embedder" else ["vision"]
                return httpx.Response(200, json={"data": [{"id": name} for name in models]})
            return real_async_client(transport=httpx.MockTransport(handle), **kwargs)

        with patch.object(server, "_effective_cfg", return_value=cfg), patch.object(httpx, "AsyncClient", side_effect=client_factory):
            down = await server._ai_services_ready(need_vlm=True)
            status = await server.model_status()
        self.assertEqual(down, ["embeddings"])
        self.assertFalse(status["services"][0]["ok"])
        self.assertIn("expected", status["services"][0]["reason"])
        self.assertTrue(status["services"][1]["ok"])

    async def test_collection_size_missing_is_distinct_from_qdrant_failure(self):
        server = self.server
        empty_client = QdrantClient(":memory:")
        try:
            with patch.object(server, "embed_service", make_service(empty_client)):
                self.assertIsNone(server._collection_vector_size())
            failing_service = Mock()
            failing_service.qdrant_client.get_collections.side_effect = RuntimeError("qdrant unavailable")
            with patch.object(server, "embed_service", failing_service):
                with self.assertRaisesRegex(RuntimeError, "qdrant unavailable"):
                    server._collection_vector_size()
        finally:
            empty_client.close()

    async def test_empty_archive_without_collection_imports(self):
        server = self.server
        empty_client = QdrantClient(":memory:")
        try:
            with patch.object(server, "embed_service", make_service(empty_client)):
                result = await server.import_library(make_archive(
                    {"vector_size": None, "videos": [], "frames": []}, []
                ))
                self.assertEqual(result["imported"], 0)
        finally:
            empty_client.close()


if __name__ == "__main__":
    unittest.main()
