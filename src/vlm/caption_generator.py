import httpx
import asyncio
import base64
from typing import Optional


class CaptionGenerator:
    """Generate captions for video frames using vLLM OpenAI-compatible API (LLaVA)."""
    
    def __init__(self, base_url: str, model: str = "llava-hf/llava-v1.6-vicuna-7b-hf"):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    async def caption_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail for video indexing."
    ) -> str:
        """Generate a caption for an image. Retries up to 3 times on transient errors."""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 300
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                await asyncio.sleep(5 * attempt)  # 5s, 10s backoff
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                last_exc = e
                # Don't retry on client errors that indicate bad input (400, 422)
                if e.response.status_code in (400, 422):
                    raise
                # A 404 means the model isn't available at this endpoint — almost
                # always an ollama model that hasn't been pulled. Retrying can't
                # help, so fail fast with an actionable message naming the model.
                if e.response.status_code == 404:
                    raise RuntimeError(
                        f"VLM model '{self.model}' not found at {self.base_url} (HTTP 404). "
                        f"If you're using an ollama backend, pull it first: "
                        f"ollama pull {self.model}"
                    ) from e
                print(f"VLM caption attempt {attempt + 1}/3 failed ({e.response.status_code}): {e}")
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                print(f"VLM caption attempt {attempt + 1}/3 failed (connection): {e}")

        raise RuntimeError(f"VLM captioning failed after 3 attempts") from last_exc
    
    async def caption_images_batch(
        self,
        image_paths: list[str],
        prompt: str = "Describe this image briefly."
    ) -> list[str]:
        """Generate captions for multiple images."""
        captions = []
        for image_path in image_paths:
            try:
                caption = await self.caption_image(image_path, prompt)
                captions.append(caption)
            except Exception as e:
                print(f"Error captioning {image_path}: {e}")
                captions.append("")
        return captions


class VLMService:
    """Service for managing VLM captioning."""
    
    def __init__(self, base_url: str, model: str = "llava-hf/llava-v1.6-vicuna-7b-hf"):
        self.generator = CaptionGenerator(base_url, model=model)
    
    async def caption_frame(self, frame_path: str, context: str = '') -> str:
        """Generate caption for a single frame."""
        # The caption is the ONLY text indexed for visual search, so it must name
        # objects with the attributes people actually search by. A generic
        # "a person using a laptop" makes "blue laptop" unsearchable even when a
        # blue laptop is clearly in frame — so we explicitly ask the VLM to record
        # distinguishing details (color, brand/model, on-screen or label text,
        # material, count) for each notable object, in plain prose (lists/markdown
        # break the reranker and the short-segment heuristics downstream).
        role = (
            'You are writing the searchable description for one frame of a video '
            'search engine. People will look for specific things by their visible '
            'attributes, so name each notable object and give its distinguishing '
            'details — color, brand or model, any readable on-screen or label text, '
            'material, and how many there are (e.g. "a blue laptop, a yellow laptop, '
            'and a silver laptop" rather than just "laptops"). Then state the main '
            'subject, what is happening, and the setting. Write 1-3 plain sentences '
            'of prose with no lists, headers, or markdown. Describe only what is '
            'actually visible; never guess or invent details.'
        )
        if context:
            prompt = (
                f'Background (may help identify subjects — ignore it if it does not '
                f'match this frame): {context}\n\n'
                + role +
                ' Do not assume the background applies if the frame shows something else.'
            )
        else:
            prompt = role
        return await self.generator.caption_image(frame_path, prompt=prompt)
    
    async def caption_frames(self, frame_paths: list[str]) -> list[str]:
        """Generate captions for multiple frames."""
        return await self.generator.caption_images_batch(
            frame_paths,
            prompt="Briefly describe this video frame."
        )