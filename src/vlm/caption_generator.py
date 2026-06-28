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
        if context:
            prompt = (
                f'Background (may help identify subjects — ignore it if it does not '
                f'match this frame): {context}\n\n'
                'Describe only what is actually visible in this video frame in 1-2 plain '
                'sentences. Include the main subject, action, and setting. Do not assume '
                'the background applies if the frame shows something else. '
                'No lists, no headers, no markdown.'
            )
        else:
            prompt = (
                'Describe what is happening in this video frame in 1-2 plain sentences. '
                'Include the main subject, action, and setting. No lists, no headers, no markdown.'
            )
        return await self.generator.caption_image(frame_path, prompt=prompt)
    
    async def caption_frames(self, frame_paths: list[str]) -> list[str]:
        """Generate captions for multiple frames."""
        return await self.generator.caption_images_batch(
            frame_paths,
            prompt="Briefly describe this video frame."
        )