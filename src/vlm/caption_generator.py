import httpx
import base64
from typing import Optional
from pathlib import Path


class CaptionGenerator:
    """Generate captions for video frames using vLLM OpenAI-compatible API (LLaVA)."""
    
    def __init__(self, vllm_url: str, model: str = "llava-hf/llava-v1.6-vicuna-7b-hf"):
        self.vllm_url = vllm_url.rstrip('/')
        self.model = model
    
    async def caption_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail for video indexing."
    ) -> str:
        """Generate a caption for an image."""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
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
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    
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
    
    def __init__(self, vllm_url: str, model: str = "llava-hf/llava-v1.6-vicuna-7b-hf"):
        self.generator = CaptionGenerator(vllm_url, model=model)
    
    async def caption_frame(self, frame_path: str, context: str = '') -> str:
        """Generate caption for a single frame."""
        prompt = 'Describe this frame from a video in detail.'
        if context:
            prompt = f'Context about this video: {context}\n\nDescribe this frame from the video in detail, keeping the context in mind.'
        return await self.generator.caption_image(frame_path, prompt=prompt)
    
    async def caption_frames(self, frame_paths: list[str]) -> list[str]:
        """Generate captions for multiple frames."""
        return await self.generator.caption_images_batch(
            frame_paths,
            prompt="Briefly describe this video frame."
        )