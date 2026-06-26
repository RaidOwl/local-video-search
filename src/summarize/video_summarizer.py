import httpx
from typing import List


class VideoSummarizer:
    """Generate summaries for videos using a remote LLM (vLLM)."""
    
    def __init__(self, vllm_url: str, model: str = "Qwen/Qwen3.5-9B"):
        self.vllm_url = vllm_url.rstrip('/')
        self.model = model
    
    async def summarize(
        self,
        captions: List[dict],
        video_name: str = ""
    ) -> str:
        """
        Generate a summary from frame captions.
        
        Args:
            captions: List of dicts with 'timestamp' and 'caption' keys
            video_name: Name of the video
            max_tokens: Maximum length of summary
        
        Returns:
            Generated summary text
        """
        if not captions:
            return "No captions available for summarization."

        sorted_captions = sorted(captions, key=lambda x: x.get('timestamp', 0))

        visual = [c for c in sorted_captions if not c.get('caption', '').startswith('[transcript]')]
        transcript = [c for c in sorted_captions if c.get('caption', '').startswith('[transcript]')]

        def sample(items, n):
            if len(items) <= n:
                return items
            step = len(items) / n
            return [items[int(i * step)] for i in range(n)]

        # 5 visual frames for scene context + up to 60 transcript segments for spoken content
        selected = sample(visual, 5) + sample(transcript, 60)
        selected.sort(key=lambda x: x.get('timestamp', 0))

        caption_texts = []
        for c in selected:
            ts = c.get('timestamp', 0)
            text = c.get('caption', '')
            if text.startswith('[transcript]'):
                caption_texts.append(f"[{self._format_timestamp(ts)}] {text[len('[transcript] '):]}")
            else:
                caption_texts.append(f"[{self._format_timestamp(ts)}] (visual) {text}")

        context = "\n".join(caption_texts)
        
        system_prompt = (
            "You are a video summarization assistant. "
            "Write a concise, human-readable summary in plain prose — no bullet points, no headers, no markdown, no reasoning steps. "
            "2-4 sentences maximum. Just the summary."
        )

        title_part = f' titled "{video_name}"' if video_name else ''
        user_prompt = (
            f"Summarize this video{title_part} "
            f"based on these timestamped frame captions:\n\n{context}"
        )
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            response.raise_for_status()
            data = response.json()
            raw = data['choices'][0]['message']['content'].strip()
            # Strip any leaked thinking block just in case
            if '<think>' in raw:
                raw = raw[raw.rfind('</think>') + len('</think>'):].strip()
            return raw
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class SummarizationService:
    """Service for managing video summarization."""
    
    def __init__(self, vllm_url: str, model: str = "Qwen/Qwen3-8B"):
        self.summarizer = VideoSummarizer(vllm_url, model=model)
    
    async def summarize_video(
        self,
        captions: List[dict],
        video_name: str = ""
    ) -> str:
        """Generate a summary for a video."""
        return await self.summarizer.summarize(captions, video_name)