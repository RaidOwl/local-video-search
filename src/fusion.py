"""Fusion: the 3rd analysis pass.

After frames are captioned (VLM) and audio is transcribed (Whisper), this pass
groups both channels into short time windows and asks the LLM to fuse each
window's on-screen captions + spoken transcript + the user's context into ONE
concise, factual description. That fused text is embedded and indexed alongside
the raw captions/transcripts (never replacing them), which lets cross-modal
moments — where the concept is split across what is shown and what is said —
rank correctly (e.g. screen shows a "transparent cylindrical display", audio
says "hologram" → fused "a man presenting in front of a holographic display").
"""
import httpx

FUSION_SYSTEM = (
    "You merge multimodal evidence from ONE short segment of a video into a single "
    "concise, factual description for a search index. You are given lines tagged "
    "(shown) — what is visible on screen — and (spoken) — what is said — plus optional "
    "user context about the video. Write 1-3 plain sentences naming the concrete "
    "subjects, objects and actions actually present, combining both channels (e.g. if "
    "the screen shows a transparent display and the audio says 'hologram', call it a "
    "hologram). Use ONLY the provided evidence and context. Do NOT invent anything not "
    "stated. No lists, no markdown, no preamble — just the description."
)


class FusionService:
    """Builds time windows from a video's frames and fuses each via the LLM."""

    def __init__(self, base_url: str, model: str, window: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.window = window

    def build_windows(self, frames: list) -> list[dict]:
        """Group frames (DB rows with .timestamp/.frame_path/.caption) into
        non-overlapping ~``window``-second buckets. Each returned dict has the
        window start ``lo``, a representative visual frame path ``rep_fp`` and the
        ``lines`` of (shown)/(spoken) evidence. Windows with no content are
        dropped. Returns [] if there is no visual content to fuse."""
        visual = sorted([f for f in frames if f.frame_path], key=lambda f: f.timestamp)
        if not visual:
            return []
        spoken = sorted([f for f in frames if not f.frame_path], key=lambda f: f.timestamp)
        dur = max((f.timestamp for f in frames), default=0.0) + self.window

        windows: list[dict] = []
        t = 0.0
        while t < dur:
            lo, hi = t, t + self.window
            lines: list[str] = []
            rep_fp = ""
            for f in visual:
                if lo <= f.timestamp < hi and f.caption:
                    lines.append(f"(shown) {f.caption}")
                    rep_fp = rep_fp or f.frame_path
            for f in spoken:
                if lo <= f.timestamp < hi and f.caption:
                    lines.append(f"(spoken) {f.caption.replace('[transcript] ', '')}")
            if lines and rep_fp:
                windows.append({"lo": lo, "rep_fp": rep_fp, "lines": lines})
            t += self.window
        return windows

    async def fuse(self, context: str, lines: list[str]) -> str:
        """Fuse one window's evidence lines into a single factual description."""
        user = ""
        if context:
            user += f"User context about the video: {context}\n\n"
        user += "Segment evidence:\n" + "\n".join(lines) + "\n\nWrite the fused description now."
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": FUSION_SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
        return raw
