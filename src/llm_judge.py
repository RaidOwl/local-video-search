"""LLM-as-judge re-ranking for Enhanced search.

The normal search pipeline scores each caption in isolation with a cross-encoder,
which is blind to a moment whose meaning is split across channels (the robot is
SHOWN at 13:05 while a man is HEARD talking at 13:04) and bad at relational
queries ("a man talking TO a robot"). Enhanced search fixes this on the shortlist:
for each candidate moment we hand the LLM the WHOLE moment as text — every
on-screen caption plus the spoken transcript in that time window — and ask it to
judge, with both channels in view, whether the moment genuinely matches the query.

This re-describes nothing and never touches the frames: it reuses the captions the
VLM already wrote, so it is a pure text reasoning step over evidence we already
have. It is intentionally gated to a small shortlist behind an explicit toggle.
"""

import json
import re
import httpx
from typing import List, Dict


class LLMJudge:
    """Score candidate moments against a query using a chat LLM as the judge."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def judge(self, query: str, moments: List[dict]) -> Dict[int, float]:
        """Return {moment_index: confidence 0..1} for moments the LLM scored.

        Moments NOT present in the returned dict were not judged (parse failure or
        omitted by the model); callers should leave those scores untouched rather
        than assume a zero. A moment the model marks as a non-match is clamped low
        so it falls out of the confident tiers.
        """
        if not moments:
            return {}

        blocks = []
        for i, m in enumerate(moments):
            blocks.append(f"MOMENT {i} ({m.get('label', '')}):\n{m.get('text', '')}".strip())
        joined = "\n\n".join(blocks)
        n = len(moments)

        system_prompt = (
            "You judge whether each candidate video moment is what the user is looking for. "
            "Each moment is timestamped evidence from one short span of a video: lines tagged "
            "(shown) describe what is visible on screen; lines tagged (spoken) are the transcript "
            "of what is said. Judge the moment as a WHOLE, combining what is shown and what is said.\n"
            "Match on MEANING, not exact words. Synonyms and paraphrases count: a 'person', 'guy' "
            "or 'woman' satisfies 'man'/'someone'; 'interacting with', 'greeting', 'addressing' or "
            "'speaking to' all satisfy 'talking to'. What matters is that the actual things and the "
            "relationship the query describes are genuinely present in the combined evidence — e.g. "
            "for 'man talking to a robot' the evidence must really involve a robot AND a person "
            "engaging with it, in either channel. Reject only coincidental overlap where the key "
            "subject the user named is not actually there.\n"
            "confidence (0.0-1.0) is how well the moment matches: ~0.9 a clear match, ~0.7 a likely "
            "match, <=0.4 not a match.\n"
            f"Output ONLY a JSON array with EXACTLY {n} objects, one per moment, in order 0..{n - 1}: "
            '[{"i":0,"match":true,"confidence":0.0}, ...]. Include every moment. No other text.'
        )
        user_prompt = (
            f'User search query: "{query}"\n\n'
            f"Candidate moments (judge all {n}):\n\n{joined}\n\n"
            f"Return the JSON array of {n} objects now."
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 600,
                        "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return {}

        # Some models leak a <think> block even with thinking disabled.
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>") + len("</think>"):].strip()

        # Extract the first JSON array from the reply, tolerating stray prose.
        match = re.search(r"\[.*\]", raw, re.S)
        payload = match.group(0) if match else raw
        try:
            arr = json.loads(payload)
        except Exception:
            return {}

        out: Dict[int, float] = {}
        if not isinstance(arr, list):
            return {}
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj.get("i"))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(moments)):
                continue
            try:
                conf = float(obj.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            # An explicit non-match cannot sit in the confident tiers, however high
            # the model rated its own confidence.
            if obj.get("match") is False:
                conf = min(conf, 0.45)
            out[idx] = conf
        return out


class JudgeService:
    """Thin wrapper mirroring the other model services in this codebase."""

    def __init__(self, base_url: str, model: str):
        self.judge = LLMJudge(base_url, model=model)

    async def judge_moments(self, query: str, moments: List[dict]) -> Dict[int, float]:
        return await self.judge.judge(query, moments)
