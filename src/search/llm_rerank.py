"""LLM compositional precision re-ranker — the final precision stage of /search.

WHY THIS EXISTS
The cross-encoder reranker (bge-reranker) scores a query against a caption as a
*bag of evidence*: it rewards a caption that CONTAINS the query's words even when
those words describe DIFFERENT objects in the scene. For a compositional query
like "mini pc with wooden accent" that lets a false match win — a scene caption
mentioning an (unrelated) "AI Mini PC" on a price tag and "keyboards on wooden
desks" in the background scores as high as the true match "a black mini PC with a
wooden base". No lexical or proximity rule can fix this: in one false positive the
words "Mac Mini" and "wooden table" sit in the *same sentence, a few words apart*,
yet the wood is the table the computers rest ON, not the computers themselves.
Only semantic reasoning distinguishes "mini PC WITH a wooden base" (the attribute
binds to the device) from "Mac Minis ON a wooden table" (background furniture).

WHAT IT DOES
Over the cross-encoder's top-K shortlist ONLY (bounded cost — one LLM call per
search), it asks the text LLM how well each scene ACTUALLY contains a SINGLE
subject matching the WHOLE query, with attributes bound to the same object. That
compositional confidence (0-1 per candidate) is blended into the final score by
the caller. This is text reasoning over the already-rich captions — NOT a second
visual pass (the captions already encode the vision) and NOT a hard gate — so it
is far cheaper and more reliable than re-looking at frames with a VLM.
"""
import json
import re

import httpx

RERANK_SYSTEM = (
    "You judge how well a video scene matches a search query, for a search engine. "
    "The query often describes ONE subject with several attributes (e.g. 'mini pc "
    "with wooden accent' means a single computer that ITSELF has wood on it). A scene "
    "matches ONLY if a SINGLE object or moment satisfies ALL parts of the query "
    "together. Attributes belonging to DIFFERENT objects do NOT count: a mini PC "
    "sitting on a wooden desk, or a mini PC near a separate wooden object, does NOT "
    "match 'mini pc with wooden accent' because the wood is not part of the computer. "
    "Synonyms and paraphrases DO match (a 'humanoid robot' matches 'android'). Judge "
    "ONLY from the provided text; do not assume details that are not stated. "
    "Score each numbered scene from 0.0 (no single subject matches the whole query) "
    "to 1.0 (a subject clearly matches every part of the query). "
    "Reply with ONLY a compact JSON array of objects like "
    '[{"i": 0, "score": 0.9}, {"i": 1, "score": 0.1}], one entry per scene, no prose.'
)


class LLMReranker:
    """One listwise LLM call that scores each shortlisted caption for compositional
    (attribute-to-object) match against the query. Reuses the text LLM endpoint."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def score(self, query: str, captions: list[str], timeout: float = 30.0):
        """Return a list of floats in [0,1] (one per caption) or None on failure.

        A None entry for an individual caption means the model did not score it —
        the caller should leave that candidate's score unchanged (never demote a
        candidate the model was silent about). Returns None entirely if the call
        or parse fails, so the caller can fall back to the cross-encoder ranking.
        """
        if not captions:
            return []
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(captions))
        user = (
            f"Query: {query}\n\n"
            f"Scenes:\n{numbered}\n\n"
            f"Score every scene 0.0-1.0 as instructed."
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": RERANK_SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 700,
                        "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[llm_rerank] call failed: {e}")
            return None
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            print(f"[llm_rerank] no JSON array in response: {raw[:200]!r}")
            return None
        try:
            arr = json.loads(m.group(0))
        except Exception as e:
            print(f"[llm_rerank] JSON parse failed: {e}")
            return None
        scores: list = [None] * len(captions)
        for obj in arr:
            try:
                i = int(obj["i"])
                sc = float(obj["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(captions):
                scores[i] = max(0.0, min(1.0, sc))
        return scores
