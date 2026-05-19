from __future__ import annotations

import json
import os
import subprocess
import time
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIError


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    timeout_s: float = float(os.getenv("DEEPSEEK_TIMEOUT_S", "180"))
    max_retries: int = int(os.getenv("DEEPSEEK_MAX_RETRIES", "4"))
    retry_backoff_s: float = float(os.getenv("DEEPSEEK_RETRY_BACKOFF_S", "1.5"))


class ResearchAgent:
    """
    Core reasoning logic:
    - Extract innovation points from each source item
    - Self-correct / contradiction-check across sources
    - Compact/merge duplicates
    - Synthesize a final Markdown research report
    """

    def __init__(self, config: Optional[DeepSeekConfig] = None):
        if config is None:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("Missing DEEPSEEK_API_KEY in environment.")
            config = DeepSeekConfig(api_key=api_key)

        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout_s)

    def skill_summarize_urls(
        self,
        *,
        urls: List[str],
        length: str = "short",
        youtube_mode: str = "auto",
        extract_only: bool = False,
        max_output_tokens: Optional[int] = None,
        firecrawl: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Call the `summarize` CLI skill per `skills/SKILL.md`.

        - Uses `--json` for machine-readable output.
        - For YouTube links, uses `--youtube auto` (best-effort transcript fallback).
        - If extract_only=True, uses `--extract-only` (URLs only).
        """
        # Per project rule: failures should not break the loop.
        # If a single URL fails to summarize, return empty string for that URL.
        clean_urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        out: Dict[str, str] = {u: "" for u in clean_urls}
        model = model or os.getenv("SUMMARIZE_MODEL")  # optional; default handled by summarize itself

        for url in clean_urls:
            try:
                cmd: List[str] = ["summarize", url, "--json", "--length", str(length)]
                if model:
                    cmd += ["--model", model]

                host = (urlparse(url).netloc or "").lower()
                if "youtube.com" in host or "youtu.be" in host:
                    cmd += ["--youtube", youtube_mode]
                    if extract_only:
                        cmd += ["--extract-only"]

                if max_output_tokens is not None:
                    cmd += ["--max-output-tokens", str(int(max_output_tokens))]
                if firecrawl:
                    cmd += ["--firecrawl", firecrawl]

                p = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if p.returncode != 0:
                    out[url] = ""
                    continue

                raw = (p.stdout or "").strip()
                if not raw:
                    out[url] = ""
                    continue

                data = json.loads(raw)
                text = (
                    data.get("summary")
                    or data.get("text")
                    or data.get("content")
                    or data.get("output")
                    or data.get("result")
                )
                out[url] = text.strip() if isinstance(text, str) and text.strip() else ""
            except Exception:
                out[url] = ""
                continue

        return out

    def _chat_json(
        self,
        *,
        system: str,
        user: str,
        json_schema_hint: str,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        DeepSeek is OpenAI-compatible, but response_format JSON schema support may vary by model/provider.
        We enforce JSON via instruction + robust parsing.
        """
        content = (
            user
            + "\n\nReturn ONLY valid JSON (no markdown). "
            + "Follow this schema:\n"
            + json_schema_hint.strip()
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                )
                last_err = None
                break
            except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
                last_err = e
                if attempt >= self.config.max_retries:
                    raise
                # exponential backoff with cap
                sleep_s = min(30.0, self.config.retry_backoff_s * (2**attempt))
                time.sleep(sleep_s)

        if last_err is not None:
            raise last_err
        text = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Best-effort recovery if the model wrapped JSON with extra text.
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(text[start : end + 1])

    def extract_innovation_points(
        self,
        *,
        title: str,
        url: str,
        content: str,
        source: str,
        published_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        system = (
            "You are a senior research analyst for LLM/ML/NLP papers and technical talks. "
            "You extract concrete, verifiable innovation points and avoid hype. "
            "When unsure, explicitly mark uncertainty."
        )
        schema = """
{
  "title": "string",
  "url": "string",
  "source": "arxiv|youtube|bilibili|other",
  "published_at": "string|null",
  "innovation_points": [
    {
      "point": "string",
      "evidence": "string",
      "novelty_type": "method|dataset|analysis|system|theory|training|evaluation|application|other",
      "confidence": 0.0
    }
  ],
  "tags": ["string"],
  "key_claims": ["string"],
  "limitations": ["string"],
  "related_work_mentions": ["string"]
}
        """.strip()

        user = f"""
Input item:
- title: {title}
- url: {url}
- source: {source}
- published_at: {published_at}

Raw content (may be abstract/transcript/notes; may contain noise):
{content}

Task:
Extract the main innovation points. Each point must include an evidence snippet grounded in the provided content.
If a claim cannot be supported by the content, exclude it.
Confidence should be 0.0-1.0 based on evidential support and specificity.
        """.strip()

        return self._chat_json(system=system, user=user, json_schema_hint=schema, temperature=0.2)

    def contradiction_check(
        self,
        *,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Cross-item contradiction / vagueness check.

        Note: We request a confidence score and a brief rationale (not private chain-of-thought).
        """
        system = (
            "You are a meticulous fact-checking and consistency-checking assistant for research synthesis. "
            "Detect contradictions, ambiguous statements, and missing definitions. "
            "Prefer caution and ask to mark uncertain parts."
        )
        schema = """
{
  "overall_confidence": 0.0,
  "issues": [
    {
      "type": "contradiction|vague|missing_definition|unsupported_claim|numeric_inconsistency|terminology_mismatch",
      "description": "string",
      "affected_items": [{"title": "string", "url": "string"}],
      "suggested_resolution": "string"
    }
  ],
  "cleaned_claims": ["string"],
  "notes": "string"
}
        """.strip()

        user = f"""
You will be given extracted innovation summaries from multiple sources. Your job:
- Identify contradictions and vagueness across items.
- Produce a single, cleaned set of claims that can coexist without contradiction.
- For each issue, propose a resolution strategy (e.g., 'report both with attribution', 'downgrade to hypothesis').

IMPORTANT:
- Output an overall_confidence score (0.0-1.0) BEFORE final synthesis, based on evidence quality and consistency.
- Do NOT reveal private chain-of-thought. Provide only brief, user-facing rationale in notes.

Items JSON:
{json.dumps(items, ensure_ascii=False)}
        """.strip()

        return self._chat_json(system=system, user=user, json_schema_hint=schema, temperature=0.1)

    def compact_topics(
        self,
        *,
        items: List[Dict[str, Any]],
        cleaned_claims: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Merge duplicates across sources and normalize topic structure.
        """
        system = (
            "You are a synthesis engine that merges duplicate topics across papers and talks. "
            "Group by true conceptual similarity, not superficial keywords."
        )
        schema = """
{
  "topics": [
    {
      "topic_title": "string",
      "why_it_matters": "string",
      "merged_sources": [{"title": "string", "url": "string", "source": "string"}],
      "core_innovations": ["string"],
      "evidence_snippets": ["string"],
      "open_questions": ["string"],
      "who_should_care": ["string"],
      "confidence": 0.0
    }
  ]
}
        """.strip()

        user = f"""
Input items (each includes innovation_points, claims, limitations, etc.):
{json.dumps(items, ensure_ascii=False)}

If provided, prefer these cleaned_claims as the global-safe statements:
{json.dumps(cleaned_claims or [], ensure_ascii=False)}

Task:
- Merge duplicates across sources into topics.
- Each topic must cite the merged_sources and include evidence_snippets grounded in the items.
- Keep confidence 0.0-1.0 based on strength and agreement across sources.
        """.strip()

        return self._chat_json(system=system, user=user, json_schema_hint=schema, temperature=0.2)

    def synthesize_markdown_report(
        self,
        *,
        topics: List[Dict[str, Any]],
        contradiction_report: Optional[Dict[str, Any]] = None,
        days: int = 3,
    ) -> str:
        system = (
            "You write compact, high-signal research digests for ML practitioners. "
            "Be specific, cite sources, and separate facts from hypotheses."
        )
        issues = (contradiction_report or {}).get("issues", [])
        overall_conf = (contradiction_report or {}).get("overall_confidence", None)

        user = f"""
Write a Markdown 'Research Report' for the last {days} days.

Input topics JSON:
{json.dumps(topics, ensure_ascii=False)}

Known issues JSON (may be empty):
{json.dumps(issues, ensure_ascii=False)}

Overall confidence (may be null): {overall_conf}

Requirements:
- Output Markdown only.
- Include: Executive Summary, Top Topics (with bullet innovations), Conflicts & Uncertainties, Notable Links.
- Each topic must include source links (title + URL) and 1-3 evidence snippets.
- Where sources conflict, explicitly attribute statements (e.g. 'Paper claims...', 'Talk suggests...') and downgrade certainty.
        """.strip()

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                last_err = None
                break
            except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
                last_err = e
                if attempt >= self.config.max_retries:
                    raise
                sleep_s = min(30.0, self.config.retry_backoff_s * (2**attempt))
                time.sleep(sleep_s)

        if last_err is not None:
            raise last_err
        return (resp.choices[0].message.content or "").strip()

    def answer_rag(
        self,
        *,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        RAG chat: answer using retrieved chunks (from ChromaDB).
        """
        system = (
            "You are an academic research assistant for LLM/ML/NLP. "
            "Answer precisely using the provided context. "
            "If the context is insufficient, say what is missing and propose a next search query."
        )
        user = f"""
Question:
{question}

Retrieved context chunks (JSON list). Each chunk has 'text' and 'metadata' with possible fields like title/url/source/published_at:
{json.dumps(retrieved_chunks, ensure_ascii=False)}

Requirements:
- Output Markdown only.
- Cite sources inline as [title](url) when available.
- Do not invent details not present in context.
        """.strip()

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                last_err = None
                break
            except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
                last_err = e
                if attempt >= self.config.max_retries:
                    raise
                sleep_s = min(30.0, self.config.retry_backoff_s * (2**attempt))
                time.sleep(sleep_s)

        if last_err is not None:
            raise last_err
        return (resp.choices[0].message.content or "").strip()


def default_agent() -> ResearchAgent:
    return ResearchAgent()

