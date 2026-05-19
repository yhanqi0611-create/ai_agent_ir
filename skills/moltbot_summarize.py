from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class MoltbotConfig:
    """
    Wrapper for moltbot-summarize skill.

    Since SkillHub auth/endpoint details can vary, this wrapper is intentionally configurable.
    """

    endpoint: str = os.getenv("MOLTBOT_SUMMARIZE_ENDPOINT", "https://www.skillhub.club/skills/moltbot-moltbot-summarize")
    api_key: Optional[str] = None  # set via SKILLHUB_API_KEY
    timeout_s: float = 60.0


def summarize_urls(
    urls: List[str],
    *,
    cfg: Optional[MoltbotConfig] = None,
) -> Dict[str, Any]:
    """
    Calls the skill with a list of URLs and returns the raw JSON response.
    Expected behavior: return summaries per URL.
    """
    cfg = cfg or MoltbotConfig(api_key=os.getenv("SKILLHUB_API_KEY"))
    if not cfg.api_key:
        raise ValueError("Missing SKILLHUB_API_KEY in environment.")

    payload = {"urls": urls}
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    r = requests.post(cfg.endpoint, json=payload, headers=headers, timeout=cfg.timeout_s)
    r.raise_for_status()
    return r.json()


def normalize_summaries(skill_response: Any) -> Dict[str, str]:
    """
    Normalize common response shapes into: { url: summary_text }.

    Supported shapes (best-effort):
    - {"summaries": {"url": "text", ...}}
    - {"results": {"url": "text", ...}}
    - {"results": [{"url": "...", "summary": "..."}, ...]}
    - {"data": [{"url": "...", "content": "..."}, ...]}
    """
    if not isinstance(skill_response, dict):
        return {}

    for key in ("summaries", "results"):
        v = skill_response.get(key)
        if isinstance(v, dict):
            return {str(u): str(t) for u, t in v.items() if isinstance(u, str) and isinstance(t, str) and t.strip()}

    results = skill_response.get("results") or skill_response.get("data")
    if isinstance(results, list):
        out: Dict[str, str] = {}
        for row in results:
            if not isinstance(row, dict):
                continue
            url = row.get("url") or row.get("link")
            text = row.get("summary") or row.get("content") or row.get("text")
            if isinstance(url, str) and isinstance(text, str) and url.strip() and text.strip():
                out[url.strip()] = text.strip()
        return out

    return {}

