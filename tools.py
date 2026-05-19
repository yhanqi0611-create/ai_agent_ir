from __future__ import annotations

import os
import re
import smtplib
import ssl
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict, Iterable, List, Optional

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class SourceItem:
    title: str
    url: str
    source: str  # "arxiv" | "youtube" | "bilibili" | "other"
    published_at: Optional[str] = None
    content: str = ""  # abstract, transcript snippet, or fetched summary
    extra: Optional[Dict[str, Any]] = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def arxiv_search(
    query: str,
    *,
    days: int = 3,
    max_results: int = 30,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> List[SourceItem]:
    """
    Uses Arxiv Atom API.
    Query syntax: https://arxiv.org/help/api/user-manual
    """
    since = _utcnow() - timedelta(days=days)
    # Prefer HTTPS; allow override for restricted networks.
    url = os.getenv("ARXIV_API_URL", "https://export.arxiv.org/api/query")
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    try:
        session = requests.Session()
        retry = Retry(
            total=int(os.getenv("ARXIV_HTTP_RETRIES", "3")),
            connect=int(os.getenv("ARXIV_HTTP_RETRIES", "3")),
            read=int(os.getenv("ARXIV_HTTP_RETRIES", "3")),
            backoff_factor=float(os.getenv("ARXIV_HTTP_BACKOFF", "0.7")),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        r = session.get(url, params=params, timeout=float(os.getenv("ARXIV_HTTP_TIMEOUT_S", "30")))
        r.raise_for_status()
    except Exception:
        # Network/DNS errors should not break the whole pipeline.
        return []
    feed = feedparser.parse(r.text)

    items: List[SourceItem] = []
    for entry in feed.entries:
        published = getattr(entry, "published", None)
        published_dt: Optional[datetime] = None
        if published:
            try:
                published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except Exception:
                published_dt = None

        if published_dt and published_dt < since:
            continue

        pdf_url = ""
        for link in getattr(entry, "links", []) or []:
            if getattr(link, "type", "") == "application/pdf":
                pdf_url = getattr(link, "href", "") or ""
                break

        primary_url = getattr(entry, "link", "") or pdf_url
        abstract = (getattr(entry, "summary", "") or "").strip()
        title = " ".join((getattr(entry, "title", "") or "").split())

        authors = [a.name for a in getattr(entry, "authors", []) or [] if getattr(a, "name", None)]
        categories = [t.term for t in getattr(entry, "tags", []) or [] if getattr(t, "term", None)]

        items.append(
            SourceItem(
                title=title,
                url=primary_url,
                source="arxiv",
                published_at=published,
                content=abstract,
                extra={
                    "pdf_url": pdf_url,
                    "authors": authors,
                    "categories": categories,
                    "arxiv_id": getattr(entry, "id", None),
                },
            )
        )

    return items


def _tavily_friendly_arxiv_query(arxiv_style_query: str) -> str:
    """
    Turn Arxiv API search_query into keywords suitable for Tavily + domain filter.
    """
    q = arxiv_style_query.strip()
    q = re.sub(r"\bcat:\s*[\w.]+\b", "", q, flags=re.IGNORECASE)
    q = q.replace(" OR ", " ").replace(" or ", " ")
    q = re.sub(r"\s+", " ", q).strip().strip('"')
    return q or "large language model NLP"


def arxiv_search_tavily_fallback(
    query: str,
    *,
    days: int = 3,
    max_results: int = 30,
) -> List[SourceItem]:
    """
    When export.arxiv.org is unreachable or returns no items, use Tavily on arxiv.org.
    Items use extra['via'] == 'tavily_fallback' for traceability.
    """
    tq = os.getenv("ARXIV_TAVILY_QUERY", "").strip() or _tavily_friendly_arxiv_query(query)
    raw_domains = os.getenv("ARXIV_TAVILY_DOMAINS", "arxiv.org")
    domains = [d.strip() for d in raw_domains.split(",") if d.strip()]
    if not domains:
        domains = ["arxiv.org"]

    try:
        results = tavily_search(
            tq,
            days=days,
            max_results=max_results,
            include_domains=domains,
        )
    except Exception:
        return []

    items: List[SourceItem] = []
    seen: set = set()
    for r in results:
        url = (r.get("url") or "").strip()
        if not url or "arxiv.org" not in url.lower():
            continue
        if url in seen:
            continue
        seen.add(url)
        items.append(
            SourceItem(
                title=r.get("title") or "arXiv (Tavily)",
                url=url,
                source="arxiv",
                published_at=r.get("published_date") or None,
                content=(r.get("content") or "").strip(),
                extra={"score": r.get("score"), "via": "tavily_fallback", "raw": r},
            )
        )
        if len(items) >= max_results:
            break
    return items


def arxiv_search_with_fallback(
    query: str,
    *,
    days: int = 3,
    max_results: int = 30,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> List[SourceItem]:
    """
    Try official Arxiv Atom API first. If it returns no items and ARXIV_FALLBACK=TAVILY,
    pull arxiv.org pages via Tavily.
    """
    primary = arxiv_search(
        query,
        days=days,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if primary:
        return primary
    if os.getenv("ARXIV_FALLBACK", "").strip().upper() != "TAVILY":
        return []
    return arxiv_search_tavily_fallback(query, days=days, max_results=max_results)


def tavily_search(
    query: str,
    *,
    days: int = 3,
    max_results: int = 10,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Lightweight Tavily API wrapper for web search.
    Use include_domains to target YouTube/Bilibili.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in environment.")

    endpoint = "https://api.tavily.com/search"
    payload: Dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "days": days,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    r = requests.post(endpoint, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []


def youtube_search(
    query: str,
    *,
    days: int = 3,
    max_results: int = 10,
) -> List[SourceItem]:
    results = tavily_search(query, days=days, max_results=max_results, include_domains=["youtube.com", "youtu.be"])
    items: List[SourceItem] = []
    for r in results:
        items.append(
            SourceItem(
                title=r.get("title") or "YouTube result",
                url=r.get("url") or "",
                source="youtube",
                published_at=r.get("published_date") or None,
                content=(r.get("content") or "").strip(),
                extra={"score": r.get("score"), "raw": r},
            )
        )
    return items


def bilibili_search(
    query: str,
    *,
    days: int = 3,
    max_results: int = 10,
) -> List[SourceItem]:
    results = tavily_search(query, days=days, max_results=max_results, include_domains=["bilibili.com"])
    items: List[SourceItem] = []
    for r in results:
        items.append(
            SourceItem(
                title=r.get("title") or "Bilibili result",
                url=r.get("url") or "",
                source="bilibili",
                published_at=r.get("published_date") or None,
                content=(r.get("content") or "").strip(),
                extra={"score": r.get("score"), "raw": r},
            )
        )
    return items


@dataclass(frozen=True)
class EmailConfig:
    smtp_host: str
    smtp_port: int = 465
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = True
    from_addr: Optional[str] = None


def send_email_markdown(
    *,
    cfg: EmailConfig,
    to_addrs: List[str],
    subject: str,
    markdown_body: str,
) -> None:
    """
    Sends a Markdown email as text/plain + text/markdown. Some clients ignore text/markdown;
    text/plain ensures basic readability.
    """
    if not to_addrs:
        raise ValueError("to_addrs must not be empty")
    if not cfg.from_addr:
        raise ValueError("EmailConfig.from_addr is required")

    msg = EmailMessage()
    msg["From"] = cfg.from_addr
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject

    plain = textwrap.dedent(markdown_body).strip()
    msg.set_content(plain)
    msg.add_alternative(markdown_body, subtype="markdown")

    if cfg.use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=context) as smtp:
            if cfg.username and cfg.password:
                smtp.login(cfg.username, cfg.password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            if cfg.username and cfg.password:
                smtp.login(cfg.username, cfg.password)
            smtp.send_message(msg)

