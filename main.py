from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from agent import ResearchAgent, default_agent
from memory.vector_store import VectorMemory
from tools import EmailConfig, SourceItem, arxiv_search_with_fallback, bilibili_search, send_email_markdown, youtube_search


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:32]


def _state_dir() -> Path:
    d = Path(os.getenv("STATE_DIR", ".state"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _was_already_sent(report_md: str) -> bool:
    fp = _state_dir() / "last_report_sha256.txt"
    h = hashlib.sha256(report_md.encode("utf-8")).hexdigest()
    if fp.exists() and fp.read_text(encoding="utf-8").strip() == h:
        return True
    fp.write_text(h, encoding="utf-8")
    return False


def run_pipeline(*, days: int = 3) -> str:
    agent: ResearchAgent = default_agent()
    memory = VectorMemory()

    query = os.getenv("RESEARCH_QUERY", 'cat:cs.CL OR cat:cs.LG OR "large language model" OR "LLM" OR "NLP"')
    arxiv_max = int(os.getenv("ARXIV_MAX_RESULTS", "25"))
    yt_max = int(os.getenv("YOUTUBE_MAX_RESULTS", "10"))
    bili_max = int(os.getenv("BILIBILI_MAX_RESULTS", "10"))

    items: List[SourceItem] = []
    try:
        items += arxiv_search_with_fallback(query, days=days, max_results=arxiv_max)
    except Exception:
        pass
    try:
        items += youtube_search(query, days=days, max_results=yt_max)
    except Exception:
        pass
    try:
        items += bilibili_search(query, days=days, max_results=bili_max)
    except Exception:
        pass

    # Skill phase: summarize URLs (best-effort; still proceed if it fails)
    url_map = {it.url: it for it in items if it.url}
    summaries = agent.skill_summarize_urls(urls=list(url_map.keys()), length=os.getenv("SUMMARIZE_LENGTH", "short"))
    for u, s in summaries.items():
        if u in url_map:
            url_map[u].content = s

    extracted: List[Dict[str, Any]] = []
    for it in items:
        extracted.append(
            agent.extract_innovation_points(
                title=it.title,
                url=it.url,
                content=it.content,
                source=it.source,
                published_at=it.published_at,
            )
        )

    contra = agent.contradiction_check(items=extracted)
    compacted = agent.compact_topics(items=extracted, cleaned_claims=contra.get("cleaned_claims") or [])
    topics = compacted.get("topics") or []
    report_md = agent.synthesize_markdown_report(topics=topics, contradiction_report=contra, days=days)

    # Action phase: store to Chroma
    now = datetime.now(timezone.utc).isoformat()
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for it in items:
        ids.append(_stable_id(it.url, it.title))
        texts.append(it.content or it.title)
        metas.append(
            {
                "title": it.title,
                "url": it.url,
                "source": it.source,
                "published_at": it.published_at,
                "ingested_at": now,
            }
        )
    memory.add_texts(ids=ids, texts=texts, metadatas=metas)

    report_id = _stable_id("report", now)
    memory.add_texts(
        ids=[report_id],
        texts=[report_md],
        metadatas=[{"type": "research_report", "ingested_at": now, "days": days}],
    )

    # Email phase (optional)
    to_addrs = [a.strip() for a in (os.getenv("EMAIL_TO", "")).split(",") if a.strip()]
    if to_addrs:
        if os.getenv("EMAIL_DEDUP", "true").lower() == "true" and _was_already_sent(report_md):
            return report_md
        cfg = EmailConfig(
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "465")),
            username=os.getenv("SMTP_USER") or None,
            password=os.getenv("SMTP_PASS") or None,
            use_ssl=(os.getenv("SMTP_USE_SSL", "true").lower() == "true"),
            from_addr=os.getenv("EMAIL_FROM") or None,
        )
        subject = os.getenv("EMAIL_SUBJECT", f"AI Research Report (last {days} days)")
        send_email_markdown(cfg=cfg, to_addrs=to_addrs, subject=subject, markdown_body=report_md)

    return report_md


def chat_loop() -> None:
    agent: ResearchAgent = default_agent()
    memory = VectorMemory()
    print("Interactive RAG chat. Type 'exit' to quit.\n")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            return
        chunks = memory.query(query_text=q, n_results=int(os.getenv("RAG_TOP_K", "6")))
        answer = agent.answer_rag(question=q, retrieved_chunks=chunks)
        print("\n" + answer + "\n")


def schedule_loop(*, interval_days: int = 3, days: int = 3) -> None:
    """
    Simple scheduler loop. Best practice is to run this under a process manager (launchd/systemd/docker).
    """
    from time import sleep

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as e:
        raise RuntimeError("APScheduler not installed. Add it to requirements to use schedule mode.") from e

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(lambda: run_pipeline(days=days), trigger="interval", days=interval_days, id="pipeline")
    scheduler.start()

    print(f"Scheduler started. Running every {interval_days} days. Ctrl+C to stop.")
    try:
        while True:
            sleep(5)
    except KeyboardInterrupt:
        scheduler.shutdown(wait=False)


def main(argv: List[str]) -> int:
    load_dotenv()

    p = argparse.ArgumentParser(prog="ai_news_agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run 3-day discovery+synthesis pipeline once")
    run_p.add_argument("--days", type=int, default=3)

    chat_p = sub.add_parser("chat", help="Interactive RAG chat over stored memory")

    sched_p = sub.add_parser("schedule", help="Run pipeline on a repeating interval")
    sched_p.add_argument("--interval-days", type=int, default=int(os.getenv("SCHEDULE_INTERVAL_DAYS", "3")))
    sched_p.add_argument("--days", type=int, default=3)

    args = p.parse_args(argv)
    if args.cmd == "run":
        md = run_pipeline(days=args.days)
        print(md)
        return 0
    if args.cmd == "chat":
        chat_loop()
        return 0
    if args.cmd == "schedule":
        schedule_loop(interval_days=args.interval_days, days=args.days)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

