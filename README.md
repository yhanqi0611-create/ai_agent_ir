## Academic Research AI Agent (DeepSeek + Arxiv/YouTube/Bilibili)

### Setup

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (use ChromaDB backend if it imports cleanly on your machine):

```bash
pip install -r requirements-chroma.txt
```

1. Configure env:

```bash
cp .env.example .env
```

Fill in `DEEPSEEK_API_KEY`, `TAVILY_API_KEY`, and optionally email + SkillHub keys.

If `export.arxiv.org` is unreachable, set `ARXIV_FALLBACK=TAVILY` in `.env` so missing Arxiv API results are backfilled via Tavily (`include_domains=arxiv.org`). Tavily rows are marked with `extra.via=tavily_fallback`.

### Run the 3-day pipeline once

```bash
python main.py run --days 3
```

- Fetches last N days from Arxiv + web search for YouTube/Bilibili
- Calls `moltbot-summarize` skill (best-effort)
- Uses DeepSeek to extract → contradiction-check → compact → synthesize Markdown report
- Stores items + report into ChromaDB (`memory/chroma/`)
- Stores items + report into local vector memory (`memory/store/`) by default; will auto-use ChromaDB if available
- Emails report if `EMAIL_TO` is set
- It may take 1-3 minutes

### Interactive RAG chat

```bash
python main.py chat
```

### Run on a 3-day schedule (long-running)

```bash
python main.py schedule --interval-days 3 --days 3
```

