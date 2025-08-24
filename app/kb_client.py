# kb_client.py
import os, json
import httpx

KB_URL = os.environ.get("KB_URL", "")
KB_API_KEY = os.environ.get("KB_API_KEY", "")
KB_MAX_K = int(os.environ.get("KB_MAX_K", "20"))  # client-side cap to mirror KB

async def kb_search(query: str, k: int = 2):
    if not KB_URL or not query:
        return []
    # enforce cap: 1 ≤ k ≤ KB_MAX_K
    try:
        k = max(1, min(int(k), KB_MAX_K))
    except Exception:
        k = 5  # safe default

    headers = {"x-api-key": KB_API_KEY} if KB_API_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=2.0) as cli:
            r = await cli.post(KB_URL, json={"query": query, "k": k}, headers=headers)
            r.raise_for_status()
            return r.json().get("results", [])
    except Exception:
        return []


def format_kb_context(hits, max_chars: int = 900) -> str:
    if not hits: return ""
    parts, total = [], 0
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").strip()
        url   = (h.get("source_url") or "").strip()
        text  = (h.get("text") or "").strip()
        snippet = f"[{i}] {title} — {url}\n{text}\n\n"
        if total + len(snippet) > max_chars: break
        parts.append(snippet); total += len(snippet)
    return ("Knowledge Base (Oasis Financial)\nUse only for factual answers; do not give legal advice.\n\n"
            + "".join(parts) + "Note: Information is general, not legal advice.\n")
