import requests
from .config import KB_URL, KB_API_KEY

async def kb_search(query: str, k: int = 3):
    if not KB_URL:
        return []
    try:
        headers = {"Content-Type": "application/json"}
        if KB_API_KEY:
            headers["x-api-key"] = KB_API_KEY
        r = requests.post(f"{KB_URL.rstrip('/')}/search", json={"q": query, "k": k}, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            return data.get("results", []) or data.get("hits", []) or []
    except Exception:
        pass
    return []
