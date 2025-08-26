# app/kb_client.py
import os, httpx
from typing import List, Dict
from .config import KB_URL, KB_API_KEY

async def kb_search(query: str, k: int = 3) -> List[Dict]:
    if not KB_URL:
        return []
    try:
        headers = {"Content-Type":"application/json"}
        if KB_API_KEY: headers["x-api-key"] = KB_API_KEY
        payload = {"q": query, "k": k}
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.post(f"{KB_URL.rstrip('/')}/search", headers=headers, json=payload)
            r.raise_for_status()
            js = r.json()
            return js.get("results", [])
    except Exception:
        return []
