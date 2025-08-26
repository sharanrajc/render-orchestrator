# app/llm_slots.py
import os, json, httpx, re
from .config import OPENAI_API_KEY, OPENAI_MODEL

TIMEOUT = 8.0

def extract_slots(text: str, wanted_fields=None) -> dict:
    """
    Best-effort LLM extraction. Returns {} on any error.
    """
    if not text or not OPENAI_API_KEY:
        return {}

    wanted_fields = wanted_fields or []
    sys = (
        "You extract structured fields from a caller utterance. "
        "Return a strict JSON object with keys only from: "
        "full_name, phone, email, address, state, has_attorney, attorney_name, attorney_phone, "
        "law_firm, law_firm_address, injury_type, injury_details, incident_date, funding_type, funding_amount. "
        "Use null for unknown. has_attorney must be true/false/null."
    )
    user = f"Utterance: {text}\nOnly include keys that are present or highly likely. Phone must be 10 digits if provided."

    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role":"system","content":sys},{"role":"user","content":user}],
            "response_format": {"type":"json_object"},
            "temperature": 0.1,
        }
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            js = r.json()
            content = js["choices"][0]["message"]["content"]
            data = json.loads(content)
            # normalize booleans
            if "has_attorney" in data and not isinstance(data["has_attorney"], bool):
                s = str(data["has_attorney"]).lower()
                data["has_attorney"] = True if "true" in s or "yes" in s else False if "false" in s or "no" in s else None
            return data
    except Exception:
        return {}
