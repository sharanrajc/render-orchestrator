from typing import Optional, Dict, Any
import json, requests, re

from .config import OPENAI_API_KEY, OPENAI_SLOT_MODEL, SLOT_CONFIDENCE_MIN

SYSTEM_PROMPT = """You are a call-center slot extractor for a pre-settlement funding intake.
From ONE short user utterance, extract ONLY the specific values (not full sentences) into the JSON schema.
For injury_details slot, extract full sentence into the JSON Schema.
Rules:
- Normalize:
  - phone: US E.164 like +13235550199 if possible; else null.
  - email: lowercase standard 'local@domain.tld' (no 'dot'/'at' words).
  - state: 2-letter US state code (CA, NY, etc) if present; else null.
  - incident_date: ISO yyyy-mm-dd if you can infer confidently; else null.
  - funding_amount: "$" + digits (no commas) if value is clear; else null.
  - funding_type: 'fresh' or 'topup' if caller says extend/top up.
  - has_attorney: true/false if clearly stated; else null.
- Values should be concise, e.g., full_name: "Joe A. Smith", law_firm: "Smith & Co".
- If unsure, leave the field null and confidence 0.0.
- Return ONLY the JSON matching the schema; no extra keys or commentary.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": ["string", "null"]},
        "phone": {"type": ["string", "null"]},
        "email": {"type": ["string", "null"]},
        "address": {"type": ["string", "null"]},
        "state": {"type": ["string", "null"]},
        "has_attorney": {"type": ["boolean", "null"]},
        "attorney_name": {"type": ["string", "null"]},
        "attorney_phone": {"type": ["string", "null"]},
        "law_firm": {"type": ["string", "null"]},
        "law_firm_address": {"type": ["string", "null"]},
        "injury_type": {"type": ["string", "null"]},
        "injury_details": {"type": ["string", "null"]},
        "incident_date": {"type": ["string", "null"]},
        "funding_type": {"type": ["string", "null"]},
        "funding_amount": {"type": ["string", "null"]},
        "_confidence": {
            "type": "object",
            "properties": {
                "full_name": {"type": "number"},
                "phone": {"type": "number"},
                "email": {"type": "number"},
                "address": {"type": "number"},
                "state": {"type": "number"},
                "has_attorney": {"type": "number"},
                "attorney_name": {"type": "number"},
                "attorney_phone": {"type": "number"},
                "law_firm": {"type": "number"},
                "law_firm_address": {"type": "number"},
                "injury_type": {"type": "number"},
                "injury_details": {"type": "number"},
                "incident_date": {"type": "number"},
                "funding_type": {"type": "number"},
                "funding_amount": {"type": "number"},
            },
            "required": []
        }
    },
    "required": ["_confidence"]
}

def _responses_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def _strict_upper_state(s: str | None) -> str | None:
    if not s: return None
    s = s.strip().upper()
    if re.fullmatch(r"[A-Z]{2}", s): return s
    return None

def extract_slots(utterance: str, wanted_fields: Optional[list] = None) -> Dict[str, Any]:
    if not OPENAI_API_KEY or not utterance:
        return {}
    hint = ""
    if wanted_fields:
        hint = "Only extract these fields if present: " + ", ".join(wanted_fields)

    payload = {
        "model": OPENAI_SLOT_MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{hint}\nUtterance: {utterance.strip()}"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "slots",
                "schema": SCHEMA,
                "strict": True
            }
        },
        "temperature": 0.1,
        "max_output_tokens": 400
    }

    try:
        data = _responses_api(payload)
        out_text = None
        if "output" in data and data["output"]:
            blocks = data["output"][0].get("content", [])
            for b in blocks:
                if b.get("type") == "output_text":
                    out_text = b.get("text")
                    break
        if not out_text and "response" in data:
            out_text = data["response"]
        if not out_text:
            return {}

        result = json.loads(out_text)
        conf = result.get("_confidence", {}) or {}
        cleaned = {}
        for k, v in result.items():
            if k == "_confidence":
                continue
            c = float(conf.get(k, 0.0) or 0.0)
            if v is None: 
                continue
            # normalize some fields
            if k == "state":
                v = _strict_upper_state(v)
                if not v:
                    continue
            if k == "email":
                v = v.strip().lower()
            if k == "funding_amount":
                v = re.sub(r"[^\d.]", "", v or "")
                if v:
                    v = f"${v}"
            if c >= SLOT_CONFIDENCE_MIN:
                cleaned[k] = v
        if conf:
            cleaned["_confidence"] = conf
        return cleaned
    except Exception:
        return {}
