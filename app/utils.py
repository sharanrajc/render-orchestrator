import re
from dateutil import parser as dtparse

def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = s.strip(" .,!?:;\"'()[]")
    return s

def normalize_phone(s: str) -> str:
    digits = re.sub(r"\D", "", s or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"+1{digits}"
    return f"+{digits}" if digits else ""

def extract_amount_usd(text: str):
    m = re.search(r"\$?\s*([0-9][0-9,]*(?:\.\d{1,2})?)", text or "")
    if not m:
        return None
    val = m.group(1).replace(",", "")
    return f"${val}"

def yes_no(text: str):
    t = (text or "").lower()
    if any(w in t for w in ["yes","yeah","yep","correct","right","affirm","sure"]): return True
    if any(w in t for w in ["no","nope","nah","negative","don’t","do not","incorrect"]): return False
    return None

def first_name(full: str | None) -> str:
    if not full: return ""
    return full.split()[0]

def parse_incident_date(text: str) -> str | None:
    if not text: return None
    try:
        d = dtparse.parse(text, fuzzy=True, dayfirst=False)
        return d.date().isoformat()
    except Exception:
        m = re.search(r"(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-]([12]\d{3})", text or "")
        if m:
            mm, dd, yy = m.groups()
            return f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}"
    return None

def infer_state_from_address(addr: str) -> str | None:
    if not addr: return None
    m = re.search(r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b", (addr or "").upper())
    return m.group(1) if m else None

def map_correction_field(text: str) -> str | None:
    t = (text or "").lower()
    if any(w in t for w in ["name", "first name", "last name", "full name"]): return "name"
    if any(w in t for w in ["phone", "number", "contact"]): return "phone"
    if "email" in t: return "email"
    if any(w in t for w in ["address", "state", "zip", "city"]): return "address"
    if any(w in t for w in ["attorney", "law firm", "firm"]): return "attorney"
    if any(w in t for w in ["case", "case type", "auto", "slip", "fall", "dog"]): return "case"
    if any(w in t for w in ["incident", "accident date", "date of incident", "doi"]): return "incident_date"
    if any(w in t for w in ["funding type", "fresh", "top up", "top-up", "extend"]): return "funding_type"
    if any(w in t for w in ["amount", "how much", "dollars", "usd"]): return "funding_amount"
    return None

def next_missing_stage(state) -> str | None:
    if not state.full_name: return "ASK_NAME"
    if not (state.best_phone or state.phone): return "ASK_PHONE"
    if not state.email: return "ASK_EMAIL"
    if not state.address or state.address == "NOT PROVIDED": return "ASK_ADDRESS"
    if state.has_attorney is None: return "ATTORNEY_YN"
    if state.has_attorney and not (state.attorney_name or state.law_firm or state.attorney_phone): return "ATTORNEY_INFO"
    if not state.injury_type: return "CASE_TYPE"
    if not state.injury_details: return "INJURY_DETAILS"
    if not state.incident_date: return "INCIDENT_DATE"
    if not state.funding_type: return "FUNDING_TYPE"
    if not state.funding_amount: return "FUNDING_AMOUNT"
    return None

# --- Spoken → email normalization ---
def normalize_email(text: str) -> str | None:
    if not text:
        return None
    t = " " + text.lower().strip() + " "
    t = re.sub(r"\s+at\s+", "@", t)
    t = re.sub(r"\s+(dot|period|point)\s+", ".", t)
    t = re.sub(r"\s+underscore\s+", "_", t)
    t = re.sub(r"\s+dash\s+", "-", t)
    t = re.sub(r"\s+plus\s+", "+", t)
    t = re.sub(r"\s+", "", t)
    t = t.strip(" .,!?:;\"'()[]")
    if re.match(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$", t):
        return t
    return None

# --- Spell-out helpers (for TTS readback) ---
def spell_for_name(name: str) -> str:
    """Return something like: J O H N   D O E (3 spaces between words)."""
    name = clean_text(name)
    words = [w for w in name.split() if w]
    spelled_words = [" ".join(list(w.upper())) for w in words]
    return "   ".join(spelled_words)

def spell_for_email(email: str) -> str:
    """Return: j o h n dot d o e at g m a i l dot com"""
    m = {
        "@": "at",
        ".": "dot",
        "-": "dash",
        "_": "underscore",
        "+": "plus",
    }
    out = []
    for ch in email.strip():
        out.append(m.get(ch, ch))
    # now space-separate characters and collapse spaces
    spoken = " ".join(out)
    spoken = re.sub(r"\s+", " ", spoken)
    return spoken.lower()
