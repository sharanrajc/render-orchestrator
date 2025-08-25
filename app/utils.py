import re
from dateutil import parser as dtparse

def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = s.strip(" .,!?:;\"'()[]")
    return s

def yes_no(text: str):
    t = (text or "").lower()
    if any(w in t for w in ["yes","yeah","yep","correct","right","affirm","sure"]): return True
    if any(w in t for w in ["no","nope","nah","negative","don’t","do not","incorrect"]): return False
    return None

def spell_for_name(name: str) -> str:
    name = clean_text(name)
    words = [w for w in name.split() if w]
    spelled_words = [" ".join(list(w.upper())) for w in words]
    return "   ".join(spelled_words)

def spell_for_email(email: str) -> str:
    m = {"@": "at", ".": "dot", "-": "dash", "_": "underscore", "+": "plus"}
    out = []
    for ch in (email or "").strip():
        out.append(m.get(ch, ch))
    spoken = " ".join(out)
    return re.sub(r"\s+", " ", spoken).lower()

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
