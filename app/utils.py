# app/utils.py
import re
from datetime import datetime, timedelta
from dateutil import parser as dtparse

YES_SET = {"yes","yeah","yep","correct","right","affirmative","sure","ok","okay"}
NO_SET = {"no","nope","incorrect","wrong","nah","negative","not correct","cancel"}

def clean_text(t: str) -> str:
    return (t or "").strip()

def yes_no(t: str):
    s = (t or "").strip().lower()
    if not s: return None
    if any(w in s for w in YES_SET): return True
    if any(w in s for w in NO_SET): return False
    return None

# spoken digits → numeric
_WORD_DIGITS = {"zero":"0","oh":"0","o":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
                "six":"6","seven":"7","eight":"8","nine":"9"}

def spoken_to_digits(text: str) -> str:
    s = " " + (text or "").lower() + " "
    for w,d in _WORD_DIGITS.items():
        s = re.sub(rf"\b{w}\b", d, s)
    s = s.replace("dash"," ").replace("-", " ").replace(".", " ").replace(",", " ")
    s = re.sub(r"\s+"," ", s)
    return s.strip()

def extract_phone(text: str):
    if not text: return None
    t = spoken_to_digits(text)
    if "same" in t and "number" in t:
        return "__SAME__"
    digits = re.sub(r"\D", "", t)
    if len(digits) >= 10: return digits[-10:]
    return None

def e164(us10: str | None) -> str | None:
    if not us10 or len(us10) != 10: return None
    return "+1" + us10

def normalize_email_spoken(text: str) -> str:
    s = (text or "")
    s = re.sub(r"\s+at\s+","@", s, flags=re.I)
    s = re.sub(r"\s+dot\s+"," . ", s, flags=re.I)  # protect dots
    s = s.replace(" underscore ","_").replace(" hyphen ","-")
    s = re.sub(r"\s+","", s).replace(" . ", ".")
    return s

_EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)

def extract_email(text: str):
    if not text: return None
    s = normalize_email_spoken(text)
    m = _EMAIL_RE.search(s)
    return m.group(0).lower() if m else None

def spell_for_email(email: str) -> str:
    return email.replace("@", " at ").replace(".", " dot ")

# names
_NAME_PREFIXES = [r"\bmy (full legal )?name is\b", r"\bthis is\b", r"\bi am\b", r"\bit is\b", r"\bthe name is\b", r"\byou can call me\b"]

def extract_name(text: str):
    if not text: return None
    t = " " + text.strip() + " "
    for pref in _NAME_PREFIXES:
        m = re.search(pref + r"\s+([A-Za-z][A-Za-z .'\-]{0,80})", t, flags=re.I)
        if m:
            cand = m.group(1).strip(" .,!?:;\"'()[]")
            toks = [w for w in cand.split() if re.search(r"[A-Za-z]", w)]
            if 1 <= len(toks) <= 4 and "@" not in cand and not any(ch.isdigit() for ch in cand):
                def _tc(w): return w if re.fullmatch(r"[A-Za-z]\.", w) else w.capitalize()
                return " ".join(_tc(w) for w in toks)
    cand = text.strip().strip(" .,!?:;\"'()[]")
    toks = [w for w in cand.split() if re.search(r"[A-Za-z]", w)]
    if 1 <= len(toks) <= 4 and "@" not in cand and not any(ch.isdigit() for ch in cand):
        def _tc(w): return w if re.fullmatch(r"[A-Za-z]\.", w) else w.capitalize()
        return " ".join(_tc(w) for w in toks)
    return None

def extract_state(text: str) -> str | None:
    m = re.search(r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b", text or "", re.I)
    return m.group(1).upper() if m else None

def extract_incident_date(text: str) -> str | None:
    if not text: return None
    s = text.strip().lower()
    try:
        if s == "yesterday": return (datetime.utcnow() - timedelta(days=1)).date().isoformat()
        if s == "today": return datetime.utcnow().date().isoformat()
        dt = dtparse.parse(text, fuzzy=True, dayfirst=False, yearfirst=False)
        return dt.date().isoformat()
    except Exception:
        return None

def extract_funding_type(text: str) -> str | None:
    s = (text or "").lower()
    if any(k in s for k in ["fresh","new","first time","first-time"]): return "fresh"
    if any(k in s for k in ["extend","extension","top up","top-up","topup","increase","more"]): return "extend"
    return None

def extract_amount(text: str) -> str | None:
    s = (text or "").lower()
    k = re.search(r"(\d[\d,\.]*)\s*k\b", s)
    if k:
        try:
            val = float(k.group(1).replace(",",""))
            return f"${int(val*1000):,}"
        except Exception:
            pass
    m = re.search(r"\$?\s*([\d,]{1,9})(\.\d+)?", s)
    if m:
        num = int(m.group(1).replace(",",""))
        return f"${num:,}"
    w = re.search(r"(\bone|two|three|four|five|six|seven|eight|nine|ten)\s+(thousand|grand)\b", s)
    if w:
        table = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        return f"${table[w.group(1)]*1000:,}"
    return None
