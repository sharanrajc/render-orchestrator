import re

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
    if any(w in t for w in ["yes","yeah","yep","correct","right","affirm"]): return True
    if any(w in t for w in ["no","nope","nah","negative"]): return False
    return None

def first_name(full: str | None) -> str:
    if not full:
        return ""
    return full.split()[0]
