import time
from typing import Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import ORCH_API_KEY
from .db import init_db, SessionLocal, upsert_application_from_state, get_latest_by_caller, Application
from .models import OrchestrateRequest, OrchestrateResponse, SessionState
from .prompts import PROMPTS
from .utils import (
    clean_text, normalize_phone, extract_amount_usd, yes_no,
    parse_incident_date, infer_state_from_address,
    map_correction_field, next_missing_stage, normalize_email,
    spell_for_name, spell_for_email
)
from .tools import verify_address, verify_attorney
from .kb_client import kb_search

app = FastAPI(title="Orchestrator", version="3.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
init_db()

SESSIONS: Dict[str, SessionState] = {}
TRANSCRIPTS: Dict[str, List[dict]] = {}
MAX_RETRIES_PER_STAGE = 2
TRANSCRIPT_MAX_TURNS = 300

DEFAULT_LISTEN = 7
LONG_LISTEN = 15

def _now_ms() -> int: return int(time.time() * 1000)

def _append_turn(session_id: str, role: str, text: str, stage: str, extra: Optional[dict] = None):
    rec = {"ts": _now_ms(), "role": role, "text": text or "", "stage": stage}
    if extra: rec["meta"] = extra
    lst = TRANSCRIPTS.setdefault(session_id, [])
    lst.append(rec)
    if len(lst) > TRANSCRIPT_MAX_TURNS: TRANSCRIPTS[session_id] = lst[-TRANSCRIPT_MAX_TURNS:]

def next_prompt_text(state: SessionState, text: str) -> str:
    if state.last_prompt and text.strip() == (state.last_prompt or "").strip():
        return PROMPTS["REPROMPT_SHORT"] + text
    return text

def bump_retry(state: SessionState, key: str) -> bool:
    n = state.retries.get(key, 0) + 1
    state.retries[key] = n
    return n > MAX_RETRIES_PER_STAGE

def respond(state: SessionState, text: str, *, completed=False, handoff=False, citations: Optional[List[int]]=None, confidence: float=0.7):
    citations = citations or []
    _append_turn(state.session_id, "ai", text, state.stage, {"completed": completed, "handoff": handoff, "citations": citations})
    SESSIONS[state.session_id] = state
    state.last_prompt = text
    with SessionLocal() as db:
        upsert_application_from_state(db, state)
        db.commit()
    return OrchestrateResponse(
        updates=state.to_updates(),
        next_prompt=text[:180],
        completed=completed,
        handoff=handoff,
        citations=citations,
        confidence=confidence,
        listen_timeout_sec=state.listen_timeout_sec or DEFAULT_LISTEN
    )

@app.get("/health")
def health():
    with SessionLocal() as db:
        total = db.query(Application).count()
    return {"status":"ok","sessions":len(SESSIONS),"applications":total}

@app.get("/applications")
def list_apps(phone: Optional[str] = Query(None)):
    with SessionLocal() as db:
        if phone:
            rows = db.query(Application).filter(Application.best_phone == phone).order_by(Application.updated_at.desc()).all()
        else:
            rows = db.query(Application).order_by(Application.updated_at.desc()).limit(50).all()
        return {"count": len(rows), "applications": [
            {"id": r.id, "session_id": r.session_id, "status": r.status, "caller_number": r.caller_number,
             "full_name": r.full_name, "best_phone": r.best_phone, "email": r.email,
             "state": r.state, "injury_type": r.injury_type, "injury_details": r.injury_details,
             "incident_date": r.incident_date, "funding_type": r.funding_type, "funding_amount": r.funding_amount,
             "updated_at": r.updated_at.isoformat() if r.updated_at else None}
            for r in rows
        ]}

@app.get("/application/{app_id}")
def get_app(app_id: int):
    with SessionLocal() as db:
        row = db.get(Application, app_id)
        if not row:
            raise HTTPException(404, "not found")
        return {"application": {c.name: getattr(row, c.name) for c in row.__table__.columns}}

@app.get("/transcript/{session_id}")
def get_transcript(session_id: str, redact: bool = True):
    turns = TRANSCRIPTS.get(session_id, [])
    if not redact: return {"session_id": session_id, "turns": turns}
    import re
    def _redact(text: str) -> str:
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[redacted-email]", text)
        text = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted-phone]", text)
        return text
    red = []
    for t in turns:
        t2 = dict(t); t2["text"] = _redact(t2.get("text","")); red.append(t2)
    return {"session_id": session_id, "turns": red}

@app.get("/transcripts")
def list_transcripts():
    return {"sessions": [{"session_id": sid, "turns": len(ts)} for sid, ts in TRANSCRIPTS.items()]}

@app.post("/reset")
def reset_session(req: OrchestrateRequest, x_api_key: Optional[str] = Header(None)):
    if ORCH_API_KEY and x_api_key != ORCH_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    SESSIONS.pop(req.session_id, None)
    TRANSCRIPTS.pop(req.session_id, None)
    return {"ok": True}

# ===== main orchestrate =====
@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest, x_api_key: Optional[str] = Header(None)):
    if ORCH_API_KEY and x_api_key != ORCH_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    state = SESSIONS.get(req.session_id) or SessionState(session_id=req.session_id)
    state.listen_timeout_sec = DEFAULT_LISTEN

    if req.caller_number and not state.caller_number:
        state.caller_number = normalize_phone(req.caller_number)
    utter = clean_text(req.last_user_utterance or "")
    stage = state.stage

    if utter:
        _append_turn(req.session_id, "user", utter, stage)

    # ENTRY
    if stage == "ENTRY":
        greet = PROMPTS["INTRO"]
        if state.caller_number:
            with SessionLocal() as db:
                latest = get_latest_by_caller(db, state.caller_number)
            if latest:
                state.stage = "RESUME_CHOICE"
                return respond(state, greet + " " + PROMPTS["EXISTING"])
        state.stage = "GREETING"
        return respond(state, greet)

    if stage == "GREETING":
        state.stage = "ASK_NAME"
        return respond(state, PROMPTS["ASK_NAME"])

    # Resume choice
    if stage == "RESUME_CHOICE":
        t = (utter or "").lower()
        if "continue" in t or "pending" in t:
            nxt = next_missing_stage(state)
            if nxt:
                state.stage = nxt
                if nxt in ("ASK_ADDRESS","INJURY_DETAILS","ATTORNEY_INFO"): state.listen_timeout_sec = LONG_LISTEN
                prompt_map = {
                    "ASK_NAME": PROMPTS["ASK_NAME"],
                    "ASK_PHONE": PROMPTS["ASK_PHONE"],
                    "ASK_EMAIL": PROMPTS["ASK_EMAIL"],
                    "ASK_ADDRESS": PROMPTS["ASK_ADDRESS"],
                    "ATTORNEY_YN": PROMPTS["ASK_ATTORNEY_YN"],
                    "ATTORNEY_INFO": PROMPTS["ASK_ATTORNEY_INFO"],
                    "CASE_TYPE": PROMPTS["ASK_CASE_TYPE"],
                    "INJURY_DETAILS": PROMPTS["ASK_INJURY_DETAILS"],
                    "INCIDENT_DATE": PROMPTS["ASK_INCIDENT_DATE"],
                    "FUNDING_TYPE": PROMPTS["ASK_FUNDING_TYPE"],
                    "FUNDING_AMOUNT": PROMPTS["ASK_FUNDING_AMOUNT"],
                }
                return respond(state, "Let’s pick up where we left off. " + prompt_map.get(nxt, ""))
            state.stage = "SUMMARY"; state.summary_read=False; state.awaiting_confirmation=False
            return respond(state, "Looks like we have everything. I’ll read back your details.")
        if "modify" in t or "update" in t or "edit" in t:
            state.stage = "SUMMARY"; state.summary_read=False; state.awaiting_confirmation=False
            return respond(state, "Okay, I’ll read back your details and we can make changes.")
        if "new" in t or "start" in t:
            state = SessionState(session_id=state.session_id, caller_number=state.caller_number)
            SESSIONS[state.session_id] = state
            state.stage = "ASK_NAME"
            return respond(state, "Starting a new application. What’s your full legal name?")
        if bump_retry(state, "RESUME_CHOICE"):
            state.stage = "ASK_NAME"
            return respond(state, "Let’s move ahead. What’s your full legal name?")
        return respond(state, PROMPTS["EXISTING"])

    # ===== NAME with spell-and-confirm =====
    if stage == "ASK_NAME":
        if utter:
            parts = utter.split()
            if len(parts) == 1:
                state.full_name = parts[0]
                state.stage = "ASK_LAST_NAME"
                return respond(state, PROMPTS["ASK_LAST_NAME"].format(first=parts[0]))
            state.full_name = utter
            state.stage = "CONFIRM_NAME"
            spelled = spell_for_name(state.full_name)
            return respond(state, PROMPTS["CONFIRM_NAME"].format(name=state.full_name, spelled=spelled))
        if bump_retry(state, "ASK_NAME"):
            state.stage = "ASK_PHONE" if not state.caller_number else "PREFERRED_PHONE"
            prompt = PROMPTS["ASK_PREFERRED_PHONE_CONFIRM"].format(caller=state.caller_number) if state.caller_number else PROMPTS["ASK_PHONE"]
            return respond(state, prompt)
        return respond(state, PROMPTS["ASK_NAME"])

    if stage == "ASK_LAST_NAME":
        if utter:
            state.full_name = f"{state.full_name} {utter}".strip()
            state.stage = "CONFIRM_NAME"
            spelled = spell_for_name(state.full_name)
            return respond(state, PROMPTS["CONFIRM_NAME"].format(name=state.full_name, spelled=spelled))
        if bump_retry(state, "ASK_LAST_NAME"):
            state.stage = "CONFIRM_NAME"
            spelled = spell_for_name(state.full_name or "")
            return respond(state, PROMPTS["CONFIRM_NAME"].format(name=(state.full_name or ""), spelled=spelled))
        return respond(state, PROMPTS["ASK_LAST_NAME"].format(first=state.full_name or ""))

    if stage == "CONFIRM_NAME":
        yn = yes_no(utter)
        if yn is True:
            state.stage = "PREFERRED_PHONE" if state.caller_number else "ASK_PHONE"
            prompt = PROMPTS["ASK_PREFERRED_PHONE_CONFIRM"].format(caller=state.caller_number) if state.caller_number else PROMPTS["ASK_PHONE"]
            return respond(state, prompt)
        if yn is False or ("spell" in (utter or "").lower()):
            state.stage = "SPELL_NAME"
            return respond(state, PROMPTS["NAME_SPELL_PROMPT"])
        # if unclear, repeat spelling
        spelled = spell_for_name(state.full_name or "")
        return respond(state, PROMPTS["CONFIRM_NAME"].format(name=(state.full_name or ""), spelled=spelled))

    if stage == "SPELL_NAME":
        if utter:
            state.full_name = clean_text(utter)
            state.stage = "CONFIRM_NAME"
            spelled = spell_for_name(state.full_name)
            return respond(state, PROMPTS["CONFIRM_NAME"].format(name=state.full_name, spelled=spelled))
        if bump_retry(state, "SPELL_NAME"):
            state.stage = "ASK_NAME"
            return respond(state, PROMPTS["ASK_NAME"])
        return respond(state, PROMPTS["NAME_SPELL_PROMPT"])

    # ===== PHONE (unchanged, with confirm) =====
    if stage == "PREFERRED_PHONE":
        yn = yes_no(utter)
        if yn is True and state.caller_number:
            state.phone = state.caller_number
            state.best_phone = state.caller_number
            state.stage = "CONFIRM_PHONE"
            return respond(state, PROMPTS["CONFIRM_PHONE"].format(phone=state.best_phone))
        if yn is False or not state.caller_number:
            state.stage = "ASK_PHONE"
            return respond(state, PROMPTS["ASK_PHONE"])
        if bump_retry(state, "PREFERRED_PHONE"):
            state.stage = "ASK_PHONE"
            return respond(state, PROMPTS["ASK_PHONE"])
        return respond(state, PROMPTS["ASK_PREFERRED_PHONE_CONFIRM"].format(caller=state.caller_number or "this number"))

    if stage == "ASK_PHONE":
        ph = normalize_phone(utter)
        if ph:
            state.phone = ph
            state.best_phone = ph
            state.stage = "CONFIRM_PHONE"
            return respond(state, PROMPTS["CONFIRM_PHONE"].format(phone=state.best_phone))
        if bump_retry(state, "ASK_PHONE"):
            state.stage = "ASK_EMAIL"
            return respond(state, PROMPTS["ASK_EMAIL"])
        return respond(state, "Please say your 10-digit phone number.")

    if stage == "CONFIRM_PHONE":
        yn = yes_no(utter)
        if yn is True or yn is None:
            state.stage = "ASK_EMAIL"
            return respond(state, PROMPTS["ASK_EMAIL"])
        state.stage = "ASK_PHONE"
        return respond(state, "Okay, please say your preferred 10-digit phone number again.")

    # ===== EMAIL with spell-and-confirm =====
    if stage == "ASK_EMAIL":
        if utter:
            em = normalize_email(utter) or normalize_email(clean_text(utter))
            if not em and not bump_retry(state, "ASK_EMAIL"):
                return respond(state, "Could you please repeat your email? For example, john dot doe at gmail dot com.")
            state.email = em or clean_text(utter)
            state.stage = "CONFIRM_EMAIL_SPELL"
            spelled = spell_for_email(state.email)
            return respond(state, PROMPTS["CONFIRM_EMAIL_SPELL"].format(email=state.email, spelled=spelled))
        if bump_retry(state, "ASK_EMAIL"):
            state.stage = "ASK_ADDRESS"
            state.listen_timeout_sec = LONG_LISTEN
            return respond(state, PROMPTS["ASK_ADDRESS"])
        return respond(state, PROMPTS["ASK_EMAIL"])

    if stage == "CONFIRM_EMAIL_SPELL":
        yn = yes_no(utter)
        if yn is True:
            state.stage = "ASK_ADDRESS"
            state.listen_timeout_sec = LONG_LISTEN
            return respond(state, PROMPTS["ASK_ADDRESS"])
        if yn is False or ("spell" in (utter or "").lower()):
            state.stage = "SPELL_EMAIL"
            return respond(state, PROMPTS["EMAIL_SPELL_PROMPT"])
        # if unclear, repeat spelling
        spelled = spell_for_email(state.email or "")
        return respond(state, PROMPTS["CONFIRM_EMAIL_SPELL"].format(email=(state.email or ""), spelled=spelled))

    if stage == "SPELL_EMAIL":
        if utter:
            em = normalize_email(utter) or normalize_email(clean_text(utter))
            if em:
                state.email = em
            else:
                state.email = clean_text(utter)
            state.stage = "CONFIRM_EMAIL_SPELL"
            spelled = spell_for_email(state.email)
            return respond(state, PROMPTS["CONFIRM_EMAIL_SPELL"].format(email=state.email, spelled=spelled))
        if bump_retry(state, "SPELL_EMAIL"):
            state.stage = "ASK_ADDRESS"
            state.listen_timeout_sec = LONG_LISTEN
            return respond(state, PROMPTS["ASK_ADDRESS"])
        return respond(state, PROMPTS["EMAIL_SPELL_PROMPT"])

    # ===== ADDRESS (confirm) =====
    if stage == "ASK_ADDRESS":
        state.listen_timeout_sec = LONG_LISTEN
        t = (utter or "").lower()
        if "skip" in t or "prefer not" in t or "no address" in t:
            state.address = "NOT PROVIDED"; state.address_norm=None; state.address_verified=False; state.state=None
            state.stage = "ATTORNEY_YN"
            return respond(state, PROMPTS["ADDRESS_SKIPPED"] + " " + PROMPTS["ASK_ATTORNEY_YN"])
        if utter:
            state.address = clean_text(utter)
            verified, normalized = await verify_address(state.address)
            state.address_verified = bool(verified)
            state.address_norm = normalized or state.address
            state.state = infer_state_from_address(state.address_norm) or state.state
            msg = ""
            if not state.address_verified: msg += PROMPTS["ADDRESS_VERIFY_FAIL"] + " "
            if state.state:
                try:
                    hits = await kb_search(f"Does Oasis serve clients in {state.state}?", k=3)
                    text = " ".join([h.get("text","") for h in hits]).lower()
                    if any(p in text for p in ["does not serve","do not serve","not available"]):
                        msg += PROMPTS["STATE_INELIGIBLE"].format(state=state.state) + " "; state.state_eligible="no"
                    elif any(p in text for p in ["serve","available","support"]):
                        msg += PROMPTS["STATE_ELIGIBLE"].format(state=state.state) + " "; state.state_eligible="yes"
                    else:
                        state.state_eligible="unknown"
                except Exception:
                    state.state_eligible = state.state_eligible or "unknown"
            state.stage = "CONFIRM_ADDRESS"
            return respond(state, (msg + PROMPTS["CONFIRM_ADDRESS"].format(address=state.address_norm or state.address)).strip())
        if bump_retry(state, "ASK_ADDRESS"):
            state.address = "NOT PROVIDED"
            state.stage = "ATTORNEY_YN"
            return respond(state, PROMPTS["ADDRESS_SKIPPED"] + " " + PROMPTS["ASK_ATTORNEY_YN"])
        return respond(state, PROMPTS["ASK_ADDRESS"])

    if stage == "CONFIRM_ADDRESS":
        yn = yes_no(utter)
        if yn is True or yn is None:
            state.stage = "ATTORNEY_YN"
            return respond(state, PROMPTS["ASK_ATTORNEY_YN"])
        state.stage = "ASK_ADDRESS"
        state.listen_timeout_sec = LONG_LISTEN
        return respond(state, "Okay, please provide the correct address, including city, state, and ZIP.")

    # ===== ATTORNEY (confirm) =====
    if stage == "ATTORNEY_YN":
        yn = yes_no(utter)
        if yn is not None:
            state.has_attorney = yn
            if yn:
                state.stage = "ATTORNEY_INFO"; state.listen_timeout_sec = LONG_LISTEN
                return respond(state, PROMPTS["ASK_ATTORNEY_INFO"])
            state.stage = "CASE_TYPE"
            return respond(state, PROMPTS["ASK_CASE_TYPE"])
        if bump_retry(state, "ATTORNEY_YN"):
            state.has_attorney = False
            state.stage = "CASE_TYPE"
            return respond(state, PROMPTS["ASK_CASE_TYPE"])
        return respond(state, PROMPTS["ASK_ATTORNEY_YN"])

    if stage == "ATTORNEY_INFO":
        state.listen_timeout_sec = LONG_LISTEN
        if utter:
            state.attorney_phone = normalize_phone(utter) or state.attorney_phone
            if not state.attorney_name: state.attorney_name = clean_text(utter)
            if not state.law_firm: state.law_firm = clean_text(utter)
            if not state.law_firm_address: state.law_firm_address = clean_text(utter)
            state.attorney_verified = await verify_attorney(state.attorney_name, state.law_firm, state.attorney_phone, state.law_firm_address)
            summary = ", ".join([p for p in [
                state.attorney_name, state.law_firm, state.attorney_phone, state.law_firm_address
            ] if p])
            state.stage = "CONFIRM_ATTORNEY"
            return respond(state, PROMPTS["CONFIRM_ATTORNEY"].format(summary=summary))
        if bump_retry(state, "ATTORNEY_INFO"):
            state.stage = "CASE_TYPE"
            return respond(state, PROMPTS["ASK_CASE_TYPE"])
        return respond(state, PROMPTS["ASK_ATTORNEY_INFO"])

    if stage == "CONFIRM_ATTORNEY":
        yn = yes_no(utter)
        if yn is True or yn is None:
            state.stage = "CASE_TYPE"
            return respond(state, PROMPTS["ASK_CASE_TYPE"])
        state.stage = "ATTORNEY_INFO"; state.listen_timeout_sec = LONG_LISTEN
        return respond(state, "Okay, please provide the correct attorney name, phone, law firm, and address.")

    # ===== CASE → INJURY (confirm) =====
    if stage == "CASE_TYPE":
        if utter:
            t = utter.lower()
            if "auto" in t or "car" in t: state.injury_type = "auto accident"
            elif "slip" in t or "fall" in t: state.injury_type = "slip and fall"
            elif "dog" in t or "bite" in t: state.injury_type = "dog bite"
            else: state.injury_type = utter
            state.stage = "INJURY_DETAILS"; state.listen_timeout_sec = LONG_LISTEN
            return respond(state, PROMPTS["ASK_INJURY_DETAILS"])
        if bump_retry(state, "CASE_TYPE"):
            state.stage = "INJURY_DETAILS"; state.listen_timeout_sec = LONG_LISTEN
            return respond(state, PROMPTS["ASK_INJURY_DETAILS"])
        return respond(state, PROMPTS["ASK_CASE_TYPE"])

    if stage == "INJURY_DETAILS":
        state.listen_timeout_sec = LONG_LISTEN
        if utter:
            state.injury_details = clean_text(utter)
            state.stage = "CONFIRM_INJURY_DETAILS"
            return respond(state, PROMPTS["CONFIRM_INJURY_DETAILS"].format(details=state.injury_details))
        if bump_retry(state, "INJURY_DETAILS"):
            state.stage = "INCIDENT_DATE"
            return respond(state, PROMPTS["ASK_INCIDENT_DATE"])
        return respond(state, PROMPTS["ASK_INJURY_DETAILS"])

    if stage == "CONFIRM_INJURY_DETAILS":
        yn = yes_no(utter)
        if yn is True or yn is None:
            state.stage = "INCIDENT_DATE"
            return respond(state, PROMPTS["ASK_INCIDENT_DATE"])
        state.stage = "INJURY_DETAILS"; state.listen_timeout_sec = LONG_LISTEN
        return respond(state, "Okay, please briefly describe what happened again.")

    # ===== INCIDENT → FUNDING =====
    if stage == "INCIDENT_DATE":
        d = parse_incident_date(utter)
        if d:
            state.incident_date = d
            state.stage = "FUNDING_TYPE"
            return respond(state, PROMPTS["ASK_FUNDING_TYPE"])
        if bump_retry(state, "INCIDENT_DATE"):
            state.stage = "FUNDING_TYPE"
            return respond(state, PROMPTS["ASK_FUNDING_TYPE"])
        return respond(state, PROMPTS["ASK_INCIDENT_DATE"])

    if stage == "FUNDING_TYPE":
        t = (utter or "").lower()
        if "fresh" in t or "new" in t: state.funding_type = "fresh"
        elif "extend" in t or "top" in t: state.funding_type = "topup"
        else:
            if bump_retry(state, "FUNDING_TYPE"):
                state.funding_type = state.funding_type or "fresh"
            else:
                return respond(state, PROMPTS["ASK_FUNDING_TYPE"])
        state.stage = "FUNDING_AMOUNT"
        return respond(state, PROMPTS["ASK_FUNDING_AMOUNT"])

    if stage == "FUNDING_AMOUNT":
        amt = extract_amount_usd(utter)
        if amt:
            state.funding_amount = amt
        else:
            if not bump_retry(state, "FUNDING_AMOUNT"):
                return respond(state, PROMPTS["ASK_FUNDING_AMOUNT"])
        state.stage = "SUMMARY"

    # ===== SUMMARY + correction → Q&A =====
    if stage == "SUMMARY":
        t = (utter or "").lower()
        if any(w in t for w in ["human","agent","representative","someone","live person"]):
            state.completed = False
            return respond(state, PROMPTS["HANDOFF"], handoff=True)

        if not state.summary_read:
            parts = []
            if state.full_name: parts.append(f"Name: {state.full_name}")
            if state.best_phone or state.phone: parts.append(f"Phone: {state.best_phone or state.phone}")
            if state.email: parts.append(f"Email: {state.email}")
            if state.address: parts.append(f"Address: {state.address_norm or state.address}")
            if state.state: parts.append(f"State: {state.state} ({state.state_eligible or 'unknown'})")
            if state.has_attorney is not None:
                if state.has_attorney:
                    att = []
                    if state.attorney_name: att.append(state.attorney_name)
                    if state.law_firm: att.append(state.law_firm)
                    if state.attorney_phone: att.append(state.attorney_phone)
                    if state.law_firm_address: att.append(state.law_firm_address)
                    parts.append("Attorney: " + ", ".join(att) if att else "Attorney: provided")
                else:
                    parts.append("Attorney: none")
            if state.injury_type: parts.append(f"Case: {state.injury_type}")
            if state.injury_details: parts.append(f"Details: {state.injury_details}")
            if state.incident_date: parts.append(f"Incident date: {state.incident_date}")
            if state.funding_type: parts.append(f"Funding type: {state.funding_type}")
            if state.funding_amount: parts.append(f"Funding amount: {state.funding_amount}")
            summary = PROMPTS["SUMMARY_INTRO"] + " " + ". ".join(parts) + ". " + PROMPTS["SUMMARY_CONFIRM"]
            state.summary_read = True
            state.awaiting_confirmation = True
            return respond(state, summary)

        if state.awaiting_confirmation:
            yn = yes_no(utter)
            if yn is True:
                state.completed = True
                state.awaiting_confirmation = False
                state.stage = "QNA_OFFER"
                return respond(state, PROMPTS["QNA_OFFER"])
            if yn is False:
                state.correction_mode = True
                state.correction_target = None
                state.summary_read = False
                state.awaiting_confirmation = False
                state.stage = "CORRECT_SELECT"
                return respond(state, PROMPTS["CORRECT_SELECT"])
            if bump_retry(state, "SUMMARY_CONFIRM"):
                state.completed = True
                state.stage = "QNA_OFFER"
                return respond(state, PROMPTS["QNA_OFFER"])
            return respond(state, PROMPTS["SUMMARY_CONFIRM"])

    if stage == "CORRECT_SELECT":
        target = map_correction_field(utter)
        if target:
            state.correction_target = target
            state.correction_mode = True
            if target == "name":
                state.stage = "ASK_NAME"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_NAME"])
            if target == "phone":
                state.stage = "ASK_PHONE"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_PHONE"])
            if target == "email":
                state.stage = "ASK_EMAIL"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_EMAIL"])
            if target == "address":
                state.stage = "ASK_ADDRESS"; state.listen_timeout_sec = LONG_LISTEN
                return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_ADDRESS"])
            if target == "attorney":
                state.stage = "ATTORNEY_YN"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_ATTORNEY_YN"])
            if target == "case":
                state.stage = "CASE_TYPE"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_CASE_TYPE"])
            if target == "incident_date":
                state.stage = "INCIDENT_DATE"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_INCIDENT_DATE"])
            if target == "funding_type":
                state.stage = "FUNDING_TYPE"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_FUNDING_TYPE"])
            if target == "funding_amount":
                state.stage = "FUNDING_AMOUNT"; return respond(state, PROMPTS["CORRECT_ACK"] + " " + PROMPTS["ASK_FUNDING_AMOUNT"])
        if bump_retry(state, "CORRECT_SELECT"):
            return respond(state, "You can say: name, phone, email, address, attorney, case, incident date, funding type, or funding amount.")
        return respond(state, PROMPTS["CORRECT_SELECT"])

    # Q&A mini flow
    if stage == "QNA_OFFER":
        yn = yes_no(utter)
        if yn is False:
            state.stage = "DONE"
            return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        state.stage = "QNA_ASK"; state.listen_timeout_sec = LONG_LISTEN
        return respond(state, PROMPTS["QNA_PROMPT"])

    if stage == "QNA_ASK":
        state.listen_timeout_sec = LONG_LISTEN
        if utter:
            hits = await kb_search(utter, k=3)
            answer = ""
            if hits:
                snippets = []
                for h in hits[:3]:
                    txt = (h.get("text","") or "").strip().replace("\n"," ")
                    if txt: snippets.append(txt[:240])
                answer = " ".join(snippets)[:600]
            if not answer:
                answer = "Here’s what I can share: a specialist will review your case specifics and provide the most accurate guidance shortly."
            state.qna_remaining = max(0, (state.qna_remaining or 1) - 1)
            if state.qna_remaining > 0:
                state.stage = "QNA_ASK"
                return respond(state, answer + " " + PROMPTS["QNA_FOLLOWUP"])
            state.stage = "DONE"
            return respond(state, answer + " " + PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        if bump_retry(state, "QNA_ASK"):
            state.stage = "DONE"
            return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        return respond(state, PROMPTS["QNA_PROMPT"])

    if stage == "DONE":
        state.completed = True
        return respond(state, PROMPTS["DONE"], completed=True)

    return respond(state, "Could you say that again?")
