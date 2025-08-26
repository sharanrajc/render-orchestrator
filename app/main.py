# app/main.py
import time
from typing import Optional, List, Dict
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import ORCH_API_KEY, DEFAULT_LISTEN, LONG_LISTEN
from .db import init_db, SessionLocal, upsert_application_from_state, get_latest_by_caller, Application
from .models import OrchestrateRequest, OrchestrateResponse, SessionState
from .prompts import PROMPTS
from .utils import (
    clean_text, yes_no, extract_name, extract_phone, e164, extract_email,
    extract_state, extract_incident_date, extract_funding_type, extract_amount
)
from .llm_slots import extract_slots
from .tools import verify_address
from .kb_client import kb_search

app = FastAPI(title="Anna Orchestrator", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
init_db()

SESSIONS: Dict[str, SessionState] = {}
TRANSCRIPTS: Dict[str, List[dict]] = {}

FIELD_ORDER = [
    "full_name", "phone", "email", "address", "attorney", "case",
    "injury_details", "incident_date", "funding_type", "funding_amount"
]

def _now_ms() -> int: return int(time.time() * 1000)
def _append_turn(session_id: str, role: str, text: str, stage: str, extra: Optional[dict] = None):
    rec = {"ts": _now_ms(), "role": role, "text": text or "", "stage": stage}
    if extra: rec["meta"] = extra
    TRANSCRIPTS.setdefault(session_id, []).append(rec)
    TRANSCRIPTS[session_id] = TRANSCRIPTS[session_id][-300:]

def respond(state: SessionState, text: str, *, completed=False, handoff=False, citations=None) -> OrchestrateResponse:
    citations = citations or []
    _append_turn(state.session_id, "ai", text or "", state.stage, {"completed": completed, "handoff": handoff, "citations": citations})
    SESSIONS[state.session_id] = state
    state.last_prompt = (text or "")
    with SessionLocal() as db:
        upsert_application_from_state(db, state)
        db.commit()
    return OrchestrateResponse(
        updates=state.to_updates(),
        next_prompt=(text or "")[:360],
        completed=completed,
        handoff=handoff,
        citations=citations,
        confidence=0.8,
        listen_timeout_sec=state.listen_timeout_sec
    )

def _field_missing(s: SessionState, f: str) -> bool:
    return {
        "full_name": not s.full_name,
        "phone": not (s.best_phone or s.phone),
        "email": not s.email,
        "address": (not s.address) and (not getattr(s, "address_skipped", False)),
        "attorney": (s.has_attorney is None) or (s.has_attorney and not (s.attorney_name or s.attorney_phone or s.law_firm)),
        "case": not s.injury_type,
        "injury_details": not s.injury_details,
        "incident_date": not s.incident_date,
        "funding_type": not s.funding_type,
        "funding_amount": not s.funding_amount,
    }[f]

def _next_missing(s: SessionState) -> Optional[str]:
    for f in FIELD_ORDER:
        if _field_missing(s, f): return f
    return None

def _spaced_digits(num: str) -> str:
    return " ".join(list(num or ""))

def _summary(s: SessionState) -> str:
    parts = []
    if s.full_name: parts.append(f"Name: {s.full_name}")
    if s.best_phone or s.phone: parts.append(f"Phone: {s.best_phone or s.phone}")
    if s.email: parts.append(f"Email: {s.email}")
    if s.address or s.address_norm: parts.append(f"Address: {s.address_norm or s.address}")
    if s.has_attorney is not None:
        if s.has_attorney:
            att = [p for p in [s.attorney_name, s.law_firm, s.attorney_phone, s.law_firm_address] if p]
            parts.append("Attorney: " + ", ".join(att) if att else "Attorney: provided")
        else:
            parts.append("Attorney: none")
    if s.injury_type: parts.append(f"Case: {s.injury_type}")
    if s.injury_details: parts.append(f"Details: {s.injury_details}")
    if s.incident_date: parts.append(f"Incident date: {s.incident_date}")
    if s.funding_type: parts.append(f"Funding type: {s.funding_type}")
    if s.funding_amount: parts.append(f"Funding amount: {s.funding_amount}")
    return PROMPTS["SUMMARY_INTRO"] + " " + ". ".join(parts) + ". " + PROMPTS["SUMMARY_CONFIRM"]

@app.get("/health")
def health():
    with SessionLocal() as db:
        total = db.query(Application).count()
    return {"status":"ok","sessions":len(SESSIONS),"applications":total}

@app.get("/applications")
def list_apps(phone: Optional[str] = Query(None)):
    with SessionLocal() as db:
        q = db.query(Application)
        if phone: q = q.filter(Application.best_phone == phone)
        rows = q.order_by(Application.updated_at.desc()).limit(50).all()
        return {"count": len(rows), "applications": [{
            "id": r.id, "session_id": r.session_id, "status": r.status, "caller_number": r.caller_number,
            "full_name": r.full_name, "best_phone": r.best_phone, "email": r.email,
            "address": r.address_norm or r.address, "address_skipped": r.address_skipped,
            "state": r.state, "state_eligible": r.state_eligible,
            "injury_type": r.injury_type, "injury_details": r.injury_details,
            "incident_date": r.incident_date,
            "funding_type": r.funding_type, "funding_amount": r.funding_amount,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        } for r in rows]}

@app.get("/application/{app_id}")
def get_app(app_id: int):
    with SessionLocal() as db:
        row = db.get(Application, app_id)
        if not row: raise HTTPException(404, "not found")
        return {"application": {c.name: getattr(row, c.name) for c in row.__table__.columns}}

@app.get("/transcript/{session_id}")
def get_transcript(session_id: str):
    return {"session_id": session_id, "turns": TRANSCRIPTS.get(session_id, [])}

@app.post("/reset")
def reset(req: OrchestrateRequest, x_api_key: Optional[str] = Header(None)):
    if ORCH_API_KEY and x_api_key != ORCH_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    SESSIONS.pop(req.session_id, None)
    TRANSCRIPTS.pop(req.session_id, None)
    return {"ok": True}

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest, x_api_key: Optional[str] = Header(None)):
    if ORCH_API_KEY and x_api_key != ORCH_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    state = SESSIONS.get(req.session_id) or SessionState(session_id=req.session_id)
    state.listen_timeout_sec = DEFAULT_LISTEN

    if req.caller_number and not state.caller_number:
        state.caller_number = req.caller_number
        # we will confirm using caller id first
        state.best_phone = state.best_phone or (req.caller_number[2:] if req.caller_number.startswith("+1") and len(req.caller_number)==12 else None)

    utter = clean_text(req.last_user_utterance or "")
    if utter:
        _append_turn(req.session_id, "user", utter, state.stage)

    # startup token → force intro & resume check
if utter == "__start__":
    if state.stage == "ENTRY":
        greet = PROMPTS["INTRO"]
        if state.caller_number:
            with SessionLocal() as db:
                latest = get_latest_by_caller(db, state.caller_number)   # ✅ fixed
            if latest:
                state.stage = "RESUME_CHOICE"
                return respond(state, greet + " " + PROMPTS["EXISTING"])
        state.stage = "GREETING"
        return respond(state, greet)

    # human handoff shortcut
    if any(w in utter.lower() for w in ["agent","human","representative","speak to a person","operator"]) and state.stage not in ("DONE",):
        state.completed = False
        return respond(state, "Okay, connecting you to a specialist now.", handoff=True)

    # ENTRY
    if state.stage == "ENTRY":
        greet = PROMPTS["INTRO"]
        if state.caller_number:
            with SessionLocal() as db:
                latest = get_latest_by_caller(db, state.caller_number)
            if latest:
                state.stage = "RESUME_CHOICE"
                return respond(state, greet + " " + PROMPTS["EXISTING"])
        state.stage = "GREETING"
        return respond(state, greet)

    # GREETING → FLOW
    if state.stage == "GREETING":
        state.stage = "FLOW"
        state.listen_timeout_sec = LONG_LISTEN
        return respond(state, PROMPTS["ASK_NAME"])

    # RESUME_CHOICE
    if state.stage == "RESUME_CHOICE":
        t = utter.lower()
        if any(w in t for w in ["continue","resume","pending","yeah","yes","yep","ok","okay"]):
            state.stage = "FLOW"
            # Confirm caller number first before asking new info
            if state.best_phone and not state.flags.get("caller_confirmed"):
                state.awaiting_confirm_field = "caller_phone"
                return respond(state, PROMPTS["ASK_CONFIRM_CALLER_PHONE"].format(phone=_spaced_digits(state.best_phone)))
            return respond(state, PROMPTS["ASK_NAME"])
        if any(w in t for w in ["modify","update","edit","change"]):
            state.stage = "SUMMARY"
            state.summary_read = False
            return respond(state, PROMPTS["MODIFY_ACK"])
        if any(w in t for w in ["new","start","start new","stop","cancel"]):
            state = SessionState(session_id=state.session_id, caller_number=state.caller_number)
            SESSIONS[state.session_id] = state
            state.stage = "FLOW"
            return respond(state, PROMPTS["NEW_ACK"] + " " + PROMPTS["ASK_NAME"])
        # fallback after 2 tries → continue
        tries = state.retries.get("RESUME_CHOICE", 0)
        if tries >= 2:
            state.stage = "FLOW"
            if state.best_phone and not state.flags.get("caller_confirmed"):
                state.awaiting_confirm_field = "caller_phone"
                return respond(state, PROMPTS["ASK_CONFIRM_CALLER_PHONE"].format(phone=_spaced_digits(state.best_phone)))
            return respond(state, PROMPTS["DEFAULT_CONTINUE"])
        state.retries["RESUME_CHOICE"] = tries + 1
        return respond(state, PROMPTS["EXISTING"])

    # FLOW — extract & advance
    if state.stage == "FLOW":
        # LLM soft assist
        slots = extract_slots(utter) if utter else {}
        conf = slots.get("_confidence", {})

        def set_if(k: str, cur: Optional[str], newv: Optional[str]) -> Optional[str]:
            if not newv: return cur
            nv = (newv or "").strip()
            if not cur or (cur != nv):
                state.confidences[k] = float(conf.get(k, 0.0) or 0.0)
                return nv
            return cur

        # Apply likely slots
        if "full_name" in slots: state.full_name = set_if("full_name", state.full_name, slots["full_name"])
        if "phone" in slots: state.phone = set_if("phone", state.phone, slots["phone"])
        if "email" in slots: state.email = set_if("email", state.email, slots["email"])
        if "address" in slots: state.address = set_if("address", state.address, slots["address"])
        if "state" in slots: state.state = set_if("state", state.state, slots["state"])
        if "injury_type" in slots: state.injury_type = set_if("injury_type", state.injury_type, slots["injury_type"])
        if "injury_details" in slots: state.injury_details = set_if("injury_details", state.injury_details, slots["injury_details"])
        if "incident_date" in slots: state.incident_date = set_if("incident_date", state.incident_date, slots["incident_date"])
        if "funding_type" in slots: state.funding_type = set_if("funding_type", state.funding_type, slots["funding_type"])
        if "funding_amount" in slots: state.funding_amount = set_if("funding_amount", state.funding_amount, slots["funding_amount"])
        if "has_attorney" in slots and state.has_attorney is None:
            state.has_attorney = slots["has_attorney"]
        if "attorney_name" in slots: state.attorney_name = set_if("attorney_name", state.attorney_name, slots["attorney_name"])
        if "attorney_phone" in slots: state.attorney_phone = set_if("attorney_phone", state.attorney_phone, slots["attorney_phone"])
        if "law_firm" in slots: state.law_firm = set_if("law_firm", state.law_firm, slots["law_firm"])
        if "law_firm_address" in slots: state.law_firm_address = set_if("law_firm_address", state.law_firm_address, slots["law_firm_address"])

        # Fallback regex extractors
        if utter:
            if not state.full_name:
                nm = extract_name(utter)
                if nm: state.full_name = nm
            if not (state.phone or state.best_phone):
                p = extract_phone(utter)
                if p == "__SAME__" and state.caller_number and state.caller_number.startswith("+1"):
                    state.best_phone = state.caller_number[2:]
                elif p and p != "__SAME__":
                    state.phone = p
                    state.best_phone = p
            if not state.email:
                em = extract_email(utter)
                if em: state.email = em
            if not state.address:
                if utter.lower() in {"skip","skip address","no address"}:
                    state.address_skipped = True
                elif len(utter.split()) >= 4 and not extract_email(utter):
                    state.address = utter
            if not state.state and state.address:
                st = extract_state(state.address)
                if st: state.state = st
            if not state.incident_date:
                dt = extract_incident_date(utter)
                if dt: state.incident_date = dt
            if not state.funding_type:
                ft = extract_funding_type(utter)
                if ft: state.funding_type = ft
            if not state.funding_amount:
                fa = extract_amount(utter)
                if fa: state.funding_amount = fa

        # Confirm caller phone once (preferred number)
        if state.best_phone and not state.flags.get("caller_confirmed"):
            state.awaiting_confirm_field = "caller_phone"
            return respond(state, PROMPTS["ASK_CONFIRM_CALLER_PHONE"].format(phone=_spaced_digits(state.best_phone)))

        # Confirm email once (spelling)
        if state.email and not state.flags.get("email_confirmed"):
            state.awaiting_confirm_field = "email"
            from .utils import spell_for_email
            return respond(state, PROMPTS["CONFIRM_EMAIL"].format(email=state.email, spelled=spell_for_email(state.email)))

        # Handle confirmations
        if state.awaiting_confirm_field:
            yn = yes_no(utter)
            target = state.awaiting_confirm_field
            if yn is True:
                state.flags[f"{target}_confirmed"] = True
                state.awaiting_confirm_field = None
            elif yn is False:
                state.awaiting_confirm_field = None
                if target == "caller_phone":
                    return respond(state, PROMPTS["ASK_PHONE"])
                if target == "email":
                    return respond(state, PROMPTS["EMAIL_SPELL_PROMPT"])
            else:
                # assume-yes to avoid loops
                state.flags[f"{target}_confirmed"] = True
                state.awaiting_confirm_field = None

        # After capturing address → try verify (non-blocking)
        if state.address and not state.address_norm and not state.address_skipped:
            norm, verified, st = verify_address(state.address)
            state.address_norm = norm or state.address
            state.address_verified = bool(verified)
            if st and not state.state: state.state = st

        # Next missing?
        nxt = _next_missing(state)
        if not nxt:
            state.stage = "SUMMARY"
            state.summary_read = False
            return respond(state, "Looks like we have everything. I’ll read back your details.")

        ask = {
            "full_name": PROMPTS["ASK_NAME"],
            "phone": PROMPTS["ASK_PHONE"],
            "email": PROMPTS["ASK_EMAIL"],
            "address": PROMPTS["ASK_ADDRESS"],
            "attorney": PROMPTS["ASK_ATTORNEY_YN"] if state.has_attorney is None else PROMPTS["ASK_ATTORNEY_INFO"],
            "case": PROMPTS["ASK_CASE_TYPE"],
            "injury_details": PROMPTS["ASK_INJURY_DETAILS"],
            "incident_date": PROMPTS["ASK_INCIDENT_DATE"],
            "funding_type": PROMPTS["ASK_FUNDING_TYPE"],
            "funding_amount": PROMPTS["ASK_FUNDING_AMOUNT"],
        }[nxt]
        if nxt in ("address","injury_details","attorney"):
            state.listen_timeout_sec = LONG_LISTEN
        return respond(state, ask)

    # SUMMARY
    if state.stage == "SUMMARY":
        if not state.summary_read:
            state.summary_read = True
            return respond(state, _summary(state))
        yn = yes_no(utter)
        if yn is True:
            state.completed = True
            state.stage = "QNA_OFFER"
            return respond(state, PROMPTS["QNA_OFFER"])
        if yn is False:
            state.stage = "CORRECT_SELECT"
            return respond(state, PROMPTS["CORRECT_SELECT"])
        # assume yes
        state.completed = True
        state.stage = "QNA_OFFER"
        return respond(state, PROMPTS["QNA_OFFER"])

    if state.stage == "CORRECT_SELECT":
        t = (utter or "").lower()
        mapping = {
            "name": "full_name", "phone": "phone", "number": "phone",
            "email": "email", "address": "address", "attorney": "attorney",
            "case": "case", "incident": "incident_date",
            "funding type": "funding_type", "amount": "funding_amount"
        }
        target = None
        for k, v in mapping.items():
            if k in t: target = v; break
        state.stage = "FLOW"
        if target == "full_name": return respond(state, PROMPTS["ASK_NAME"])
        prompts_map = {
            "phone": "ASK_PHONE","email": "ASK_EMAIL","address":"ASK_ADDRESS","attorney":"ASK_ATTORNEY_INFO",
            "case":"ASK_CASE_TYPE","incident_date":"ASK_INCIDENT_DATE","funding_type":"ASK_FUNDING_TYPE","funding_amount":"ASK_FUNDING_AMOUNT"
        }
        return respond(state, PROMPTS.get(prompts_map.get(target,"ASK_NAME"), PROMPTS["ASK_NAME"]))

    if state.stage == "QNA_OFFER":
        yn = yes_no(utter)
        if yn is False:
            state.stage = "DONE"; state.completed = True
            return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        state.stage = "QNA_ASK"; state.listen_timeout_sec = LONG_LISTEN
        return respond(state, PROMPTS["QNA_PROMPT"])

    if state.stage == "QNA_ASK":
        if utter:
            hits = await kb_search(utter, k=3)
            answer = " ".join([(h.get("text","") or "").replace("\n"," ")[:240] for h in hits])[:600] if hits else ""
            if not answer:
                answer = "Here’s what I can share: a specialist will review your case specifics and provide the most accurate guidance shortly."
            state.stage = "DONE"; state.completed = True
            return respond(state, answer + " " + PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        state.stage = "DONE"; state.completed = True
        return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)

    if state.stage == "DONE":
        state.completed = True
        return respond(state, PROMPTS["DONE"], completed=True)

    return respond(state, "Could you say that again?")
