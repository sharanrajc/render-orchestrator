# app/main.py
import time, re
from typing import Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import ORCH_API_KEY, DEFAULT_LISTEN, LONG_LISTEN
from .db import init_db, SessionLocal, upsert_application_from_state, get_latest_by_caller, Application
from .models import OrchestrateRequest, OrchestrateResponse, SessionState
from .prompts import PROMPTS
from .utils import (
    clean_text, yes_no, spell_for_name, spell_for_email,
    extract_name, extract_phone, e164, extract_email,
    said_skip, extract_state, extract_incident_date,
    extract_funding_type, extract_amount
)
from .llm_slots import extract_slots
from .tools import verify_address, verify_attorney
from .kb_client import kb_search

app = FastAPI(title="Orchestrator (stabilized)", version="4.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
init_db()

SESSIONS: Dict[str, SessionState] = {}
TRANSCRIPTS: Dict[str, List[dict]] = {}

MAX_RETRIES_PER_STAGE = 2
CONFIRM_MAX_REPROMPTS = 1   # confirmations are one-and-done
TRANSCRIPT_MAX_TURNS = 300

FIELD_ORDER = [
    "full_name", "phone", "email", "address", "attorney", "case",
    "injury_details", "incident_date", "funding_type", "funding_amount"
]

def _now_ms() -> int: return int(time.time() * 1000)

def _append_turn(session_id: str, role: str, text: str, stage: str, extra: Optional[dict] = None):
    rec = {"ts": _now_ms(), "role": role, "text": text or "", "stage": stage}
    if extra: rec["meta"] = extra
    lst = TRANSCRIPTS.setdefault(session_id, [])
    lst.append(rec)
    if len(lst) > TRANSCRIPT_MAX_TURNS: TRANSCRIPTS[session_id] = lst[-TRANSCRIPT_MAX_TURNS:]

def respond(state: SessionState, text: str, *, completed=False, handoff=False, citations=None, confidence: float = 0.8):
    citations = citations or []
    _append_turn(state.session_id, "ai", text or "", state.stage, {"completed": completed, "handoff": handoff, "citations": citations})
    SESSIONS[state.session_id] = state
    state.last_prompt = (text or "")
    with SessionLocal() as db:
        upsert_application_from_state(db, state)
        db.commit()
    return OrchestrateResponse(
        updates=state.to_updates(),
        next_prompt=(text or "")[:180],
        completed=completed, handoff=handoff, citations=citations,
        confidence=confidence, listen_timeout_sec=state.listen_timeout_sec,
    )

def _field_missing(s: SessionState, f: str) -> bool:
    return {
        "full_name": not s.full_name,
        "phone": not (s.best_phone or s.phone),
        "email": not s.email,
        "address": not s.address and not s.address_skipped,
        "attorney": s.has_attorney is None or (s.has_attorney and not (s.attorney_name or s.attorney_phone or s.law_firm)),
        "case": not s.injury_type,
        "injury_details": not s.injury_details,
        "incident_date": not s.incident_date,
        "funding_type": not s.funding_type,
        "funding_amount": not s.funding_amount,
    }[f]

def _next_missing(s: SessionState) -> Optional[str]:
    for f in FIELD_ORDER:
        if _field_missing(s, f):
            return f
    return None

def _attorney_summary(s: SessionState) -> str:
    parts = [p for p in [s.attorney_name, s.law_firm, s.attorney_phone, s.law_firm_address] if p]
    return ", ".join(parts) if parts else "attorney info provided"

def _summarize(s: SessionState) -> str:
    parts = []
    if s.full_name: parts.append(f"Name: {s.full_name}")
    if s.best_phone or s.phone: parts.append(f"Phone: {s.best_phone or s.phone}")
    if s.email: parts.append(f"Email: {s.email}")
    if s.address or s.address_norm: parts.append(f"Address: {s.address_norm or s.address}")
    if s.state: parts.append(f"State: {s.state} ({s.state_eligible or 'unknown'})")
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

# ------------------- Public endpoints -------------------

from fastapi import APIRouter
router = APIRouter()

@app.get("/health")
def health():
    with SessionLocal() as db:
        total = db.query(Application).count()
    return {"status": "ok", "sessions": len(SESSIONS), "applications": total}

@app.get("/applications")
def list_apps(phone: Optional[str] = Query(None)):
    with SessionLocal() as db:
        if phone:
            rows = db.query(Application).filter(Application.best_phone == phone).order_by(Application.updated_at.desc()).all()
        else:
            rows = db.query(Application).order_by(Application.updated_at.desc()).limit(50).all()
        return {"count": len(rows), "applications": [{
            "id": r.id, "session_id": r.session_id, "status": r.status, "caller_number": r.caller_number,
            "full_name": r.full_name, "best_phone": r.best_phone, "email": r.email,
            "address": r.address_norm or r.address, "state": r.state,
            "injury_type": r.injury_type, "injury_details": r.injury_details,
            "incident_date": r.incident_date, "funding_type": r.funding_type, "funding_amount": r.funding_amount,
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

# ------------------- Orchestrator -------------------

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest, x_api_key: Optional[str] = Header(None)):
    if ORCH_API_KEY and x_api_key != ORCH_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    state = SESSIONS.get(req.session_id) or SessionState(session_id=req.session_id)
    state.listen_timeout_sec = DEFAULT_LISTEN

    if req.caller_number and not state.caller_number:
        state.caller_number = req.caller_number
        state.best_phone = state.best_phone or req.caller_number

    utter = clean_text(req.last_user_utterance or "")
    stage = state.stage
    if utter:
        TRANSCRIPTS.setdefault(req.session_id, [])
        _append_turn(req.session_id, "user", utter, stage)

    # Treat startup token as empty utterance but force ENTRY greeting
    if (req.last_user_utterance or "").strip() == "__start__":
        utter = ""
        if state.stage == "ENTRY":
            greet = PROMPTS["INTRO"]
            if state.caller_number:
                with SessionLocal() as db:
                    latest = get_latest_by_caller(db, state.caller_number)
                if latest:
                    state.stage = "RESUME_CHOICE"
                    return respond(state, greet + " " + PROMPTS["EXISTING_V2"])
            state.stage = "GREETING"
            return respond(state, greet)

    # ENTRY → RESUME or GREETING
    if stage == "ENTRY":
        greet = PROMPTS["INTRO"]
        if state.caller_number:
            with SessionLocal() as db:
                latest = get_latest_by_caller(db, state.caller_number)
            if latest:
                state.stage = "RESUME_CHOICE"
                return respond(state, greet + " " + PROMPTS["EXISTING_V2"])
        state.stage = "GREETING"
        return respond(state, greet)

    if stage == "GREETING":
        state.stage = "FLOW"
        state.listen_timeout_sec = LONG_LISTEN
        state.retries["ASK_NAME"] = 0
        return respond(state, PROMPTS["ASK_NAME_V2"])

    if stage == "RESUME_CHOICE":
        t = (utter or "").lower()
        if any(w in t for w in ["continue", "resume", "pending", "yeah", "yes", "yep", "ok", "okay", "go ahead"]):
            state.stage = "FLOW"
            nxt = _next_missing(state) or "summary"
            return respond(state, PROMPTS["CONTINUE_ACK"] + " " + (PROMPTS["ASK_NAME_V2"] if nxt == "full_name" else PROMPTS["ASK_PHONE"]))
        if any(w in t for w in ["modify", "update", "edit", "change"]):
            state.stage = "SUMMARY"
            state.summary_read = False
            state.awaiting_confirmation = False
            return respond(state, PROMPTS["MODIFY_ACK"])
        if any(w in t for w in ["new", "start", "start new", "stop", "cancel"]):
            state = SessionState(session_id=state.session_id, caller_number=state.caller_number)
            SESSIONS[state.session_id] = state
            state.stage = "FLOW"
            state.listen_timeout_sec = LONG_LISTEN
            state.retries["ASK_NAME"] = 0
            return respond(state, PROMPTS["NEW_ACK"] + " " + PROMPTS["ASK_NAME_V2"])
        tries = state.retries.get("RESUME_CHOICE", 0)
        if tries >= MAX_RETRIES_PER_STAGE:
            state.stage = "FLOW"
            return respond(state, PROMPTS["DEFAULT_CONTINUE"])
        state.retries["RESUME_CHOICE"] = tries + 1
        return respond(state, PROMPTS["EXISTING_V2"])

    # ------------------- FLOW -------------------
    if stage == "FLOW":
        # Apply LLM slots (soft), then regex fallbacks (hard)
        wanted = []
        nxt = _next_missing(state)
        if nxt == "attorney":
            wanted = ["has_attorney", "attorney_name", "attorney_phone", "law_firm", "law_firm_address"]
        elif nxt == "case":
            wanted = ["injury_type"]
        elif nxt:
            wanted = [nxt]
        slots = extract_slots(utter, wanted_fields=wanted) if utter else {}
        conf = slots.get("_confidence", {})

        def set_if(k: str, cur: Optional[str], newv: Optional[str]) -> Optional[str]:
            if not newv: return cur
            nv = clean_text(newv)
            if not cur or (cur != nv):
                state.confidences[k] = float(conf.get(k, 0.0) or 0.0)
                return nv
            return cur

        # LLM first
        if "full_name" in slots: state.full_name = set_if("full_name", state.full_name, slots["full_name"])
        if "phone" in slots: state.phone = set_if("phone", state.phone, slots["phone"])
        if "email" in slots: state.email = set_if("email", state.email, slots["email"])
        if "address" in slots: state.address = set_if("address", state.address, slots["address"])
        if "state" in slots: state.state = set_if("state", state.state, slots["state"])
        if "has_attorney" in slots: state.has_attorney = slots["has_attorney"]
        if "attorney_name" in slots: state.attorney_name = set_if("attorney_name", state.attorney_name, slots["attorney_name"])
        if "attorney_phone" in slots: state.attorney_phone = set_if("attorney_phone", state.attorney_phone, slots["attorney_phone"])
        if "law_firm" in slots: state.law_firm = set_if("law_firm", state.law_firm, slots["law_firm"])
        if "law_firm_address" in slots: state.law_firm_address = set_if("law_firm_address", state.law_firm_address, slots["law_firm_address"])
        if "injury_type" in slots: state.injury_type = set_if("injury_type", state.injury_type, slots["injury_type"])
        if "injury_details" in slots: state.injury_details = set_if("injury_details", state.injury_details, slots["injury_details"])
        if "incident_date" in slots: state.incident_date = set_if("incident_date", state.incident_date, slots["incident_date"])
        if "funding_type" in slots: state.funding_type = set_if("funding_type", state.funding_type, slots["funding_type"])
        if "funding_amount" in slots: state.funding_amount = set_if("funding_amount", state.funding_amount, slots["funding_amount"])

        # Hard fallbacks
        if not state.full_name and utter:
            name_fb = extract_name(utter)
            if name_fb: state.full_name = name_fb

        if not (state.phone or state.best_phone) and utter:
            p10 = extract_phone(utter)
            if p10:
                state.phone = p10
                state.best_phone = p10

        if not state.email and utter:
            em = extract_email(utter)
            if em: state.email = em

        if not state.address and utter:
            if said_skip(utter):
                state.address_skipped = True
            elif len(utter.split()) >= 4 and not extract_email(utter) and not extract_phone(utter):
                state.address = utter
                st = extract_state(utter)
                if st: state.state = st

        if not state.incident_date and utter:
            dt = extract_incident_date(utter)
            if dt: state.incident_date = dt

        if not state.funding_type and utter:
            ft = extract_funding_type(utter)
            if ft: state.funding_type = ft

        if not state.funding_amount and utter:
            amt = extract_amount(utter)
            if amt: state.funding_amount = amt

        # Confirmations (one-shot). Only email + phone to keep friction low.
        if state.email and not state._flags.get("email_confirmed"):
            state._confirm_retries = state._confirm_retries + 1 if hasattr(state, "_confirm_retries") else 0
            state.awaiting_confirm_field = "email"
            return respond(state, PROMPTS["CONFIRM_EMAIL_SPELL"].format(email=state.email, spelled=spell_for_email(state.email)))

        if (state.best_phone or state.phone) and not state._flags.get("phone_confirmed"):
            pn = state.best_phone or state.phone
            # format digits spaced so TTS reads them cleanly
            spaced = " ".join(list(pn))
            state.awaiting_confirm_field = "phone"
            return respond(state, PROMPTS["CONFIRM_PHONE"].format(phone=spaced))

        # Handle confirmations response (one-and-done)
        if state.awaiting_confirm_field:
            f = state.awaiting_confirm_field
            yn = yes_no(utter)
            if yn is True:
                state._flags[f"{f}_confirmed"] = True
                state.awaiting_confirm_field = None
            elif yn is False:
                state.awaiting_confirm_field = None
                # Ask the field again once, then move on
                if f == "email":
                    return respond(state, PROMPTS["EMAIL_SPELL_PROMPT"])
                if f == "phone":
                    return respond(state, PROMPTS["ASK_PHONE"])
            else:
                # no clear answer; proceed (avoid loops)
                state.awaiting_confirm_field = None

        # Next missing?
        nxt = _next_missing(state)
        if not nxt:
            state.stage = "SUMMARY"
            state.summary_read = False
            state.awaiting_confirmation = True
            return respond(state, "Looks like we have everything. I’ll read back your details.")

        # Ask prompts with long listen for verbose fields
        ask = {
            "full_name": PROMPTS["ASK_NAME_V2"],
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
        if nxt in ("address", "injury_details", "attorney"):
            state.listen_timeout_sec = LONG_LISTEN
        return respond(state, ask)

    # SUMMARY / CORRECT / Q&A
    if stage == "SUMMARY":
        if not state.summary_read:
            state.summary_read = True
            return respond(state, _summarize(state))
        yn = yes_no(utter)
        if yn is True:
            state.completed = True
            state.stage = "QNA_OFFER"
            return respond(state, PROMPTS["QNA_OFFER"])
        if yn is False:
            state.stage = "CORRECT_SELECT"
            return respond(state, PROMPTS["CORRECT_SELECT"])
        # no clear answer → continue
        state.completed = True
        state.stage = "QNA_OFFER"
        return respond(state, PROMPTS["QNA_OFFER"])

    if stage == "CORRECT_SELECT":
        t = (utter or "").lower()
        mapping = {"name":"full_name","phone":"phone","number":"phone","email":"email","address":"address","attorney":"attorney","case":"case","incident":"incident_date","funding type":"funding_type","amount":"funding_amount"}
        target = None
        for k,v in mapping.items():
            if k in t: target = v; break
        state.stage = "FLOW"
        return respond(state, PROMPTS["CORRECT_ACK"] + " " + (PROMPTS["ASK_NAME_V2"] if target=="full_name" else PROMPTS.get({
            "phone":"ASK_PHONE","email":"ASK_EMAIL","address":"ASK_ADDRESS","attorney":"ASK_ATTORNEY_INFO","case":"ASK_CASE_TYPE","incident_date":"ASK_INCIDENT_DATE","funding_type":"ASK_FUNDING_TYPE","funding_amount":"ASK_FUNDING_AMOUNT"
        }.get(target, "ASK_NAME_V2"))))

    if stage == "QNA_OFFER":
        yn = yes_no(utter)
        if yn is False:
            state.stage = "DONE"; state.completed = True
            return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        state.stage = "QNA_ASK"; state.listen_timeout_sec = LONG_LISTEN
        return respond(state, PROMPTS["QNA_PROMPT"])

    if stage == "QNA_ASK":
        if utter:
            hits = await kb_search(utter, k=3)
            answer = " ".join([(h.get("text","") or "").replace("\n"," ")[:240] for h in hits])[:600] if hits else ""
            if not answer:
                answer = "Here’s what I can share: a specialist will review your case specifics and provide the most accurate guidance shortly."
            state.stage = "DONE"; state.completed = True
            return respond(state, answer + " " + PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        state.stage = "DONE"; state.completed = True
        return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)

    if stage == "DONE":
        state.completed = True
        return respond(state, PROMPTS["DONE"], completed=True)

    return respond(state, "Could you say that again?")
