# app/main.py
import time, re
from typing import Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import ORCH_API_KEY, DEFAULT_LISTEN, LONG_LISTEN
from .db import init_db, SessionLocal, upsert_application_from_state, get_latest_by_caller, Application
from .models import OrchestrateRequest, OrchestrateResponse, SessionState
from .prompts import PROMPTS
from .utils import clean_text, yes_no, spell_for_name, spell_for_email
from .llm_slots import extract_slots
from .tools import verify_address, verify_attorney
from .kb_client import kb_search

app = FastAPI(title="Orchestrator (LLM slots, anti-loop)", version="4.0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
init_db()

SESSIONS: Dict[str, SessionState] = {}
TRANSCRIPTS: Dict[str, List[dict]] = {}
MAX_RETRIES_PER_STAGE = 3
TRANSCRIPT_MAX_TURNS = 300

FIELD_ORDER = [
    "full_name", "phone", "email", "address", "attorney", "case",
    "injury_details", "incident_date", "funding_type", "funding_amount"
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _append_turn(session_id: str, role: str, text: str, stage: str, extra: Optional[dict] = None):
    rec = {"ts": _now_ms(), "role": role, "text": text or "", "stage": stage}
    if extra:
        rec["meta"] = extra
    lst = TRANSCRIPTS.setdefault(session_id, [])
    lst.append(rec)
    if len(lst) > TRANSCRIPT_MAX_TURNS:
        TRANSCRIPTS[session_id] = lst[-TRANSCRIPT_MAX_TURNS:]


def respond(
    state: SessionState,
    text: str,
    *,
    completed: bool = False,
    handoff: bool = False,
    citations: Optional[List[int]] = None,
    confidence: float = 0.8
):
    """Build the OrchestrateResponse, persist state to DB, and append transcript."""
    citations = citations or []
    _append_turn(
        state.session_id,
        "ai",
        text or "",
        state.stage,
        {"completed": completed, "handoff": handoff, "citations": citations},
    )
    SESSIONS[state.session_id] = state
    state.last_prompt = (text or "")
    with SessionLocal() as db:
        upsert_application_from_state(db, state)
        db.commit()
    return OrchestrateResponse(
        updates=state.to_updates(),
        next_prompt=(text or "")[:180],
        completed=completed,
        handoff=handoff,
        citations=citations,
        confidence=confidence,
        listen_timeout_sec=state.listen_timeout_sec,
    )


def _field_missing(state: SessionState, f: str) -> bool:
    if f == "full_name":
        return not state.full_name
    if f == "phone":
        return not (state.best_phone or state.phone)
    if f == "email":
        return not state.email
    if f == "address":
        return not state.address
    if f == "attorney":
        return state.has_attorney is None or (
            state.has_attorney and not (state.attorney_name or state.attorney_phone or state.law_firm)
        )
    if f == "case":
        return not state.injury_type
    if f == "injury_details":
        return not state.injury_details
    if f == "incident_date":
        return not state.incident_date
    if f == "funding_type":
        return not state.funding_type
    if f == "funding_amount":
        return not state.funding_amount
    return False


def _next_missing(state: SessionState) -> Optional[str]:
    for f in FIELD_ORDER:
        if _field_missing(state, f):
            return f
    return None


def _attorney_summary(state: SessionState) -> str:
    parts = [p for p in [state.attorney_name, state.law_firm, state.attorney_phone, state.law_firm_address] if p]
    return ", ".join(parts) if parts else "attorney info provided"


def _summarize(state: SessionState) -> str:
    parts = []
    if state.full_name:
        parts.append(f"Name: {state.full_name}")
    if state.best_phone or state.phone:
        parts.append(f"Phone: {state.best_phone or state.phone}")
    if state.email:
        parts.append(f"Email: {state.email}")
    if state.address:
        parts.append(f"Address: {state.address_norm or state.address}")
    if state.state:
        parts.append(f"State: {state.state} ({state.state_eligible or 'unknown'})")
    if state.has_attorney is not None:
        if state.has_attorney:
            att = []
            if state.attorney_name:
                att.append(state.attorney_name)
            if state.law_firm:
                att.append(state.law_firm)
            if state.attorney_phone:
                att.append(state.attorney_phone)
            if state.law_firm_address:
                att.append(state.law_firm_address)
            parts.append("Attorney: " + ", ".join(att) if att else "Attorney: provided")
        else:
            parts.append("Attorney: none")
    if state.injury_type:
        parts.append(f"Case: {state.injury_type}")
    if state.injury_details:
        parts.append(f"Details: {state.injury_details}")
    if state.incident_date:
        parts.append(f"Incident date: {state.incident_date}")
    if state.funding_type:
        parts.append(f"Funding type: {state.funding_type}")
    if state.funding_amount:
        parts.append(f"Funding amount: {state.funding_amount}")
    return PROMPTS["SUMMARY_INTRO"] + " " + ". ".join(parts) + ". " + PROMPTS["SUMMARY_CONFIRM"]


# ---------- anti-loop helpers ----------

_NAME_PREFIXES = [
    r"\bmy (full legal )?name is\b",
    r"\bthis is\b",
    r"\bi am\b",
    r"\bit is\b",
    r"\bthe name is\b",
    r"\byou can call me\b",
]

def _fallback_name_extract(text: str) -> Optional[str]:
    """Deterministic backup: pull 'Joe Smith' from 'My full legal name is Joe Smith' etc."""
    if not text:
        return None
    t = " " + text.strip() + " "
    for pref in _NAME_PREFIXES:
        m = re.search(pref + r"\s+([A-Za-z][A-Za-z .'\-]{0,80})", t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip().strip(" .,!?:;\"'()[]")
            toks = [w for w in cand.split() if re.search(r"[A-Za-z]", w)]
            if 1 <= len(toks) <= 4:
                def _tc(w): return w if re.fullmatch(r"[A-Za-z]\.", w) else w.capitalize()
                return " ".join(_tc(w) for w in toks)
    # if whole utterance looks like a name
    cand = text.strip().strip(" .,!?:;\"'()[]")
    toks = [w for w in cand.split() if re.search(r"[A-Za-z]", w)]
    if 1 <= len(toks) <= 4 and not any(ch.isdigit() for ch in cand) and "@" not in cand:
        def _tc(w): return w if re.fullmatch(r"[A-Za-z]\.", w) else w.capitalize()
        return " ".join(_tc(w) for w in toks)
    return None

def _name_prompt_for_try(tries: int) -> str:
    if tries <= 0:
        return PROMPTS["ASK_NAME"] + " You can simply say, “Joe Smith.”"
    if tries == 1:
        return "Please say just your first and last name, like “Joe Smith.”"
    if tries == 2:
        return "Please spell your full legal name, letter by letter."
    # tries >= 3
    return "I’m still not getting that. Would you like me to connect you to a specialist, or would you like to try your name once more?"


def _resume_prompt_for_try(tries: int) -> str:
    base = "I see an application linked to this number. Please say “continue”, “modify”, or “start new application.”"
    if tries == 0:
        return base
    if tries == 1:
        return "Say “continue” to pick up where you left off, “modify” to change an application, or “start new application.”"
    if tries >= 2:
        return "Would you like to continue the pending application, modify a completed one, or start a new application? Please say one of: “continue”, “modify”, or “start new.”"


# ------------------- Public endpoints -------------------

@app.get("/health")
def health():
    with SessionLocal() as db:
        total = db.query(Application).count()
    return {"status": "ok", "sessions": len(SESSIONS), "applications": total}


@app.get("/applications")
def list_apps(phone: Optional[str] = Query(None)):
    with SessionLocal() as db:
        if phone:
            rows = (
                db.query(Application)
                .filter(Application.best_phone == phone)
                .order_by(Application.updated_at.desc())
                .all()
            )
        else:
            rows = db.query(Application).order_by(Application.updated_at.desc()).limit(50).all()
        return {
            "count": len(rows),
            "applications": [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "status": r.status,
                    "caller_number": r.caller_number,
                    "full_name": r.full_name,
                    "best_phone": r.best_phone,
                    "email": r.email,
                    "address": r.address_norm or r.address,
                    "state": r.state,
                    "injury_type": r.injury_type,
                    "injury_details": r.injury_details,
                    "incident_date": r.incident_date,
                    "funding_type": r.funding_type,
                    "funding_amount": r.funding_amount,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ],
        }


@app.get("/application/{app_id}")
def get_app(app_id: int):
    with SessionLocal() as db:
        row = db.get(Application, app_id)
        if not row:
            raise HTTPException(404, "not found")
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
        state.best_phone = req.caller_number  # tentative; will confirm later

    utter = clean_text(req.last_user_utterance or "")
    stage = state.stage
    if utter:
        _append_turn(req.session_id, "user", utter, stage)

    # ---------- ENTRY / GREETING / RESUME ----------
    if stage == "ENTRY":
        greet = PROMPTS["INTRO"]
        if state.caller_number:
            with SessionLocal() as db:
                latest = get_latest_by_caller(db, state.caller_number)
            if latest:
                state.stage = "RESUME_CHOICE"
                state.listen_timeout_sec = DEFAULT_LISTEN
                tries = state.retries.get("RESUME_CHOICE", 0)
                return respond(state, greet + " " + _resume_prompt_for_try(tries))
        state.stage = "GREETING"
        return respond(state, greet)

    if stage == "GREETING":
        state.stage = "FLOW"
        state.listen_timeout_sec = LONG_LISTEN  # allow natural “my full legal name is …”
        # first time ask (tries=0)
        state.retries["ASK_NAME"] = 0
        return respond(state, _name_prompt_for_try(0))

    if stage == "RESUME_CHOICE":
        t = (utter or "").lower()
        if any(w in t for w in ["continue", "resume", "pending"]):
            state.stage = "FLOW"
            nxt = _next_missing(state) or "summary"
            # move on
            return respond(
                state,
                "Got it—continuing your application. " + (PROMPTS["ASK_NAME"] if nxt == "full_name" else PROMPTS["ASK_PHONE"])
            )
        if any(w in t for w in ["modify", "update", "edit", "change"]):
            state.stage = "SUMMARY"
            state.summary_read = False
            state.awaiting_confirmation = False
            return respond(state, "Okay, I’ll read back your details and we can make changes.")
        if any(w in t for w in ["new", "start"]):
            state = SessionState(session_id=state.session_id, caller_number=state.caller_number)
            SESSIONS[state.session_id] = state
            state.stage = "FLOW"
            state.listen_timeout_sec = LONG_LISTEN
            state.retries["ASK_NAME"] = 0
            return respond(state, "Starting a new application. " + _name_prompt_for_try(0))

        # no/ambiguous input → escalate, then default to continue
        tries = state.retries.get("RESUME_CHOICE", 0)
        if tries >= MAX_RETRIES_PER_STAGE:
            state.stage = "FLOW"
            return respond(state, "I’ll continue your pending application. If that’s not right, just say “stop” anytime.")
        state.retries["RESUME_CHOICE"] = tries + 1
        state.listen_timeout_sec = DEFAULT_LISTEN
        return respond(state, _resume_prompt_for_try(tries))

    # ------------------- FLOW -------------------
    if stage == "FLOW":
        # Handle pending confirmations without clearing on silence
        if state.awaiting_confirm_field:
            f = state.awaiting_confirm_field
            if not utter:
                # repeat same confirmation; keep waiting
                if f == "full_name" and state.full_name:
                    return respond(state, PROMPTS["CONFIRM_NAME"].format(
                        name=state.full_name, spelled=spell_for_name(state.full_name)))
                if f == "phone" and (state.best_phone or state.phone):
                    return respond(state, PROMPTS["CONFIRM_PHONE"].format(phone=state.best_phone or state.phone))
                if f == "email" and state.email:
                    return respond(state, PROMPTS["CONFIRM_EMAIL_SPELL"].format(
                        email=state.email, spelled=spell_for_email(state.email)))
                if f == "address" and (state.address_norm or state.address):
                    return respond(state, PROMPTS["CONFIRM_ADDRESS"].format(
                        address=state.address_norm or state.address))
                if f == "attorney":
                    return respond(state, PROMPTS["CONFIRM_ATTORNEY"].format(summary=_attorney_summary(state)))
                if f == "injury_details" and state.injury_details:
                    return respond(state, PROMPTS["CONFIRM_INJURY_DETAILS"].format(details=state.injury_details))
                return respond(state, "Please say yes or no.")
            yn = yes_no(utter)
            if yn is True:
                state.awaiting_confirm_field = None
            elif yn is False:
                field = state.awaiting_confirm_field
                state.awaiting_confirm_field = None
                if field == "full_name":
                    state.listen_timeout_sec = LONG_LISTEN
                    return respond(state, "Okay—let’s try again. " + _name_prompt_for_try(1))
                if field == "email":
                    return respond(state, PROMPTS["EMAIL_SPELL_PROMPT"])
                if field == "phone":
                    return respond(state, PROMPTS["ASK_PHONE"])
                if field == "address":
                    state.listen_timeout_sec = LONG_LISTEN
                    return respond(state, PROMPTS["ASK_ADDRESS"])
                if field == "attorney":
                    state.listen_timeout_sec = LONG_LISTEN
                    return respond(state, PROMPTS["ASK_ATTORNEY_INFO"])
                if field == "injury_details":
                    state.listen_timeout_sec = LONG_LISTEN
                    return respond(state, PROMPTS["ASK_INJURY_DETAILS"])
                return respond(state, "Okay, let’s update that.")
            else:
                # ambiguous — re-ask same confirm
                if f == "full_name" and state.full_name:
                    return respond(state, PROMPTS["CONFIRM_NAME"].format(
                        name=state.full_name, spelled=spell_for_name(state.full_name)))
                if f == "email" and state.email:
                    return respond(state, PROMPTS["CONFIRM_EMAIL_SPELL"].format(
                        email=state.email, spelled=spell_for_email(state.email)))
                if f == "phone" and (state.best_phone or state.phone):
                    return respond(state, PROMPTS["CONFIRM_PHONE"].format(phone=state.best_phone or state.phone))
                if f == "address" and (state.address_norm or state.address):
                    return respond(state, PROMPTS["CONFIRM_ADDRESS"].format(
                        address=state.address_norm or state.address))
                if f == "attorney":
                    return respond(state, PROMPTS["CONFIRM_ATTORNEY"].format(summary=_attorney_summary(state)))
                if f == "injury_details" and state.injury_details:
                    return respond(state, PROMPTS["CONFIRM_INJURY_DETAILS"].format(details=state.injury_details))
                return respond(state, "Please say yes or no.")

        # Run LLM slot extraction for latest utterance (bias to next missing)
        wanted = []
        nxt = _next_missing(state)
        if nxt == "attorney":
            wanted = ["has_attorney", "attorney_name", "attorney_phone", "law_firm", "law_firm_address"]
        elif nxt == "case":
            wanted = ["injury_type"]
        elif nxt:
            wanted = [nxt]
        slots = extract_slots(utter, wanted_fields=wanted) if utter else {}

        # Apply slots (extract_slots already applies a confidence threshold)
        conf = slots.get("_confidence", {})
        def set_if(k: str, cur: Optional[str], newv: Optional[str]) -> Optional[str]:
            if not newv:
                return cur
            nv = clean_text(newv)
            if not cur or (cur != nv):
                state.confidences[k] = float(conf.get(k, 0.0) or 0.0)
                return nv
            return cur

        if "full_name" in slots: state.full_name = set_if("full_name", state.full_name, slots["full_name"])
        if "phone" in slots:
            state.phone = set_if("phone", state.phone, slots["phone"])
            state.best_phone = state.best_phone or state.phone
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

        # --- SPECIAL: name fallback if still missing but user said something like "my name is ..."
        if nxt == "full_name" and not state.full_name and utter:
            fb = _fallback_name_extract(utter)
            if fb:
                state.full_name = fb

        # Critical confirmations (low friction)
        if state.full_name and (state.awaiting_confirm_field is None) and state.confidences.get("full_name", 0) >= 0.50:
            state.awaiting_confirm_field = "full_name"
            return respond(state, PROMPTS["CONFIRM_NAME"].format(
                name=state.full_name, spelled=spell_for_name(state.full_name)
            ))

        if state.phone and (state.awaiting_confirm_field is None) and state.confidences.get("phone", 0) >= 0.65 and not state.best_phone:
            state.best_phone = state.phone
        if state.best_phone and (state.awaiting_confirm_field is None) and state.confidences.get("phone", 0) >= 0.65:
            state.awaiting_confirm_field = "phone"
            return respond(state, PROMPTS["CONFIRM_PHONE"].format(phone=state.best_phone))

        if state.email and (state.awaiting_confirm_field is None) and state.confidences.get("email", 0) >= 0.65:
            state.awaiting_confirm_field = "email"
            return respond(state, PROMPTS["CONFIRM_EMAIL_SPELL"].format(
                email=state.email, spelled=spell_for_email(state.email)
            ))

        if state.address and (state.awaiting_confirm_field is None) and state.confidences.get("address", 0) >= 0.65:
            verified, normalized = await verify_address(state.address)
            state.address_verified = bool(verified)
            state.address_norm = normalized or state.address
            if state.state:
                try:
                    hits = await kb_search(f"Does Oasis serve clients in {state.state}?", k=3)
                    text = " ".join([h.get("text", "") for h in hits]).lower()
                    if any(p in text for p in ["does not serve", "do not serve", "not available"]):
                        state.state_eligible = "no"
                    elif any(p in text for p in ["serve", "available", "support"]):
                        state.state_eligible = "yes"
                    else:
                        state.state_eligible = "unknown"
                except Exception:
                    state.state_eligible = state.state_eligible or "unknown"
            state.awaiting_confirm_field = "address"
            state.listen_timeout_sec = LONG_LISTEN
            return respond(state, PROMPTS["CONFIRM_ADDRESS"].format(address=state.address_norm or state.address))

        if state.has_attorney and (state.awaiting_confirm_field is None) and (
            state.attorney_name or state.attorney_phone or state.law_firm or state.law_firm_address
        ):
            state.attorney_verified = await verify_attorney(
                state.attorney_name, state.law_firm, state.attorney_phone, state.law_firm_address
            )
            state.awaiting_confirm_field = "attorney"
            return respond(state, PROMPTS["CONFIRM_ATTORNEY"].format(summary=_attorney_summary(state)))

        if state.injury_details and (state.awaiting_confirm_field is None) and state.confidences.get("injury_details", 0) >= 0.65:
            state.awaiting_confirm_field = "injury_details"
            state.listen_timeout_sec = LONG_LISTEN
            return respond(state, PROMPTS["CONFIRM_INJURY_DETAILS"].format(details=state.injury_details))

        # Ask the next missing field with progressive guidance
        nxt = _next_missing(state)
        if not nxt:
            state.stage = "SUMMARY"
            state.summary_read = False
            state.awaiting_confirmation = True
            return respond(state, "Looks like we have everything. I’ll read back your details.")

        if nxt == "full_name":
            tries = state.retries.get("ASK_NAME", 0)
            state.retries["ASK_NAME"] = tries + 1
            state.listen_timeout_sec = LONG_LISTEN
            prompt = _name_prompt_for_try(tries)
            if tries >= MAX_RETRIES_PER_STAGE:
                # after multiple misses, offer human
                return respond(state, prompt + " If you’d prefer a human, say “agent”.")
            return respond(state, prompt)

        ask = {
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

    # ------------------- SUMMARY / CORRECTION / Q&A -------------------

    if stage == "SUMMARY":
        t = (utter or "").lower()
        if any(w in t for w in ["human", "agent", "representative", "someone", "live person", "specialist"]):
            return respond(state, PROMPTS["HANDOFF"], handoff=True)

        if not state.summary_read:
            state.summary_read = True
            state.awaiting_confirmation = True
            return respond(state, _summarize(state))

        if state.awaiting_confirmation:
            yn = yes_no(utter)
            if yn is True:
                state.completed = True
                state.stage = "QNA_OFFER"
                return respond(state, PROMPTS["QNA_OFFER"])
            if yn is False:
                state.correction_mode = True
                state.correction_target = None
                state.summary_read = False
                state.awaiting_confirmation = False
                state.stage = "CORRECT_SELECT"
                return respond(state, PROMPTS["CORRECT_SELECT"])
            # re-ask once or twice, then proceed
            tries = state.retries.get("SUMMARY_CONFIRM", 0)
            if tries >= 2:
                state.completed = True
                state.stage = "QNA_OFFER"
                return respond(state, PROMPTS["QNA_OFFER"])
            state.retries["SUMMARY_CONFIRM"] = tries + 1
            return respond(state, PROMPTS["SUMMARY_CONFIRM"])

    if stage == "CORRECT_SELECT":
        target_text = (utter or "").lower()
        mapping = {
            "name": "full_name",
            "phone": "phone",
            "number": "phone",
            "email": "email",
            "address": "address",
            "attorney": "attorney",
            "case": "case",
            "incident": "incident_date",
            "funding type": "funding_type",
            "amount": "funding_amount",
        }
        chosen = None
        for k, v in mapping.items():
            if k in target_text:
                chosen = v
                break
        if not chosen:
            tries = state.retries.get("CORRECT_SELECT", 0)
            if tries >= 2:
                # fallback: move to Q&A/Done
                state.stage = "QNA_OFFER"
                return respond(state, PROMPTS["QNA_OFFER"])
            state.retries["CORRECT_SELECT"] = tries + 1
            return respond(state, PROMPTS["CORRECT_SELECT"])

        state.stage = "FLOW"
        prompts = {
            "full_name": _name_prompt_for_try(1),
            "phone": PROMPTS["ASK_PHONE"],
            "email": PROMPTS["ASK_EMAIL"],
            "address": PROMPTS["ASK_ADDRESS"],
            "attorney": PROMPTS["ASK_ATTORNEY_INFO"] if state.has_attorney else PROMPTS["ASK_ATTORNEY_YN"],
            "case": PROMPTS["ASK_CASE_TYPE"],
            "incident_date": PROMPTS["ASK_INCIDENT_DATE"],
            "funding_type": PROMPTS["ASK_FUNDING_TYPE"],
            "funding_amount": PROMPTS["ASK_FUNDING_AMOUNT"],
        }
        if chosen in ("address", "attorney"): state.listen_timeout_sec = LONG_LISTEN
        return respond(state, PROMPTS["CORRECT_ACK"] + " " + prompts.get(chosen, PROMPTS["ASK_NAME"]))

    if stage == "QNA_OFFER":
        yn = yes_no(utter)
        if yn is False:
            state.stage = "DONE"
            state.completed = True
            return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        state.stage = "QNA_ASK"
        state.listen_timeout_sec = LONG_LISTEN
        return respond(state, PROMPTS["QNA_PROMPT"])

    if stage == "QNA_ASK":
        state.listen_timeout_sec = LONG_LISTEN
        if utter:
            hits = await kb_search(utter, k=3)
            answer = ""
            if hits:
                snippets = []
                for h in hits[:3]:
                    txt = (h.get("text", "") or "").strip().replace("\n", " ")
                    if txt:
                        snippets.append(txt[:240])
                answer = " ".join(snippets)[:600]
            if not answer:
                answer = "Here’s what I can share: a specialist will review your case specifics and provide the most accurate guidance shortly."
            state.qna_remaining = max(0, (state.qna_remaining or 1) - 1)
            if state.qna_remaining > 0:
                state.stage = "QNA_ASK"
                return respond(state, answer + " " + PROMPTS["QNA_FOLLOWUP"])
            state.stage = "DONE"
            state.completed = True
            return respond(state, answer + " " + PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)
        # no question — wrap it up
        state.stage = "DONE"
        state.completed = True
        return respond(state, PROMPTS["QNA_WRAP"] + " " + PROMPTS["DONE"], completed=True)

    if stage == "DONE":
        state.completed = True
        return respond(state, PROMPTS["DONE"], completed=True)

    # Fallback
    return respond(state, "Could you say that again?")
