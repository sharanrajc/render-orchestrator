import os
import time
from typing import Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import OrchestrateRequest, OrchestrateResponse, SessionState
from .prompts import PROMPTS
from .utils import normalize_phone, extract_amount_usd, yes_no, first_name
from .kb_client import kb_search

API_KEY = os.environ.get("ORCH_API_KEY", "")

# In-memory state (PoC). For prod, back with a DB.
SESSIONS: Dict[str, SessionState] = {}
TRANSCRIPTS: Dict[str, List[dict]] = {}

MAX_RETRIES_PER_STAGE = 2  # debounce repeated questions
TRANSCRIPT_MAX_TURNS = 200


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


def next_prompt_text(state: SessionState, text: str) -> str:
    """Avoid repeating the exact same prompt; prepend a gentle reprompt if needed."""
    if state.last_prompt and text.strip() == (state.last_prompt or "").strip():
        return PROMPTS["REPROMPT_SHORT"] + text
    return text


def bump_retry(state: SessionState, key: str) -> bool:
    """Increment and return True if we exceeded the retry limit for this stage/key."""
    n = state.retries.get(key, 0) + 1
    state.retries[key] = n
    return n > MAX_RETRIES_PER_STAGE


app = FastAPI(title="Orchestrator", version="1.0.0")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


def respond(
    state: SessionState,
    text: str,
    completed: bool = False,
    handoff: bool = False,
    citations: Optional[List[int]] = None,
    confidence: float = 0.7,
) -> OrchestrateResponse:
    citations = citations or []
    # Log AI turn to transcript
    _append_turn(
        state.session_id,
        "ai",
        text,
        state.stage,
        {"completed": completed, "handoff": handoff, "citations": citations},
    )
    # Persist state
    SESSIONS[state.session_id] = state
    state.last_prompt = text
    return OrchestrateResponse(
        updates=state.to_updates(),
        next_prompt=text[:180],  # keep TTS snappy
        completed=completed,
        handoff=handoff,
        citations=citations,
        confidence=confidence,
    )


@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest, x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    # Create or load session
    state = SESSIONS.get(req.session_id) or SessionState(session_id=req.session_id)
    utter = (req.last_user_utterance or "").strip()
    stage = state.stage

    # Log user turn
    if utter:
        _append_turn(req.session_id, "user", utter, stage)

    # === FLOW ===

    # 1) Persona greeting
    if stage == "GREETING":
        state.stage = "ASK_NAME"
        np = next_prompt_text(state, PROMPTS["GREETING"])
        return respond(state, np)

    # 2) Full name
    if stage == "ASK_NAME":
        if utter:
            state.full_name = utter
            state.stage = "ASK_PHONE"
            np = next_prompt_text(state, PROMPTS["ASK_PHONE"].format(first_name=first_name(state.full_name)))
            return respond(state, np)
        if bump_retry(state, "ASK_NAME"):
            state.stage = "ASK_PHONE"
            np = next_prompt_text(state, PROMPTS["ASK_PHONE"].format(first_name="there"))
            return respond(state, np)
        np = next_prompt_text(state, "Please tell me your full name.")
        return respond(state, np)

    # 3) Phone (best phone = phone by default)
    if stage == "ASK_PHONE":
        ph = normalize_phone(utter)
        if ph:
            state.phone = ph
            state.best_phone = ph
            state.stage = "ASK_EMAIL"
            np = next_prompt_text(state, PROMPTS["ASK_EMAIL"])
            return respond(state, np)
        if bump_retry(state, "ASK_PHONE"):
            state.stage = "ASK_EMAIL"
            np = next_prompt_text(state, PROMPTS["ASK_EMAIL"])
            return respond(state, np)
        np = next_prompt_text(state, "Please say your 10-digit phone number.")
        return respond(state, np)

    # 4) Email
    if stage == "ASK_EMAIL":
        if utter:
            state.email = utter
            state.stage = "ASK_ADDRESS"
            np = next_prompt_text(state, PROMPTS["ASK_ADDRESS"])
            return respond(state, np)
        if bump_retry(state, "ASK_EMAIL"):
            state.stage = "ASK_ADDRESS"
            np = next_prompt_text(state, PROMPTS["ASK_ADDRESS"])
            return respond(state, np)
        np = next_prompt_text(state, "What’s the best email address?")
        return respond(state, np)

    # 5) Address
    if stage == "ASK_ADDRESS":
        if utter:
            state.address = utter
            state.stage = "ASK_ATTORNEY_YN"
            np = next_prompt_text(state, PROMPTS["ASK_ATTORNEY_YN"])
            return respond(state, np)
        if bump_retry(state, "ASK_ADDRESS"):
            state.stage = "ASK_ATTORNEY_YN"
            np = next_prompt_text(state, PROMPTS["ASK_ATTORNEY_YN"])
            return respond(state, np)
        np = next_prompt_text(state, PROMPTS["ASK_ADDRESS"])
        return respond(state, np)

    # 6) Attorney yes/no
    if stage == "ASK_ATTORNEY_YN":
        yn = yes_no(utter)
        if yn is not None:
            state.has_attorney = yn
            state.stage = "ASK_ATTORNEY_INFO" if yn else "ASK_INJURY_TYPE"
            np = next_prompt_text(
                state, PROMPTS["ASK_ATTORNEY_INFO"] if yn else PROMPTS["ASK_INJURY_TYPE"]
            )
            return respond(state, np)
        if bump_retry(state, "ASK_ATTORNEY_YN"):
            state.has_attorney = False
            state.stage = "ASK_INJURY_TYPE"
            np = next_prompt_text(state, PROMPTS["ASK_INJURY_TYPE"])
            return respond(state, np)
        np = next_prompt_text(state, PROMPTS["ASK_ATTORNEY_YN"])
        return respond(state, np)

    # 7) Attorney info (free form; best-effort parse)
    if stage == "ASK_ATTORNEY_INFO":
        if utter:
            state.attorney_phone = normalize_phone(utter)
            if not state.attorney_name:
                state.attorney_name = utter
            if not state.law_firm:
                state.law_firm = utter
            state.stage = "ASK_INJURY_TYPE"
            np = next_prompt_text(state, PROMPTS["ASK_INJURY_TYPE"])
            return respond(state, np)
        if bump_retry(state, "ASK_ATTORNEY_INFO"):
            state.stage = "ASK_INJURY_TYPE"
            np = next_prompt_text(state, PROMPTS["ASK_INJURY_TYPE"])
            return respond(state, np)
        np = next_prompt_text(state, PROMPTS["ASK_ATTORNEY_INFO"])
        return respond(state, np)

    # 8) Injury type
    if stage == "ASK_INJURY_TYPE":
        if utter:
            t = utter.lower()
            if "auto" in t or "car" in t:
                state.injury_type = "auto accident"
            elif "slip" in t or "fall" in t:
                state.injury_type = "slip and fall"
            elif "dog" in t or "bite" in t:
                state.injury_type = "dog bite"
            else:
                state.injury_type = utter
            state.stage = "ASK_INJURY_DETAILS"
            np = next_prompt_text(state, PROMPTS["ASK_INJURY_DETAILS"])
            return respond(state, np)
        if bump_retry(state, "ASK_INJURY_TYPE"):
            state.stage = "ASK_INJURY_DETAILS"
            np = next_prompt_text(state, PROMPTS["ASK_INJURY_DETAILS"])
            return respond(state, np)
        np = next_prompt_text(state, PROMPTS["ASK_INJURY_TYPE"])
        return respond(state, np)

    # 9) Injury details (empathetic)
    if stage == "ASK_INJURY_DETAILS":
        if utter:
            state.injury_details = utter
            state.stage = "ASK_FUNDING"
            np = next_prompt_text(state, PROMPTS["ASK_FUNDING"])
            return respond(state, np)
        if bump_retry(state, "ASK_INJURY_DETAILS"):
            state.stage = "ASK_FUNDING"
            np = next_prompt_text(state, PROMPTS["ASK_FUNDING"])
            return respond(state, np)
        np = next_prompt_text(state, "I’m sorry this happened. Briefly, what occurred and when?")
        return respond(state, np)

    # 10) Funding amount
    if stage == "ASK_FUNDING":
        amt = extract_amount_usd(utter)
        if amt:
            state.funding_amount = amt
            state.stage = "SUMMARY"
        else:
            if bump_retry(state, "ASK_FUNDING"):
                state.stage = "SUMMARY"
            else:
                np = next_prompt_text(state, PROMPTS["ASK_FUNDING"])
                return respond(state, np)

    # 11) Readback summary + confirmation
    if stage == "SUMMARY":
        if not state.summary_read:
            parts = []
            if state.full_name:
                parts.append(f"Name: {state.full_name}")
            if state.best_phone or state.phone:
                parts.append(f"Phone: {state.best_phone or state.phone}")
            if state.email:
                parts.append(f"Email: {state.email}")
            if state.address:
                parts.append(f"Address: {state.address}")
            if state.has_attorney is not None:
                if state.has_attorney:
                    att = []
                    if state.attorney_name:
                        att.append(state.attorney_name)
                    if state.law_firm:
                        att.append(state.law_firm)
                    if state.attorney_phone:
                        att.append(state.attorney_phone)
                    parts.append("Attorney: " + ", ".join(att) if att else "Attorney: provided")
                else:
                    parts.append("Attorney: none")
            if state.injury_type:
                parts.append(f"Case type: {state.injury_type}")
            if state.injury_details:
                parts.append(f"Details: {state.injury_details}")
            if state.funding_amount:
                parts.append(f"Funding requested: {state.funding_amount}")

            summary_text = (
                PROMPTS["SUMMARY_INTRO"]
                + " "
                + ". ".join(parts)
                + ". "
                + PROMPTS["SUMMARY_CONFIRM"]
            )

            state.summary_read = True
            state.awaiting_confirmation = True
            np = next_prompt_text(state, summary_text)
            return respond(state, np)

        if state.awaiting_confirmation:
            yn = yes_no(utter)
            if yn is True:
                state.completed = True
                state.stage = "DONE"
                np = next_prompt_text(state, PROMPTS["DONE"])
                return respond(state, np, completed=True)
            if yn is False:
                # Simple correction path: start over from name (could be more granular)
                state.stage = "ASK_NAME"
                np = next_prompt_text(state, "No problem. Let’s fix it. What’s your full name?")
                return respond(state, np)
            if bump_retry(state, "SUMMARY_CONFIRM"):
                state.completed = True
                state.stage = "DONE"
                np = next_prompt_text(state, PROMPTS["DONE"])
                return respond(state, np, completed=True)
            np = next_prompt_text(state, PROMPTS["SUMMARY_CONFIRM"])
            return respond(state, np)

    # 12) Done
    if stage == "DONE":
        state.completed = True
        np = next_prompt_text(state, PROMPTS["DONE"])
        return respond(state, np, completed=True)

    # Fallback
    np = next_prompt_text(state, "Could you say that again?")
    return respond(state, np)


# ===== Transcript Endpoints =====

@app.get("/transcript/{session_id}")
def get_transcript(session_id: str, redact: bool = True):
    turns = TRANSCRIPTS.get(session_id, [])
    if not redact:
        return {"session_id": session_id, "turns": turns}

    # Minimal redaction for PoC
    import re

    def _redact(text: str) -> str:
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[redacted-email]", text)
        text = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted-phone]", text)
        return text

    red = []
    for t in turns:
        t2 = dict(t)
        t2["text"] = _redact(t2.get("text", ""))
        red.append(t2)
    return {"session_id": session_id, "turns": red}


@app.get("/transcripts")
def list_transcripts():
    return {"sessions": [{"session_id": sid, "turns": len(ts)} for sid, ts in TRANSCRIPTS.items()]}


@app.post("/reset")
def reset_session(req: OrchestrateRequest, x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    SESSIONS.pop(req.session_id, None)
    TRANSCRIPTS.pop(req.session_id, None)
    return {"ok": True}
