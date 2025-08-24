
import os, re
from fastapi import FastAPI, Header, HTTPException
from .models import OrchestrateRequest, OrchestrateResponse, SessionState
from .kb_client import kb_search, format_kb_context
from .llm import call_llm_mini

API_KEY = os.environ.get("ORCH_API_KEY","")
app = FastAPI(title="Agentic Orchestrator (PoC, mini-only)", version="1.0.0")

SESSIONS = {}
ORDER = ["CONSENT","IDENTITY","CONTACT","CASE","ATTORNEY","INJURY","FUNDING","SUMMARY"]
REQUIRED = {
    "IDENTITY": ["first","last"],
    "CONTACT": ["phone","email"],
    "CASE": ["type"],
    "ATTORNEY": ["has"],
    "INJURY": ["narrative"],
    "FUNDING": ["amount_usd"],
}
PROMPTS = {
    "CONSENT":  "This call may be recorded. I’m an automated intake assistant, not a lawyer. May I continue?",
    "IDENTITY": "What is your full legal name?",
    "CONTACT":  "Thanks. What’s the best phone number to reach you?",
    "CASE":     "Briefly, is this auto accident, slip and fall, dog bite, or other?",
    "ATTORNEY": "Do you already have an attorney representing you?",
    "INJURY":   "Could you describe your injuries in a few words?",
    "FUNDING":  "About how much funding are you looking for, in US dollars?",
    "SUMMARY":  "Thanks. I’ll read back what I captured to make sure it’s right. Ready?",
    "DONE":     "Thank you. That’s everything I needed. We’ll send a confirmation shortly."
}

def stage_filled(slots, stage):
    if stage not in REQUIRED: return True
    section = slots.get(stage.lower(), {}) or {}
    return all(section.get(k) not in (None, "", []) for k in REQUIRED[stage])

def advance_stage(current, slots):
    if not stage_filled(slots, current):
        return current
    idx = ORDER.index(current)
    for nxt in ORDER[idx+1:]:
        if not stage_filled(slots, nxt):
            return nxt
    return "SUMMARY"

def should_retrieve(stage: str, utter: str) -> bool:
    return bool(re.search(r"\b(what|how|why|explain|define|loan)\b", (utter or "").lower()))

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    state = SESSIONS.get(req.session_id) or SessionState(session_id=req.session_id)
    stage = state.stage
    utter = (req.last_user_utterance or "").strip()

    if stage == "CONTACT" and utter and any(ch.isdigit() for ch in utter) and "phone" not in state.slots.contact:
        state.slots.contact["phone"] = utter
    elif stage == "CONTACT" and "@" in utter and "email" not in state.slots.contact:
        state.slots.contact["email"] = utter

    kb_ctx = ""
    citations = []
    if should_retrieve(stage, utter):
        hits = await kb_search(utter, k=2)
        if hits:
            kb_ctx = format_kb_context(hits)
            citations = [1]

    allowed = REQUIRED.get(stage, [])
    llm_out = call_llm_mini(stage, utter, state.slots.model_dump(), kb_ctx, allowed)

    updates = llm_out.get("updates", {})
    if stage.lower() in updates and isinstance(updates[stage.lower()], dict):
        section = getattr(state.slots, stage.lower())
        section.update(updates[stage.lower()])

    if stage == "SUMMARY" or all(stage_filled(state.slots.model_dump(), s) for s in REQUIRED):
        state.completed = True

    next_stage = advance_stage(stage, state.slots.model_dump())
    if stage == "SUMMARY" and state.completed:
        next_prompt = PROMPTS["DONE"]
    elif next_stage != stage:
        state.stage = next_stage
        next_prompt = PROMPTS.get(next_stage, PROMPTS["SUMMARY"])
    else:
        next_prompt = llm_out.get("next_prompt") or PROMPTS.get(stage, PROMPTS["SUMMARY"])

    SESSIONS[req.session_id] = state

    return OrchestrateResponse(
        updates=state.slots.model_dump(),
        next_prompt=next_prompt[:180],
        completed=bool(state.completed and next_prompt == PROMPTS["DONE"]),
        handoff=bool(llm_out.get("handoff", False)),
        citations=citations,
        confidence=float(llm_out.get("confidence", 0.7)),
    )
