# app/models.py
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class SessionState(BaseModel):
    # Lifecycle / routing
    session_id: str
    stage: str = "ENTRY"
    last_prompt: Optional[str] = None

    # Caller & core identity
    caller_number: Optional[str] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    best_phone: Optional[str] = None
    email: Optional[str] = None

    # Address & state
    address: Optional[str] = None
    address_norm: Optional[str] = None
    address_verified: bool = False
    address_skipped: bool = False
    state: Optional[str] = None
    state_eligible: Optional[str] = None  # "yes" | "no" | "unknown"
    state_eligibility_note: str | None = None   # brief note like "Oasis serves in FL"

    # Attorney
    has_attorney: Optional[bool] = None
    attorney_name: Optional[str] = None
    attorney_phone: Optional[str] = None
    law_firm: Optional[str] = None
    law_firm_address: Optional[str] = None
    attorney_verified: Optional[bool] = None

    # Case
    injury_type: Optional[str] = None
    injury_details: Optional[str] = None
    incident_date: Optional[str] = None

    # Funding
    funding_type: Optional[str] = None    # "fresh" | "extend"
    funding_amount: Optional[str] = None  # normalized like "$2,000"

    # Orchestration metadata
    retries: Dict[str, int] = Field(default_factory=dict)
    confidences: Dict[str, float] = Field(default_factory=dict)
    flags: Dict[str, bool] = Field(default_factory=dict)  # <— renamed from _flags

    awaiting_confirm_field: Optional[str] = None
    summary_read: bool = False
    awaiting_confirmation: bool = False
    correction_mode: bool = False
    correction_target: Optional[str] = None
    qna_remaining: int = 1
    completed: bool = False

    # Telephony controls
    listen_timeout_sec: int = 7

    def to_updates(self) -> Dict[str, str]:
        """What the Twilio client/UI might want to keep in sync."""
        return {
            "stage": self.stage,
            "full_name": self.full_name or "",
            "best_phone": self.best_phone or self.phone or "",
            "email": self.email or "",
            "address": self.address_norm or self.address or "",
            "state": self.state or "",
            "injury_type": self.injury_type or "",
            "injury_details": self.injury_details or "",
            "incident_date": self.incident_date or "",
            "funding_type": self.funding_type or "",
            "funding_amount": self.funding_amount or "",
        }


class OrchestrateRequest(BaseModel):
    session_id: str
    caller_number: Optional[str] = None
    last_user_utterance: Optional[str] = None


class OrchestrateResponse(BaseModel):
    updates: Dict[str, str] = Field(default_factory=dict)
    next_prompt: str = ""
    completed: bool = False
    handoff: bool = False
    citations: List[int] = Field(default_factory=list)
    confidence: float = 0.8
    listen_timeout_sec: int = 7
