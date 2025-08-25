from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import time

class OrchestrateRequest(BaseModel):
    session_id: str
    last_user_utterance: Optional[str] = ""
    caller_number: Optional[str] = None

class OrchestrateResponse(BaseModel):
    updates: dict
    next_prompt: str
    completed: bool = False
    handoff: bool = False
    citations: List[int] = []
    confidence: float = 0.7
    listen_timeout_sec: int = 7

class SessionState(BaseModel):
    session_id: str
    stage: str = "ENTRY"
    completed: bool = False

    # Caller
    caller_number: Optional[str] = None

    # Contact
    full_name: Optional[str] = None
    phone: Optional[str] = None
    best_phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    address_norm: Optional[str] = None
    address_verified: bool = False
    state: Optional[str] = None
    state_eligible: Optional[str] = None
    state_eligibility_note: Optional[str] = None

    # Attorney
    has_attorney: Optional[bool] = None
    attorney_name: Optional[str] = None
    attorney_phone: Optional[str] = None
    law_firm: Optional[str] = None
    law_firm_address: Optional[str] = None
    attorney_verified: bool = False

    # Case
    injury_type: Optional[str] = None
    injury_details: Optional[str] = None
    incident_date: Optional[str] = None

    # Funding
    funding_type: Optional[str] = None
    funding_amount: Optional[str] = None

    # LLM confidences (last seen)
    confidences: Dict[str, float] = Field(default_factory=dict)

    # Flow control
    retries: Dict[str, int] = Field(default_factory=dict)
    last_prompt: Optional[str] = None
    summary_read: bool = False
    awaiting_confirmation: bool = False
    correction_mode: bool = False
    correction_target: Optional[str] = None
    awaiting_confirm_field: Optional[str] = None
    updated_at: float = Field(default_factory=lambda: time.time())

    # Listen hints and Q&A
    listen_timeout_sec: int = 7
    qna_offered: bool = False
    qna_remaining: int = 2
    extra: Dict[str, str] = Field(default_factory=dict)

    def to_updates(self) -> dict:
        return {
            "full_name": self.full_name,
            "phone": self.phone,
            "best_phone": self.best_phone or self.phone,
            "email": self.email,
            "address": self.address,
            "address_norm": self.address_norm,
            "address_verified": self.address_verified,
            "state": self.state,
            "state_eligible": self.state_eligible,
            "state_eligibility_note": self.state_eligibility_note,
            "has_attorney": self.has_attorney,
            "attorney_name": self.attorney_name,
            "attorney_phone": self.attorney_phone,
            "law_firm": self.law_firm,
            "law_firm_address": self.law_firm_address,
            "attorney_verified": self.attorney_verified,
            "injury_type": self.injury_type,
            "injury_details": self.injury_details,
            "incident_date": self.incident_date,
            "funding_type": self.funding_type,
            "funding_amount": self.funding_amount,
        }
