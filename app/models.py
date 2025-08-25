# app/models.py (snippet)
from pydantic import BaseModel, Field
from typing import Optional, Dict

class SessionState(BaseModel):
    session_id: str
    stage: str = "ENTRY"
    last_prompt: Optional[str] = None

    # caller & core fields
    caller_number: Optional[str] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    best_phone: Optional[str] = None
    email: Optional[str] = None

    address: Optional[str] = None
    address_norm: Optional[str] = None
    address_verified: bool = False
    address_skipped: bool = False          # <-- NEW

    state: Optional[str] = None
    state_eligible: Optional[str] = None   # "yes"/"no"/"unknown"

    has_attorney: Optional[bool] = None
    attorney_name: Optional[str] = None
    attorney_phone: Optional[str] = None
    law_firm: Optional[str] = None
    law_firm_address: Optional[str] = None
    attorney_verified: Optional[bool] = None

    injury_type: Optional[str] = None
    injury_details: Optional[str] = None
    incident_date: Optional[str] = None
    funding_type: Optional[str] = None
    funding_amount: Optional[str] = None

    # runtime controls
    retries: Dict[str, int] = Field(default_factory=dict)
    confidences: Dict[str, float] = Field(default_factory=dict)
    _flags: Dict[str, bool] = Field(default_factory=dict)

    awaiting_confirm_field: Optional[str] = None
    summary_read: bool = False
    awaiting_confirmation: bool = False
    correction_mode: bool = False
    correction_target: Optional[str] = None
    qna_remaining: int = 1
    completed: bool = False

    listen_timeout_sec: int = 7

    def to_updates(self) -> Dict[str, str]:
        # (unchanged) returns dict of fields the Twilio client uses
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
