from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import time

class OrchestrateRequest(BaseModel):
    session_id: str
    last_user_utterance: Optional[str] = ""

class OrchestrateResponse(BaseModel):
    updates: dict
    next_prompt: str
    completed: bool = False
    handoff: bool = False
    citations: List[int] = []
    confidence: float = 0.7

class SessionState(BaseModel):
    session_id: str
    stage: str = "GREETING"
    completed: bool = False

    # Contact
    full_name: Optional[str] = None
    phone: Optional[str] = None
    best_phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None

    # Attorney
    has_attorney: Optional[bool] = None
    attorney_name: Optional[str] = None
    attorney_phone: Optional[str] = None
    law_firm: Optional[str] = None

    # Injury & funding
    injury_type: Optional[str] = None
    injury_details: Optional[str] = None
    funding_amount: Optional[str] = None

    # Flow control
    retries: Dict[str, int] = Field(default_factory=dict)
    last_prompt: Optional[str] = None
    summary_read: bool = False
    awaiting_confirmation: bool = False
    updated_at: float = Field(default_factory=lambda: time.time())

    def to_updates(self) -> dict:
        return {
            "full_name": self.full_name,
            "phone": self.phone,
            "best_phone": self.best_phone or self.phone,
            "email": self.email,
            "address": self.address,
            "has_attorney": self.has_attorney,
            "attorney_name": self.attorney_name,
            "attorney_phone": self.attorney_phone,
            "law_firm": self.law_firm,
            "injury_type": self.injury_type,
            "injury_details": self.injury_details,
            "funding_amount": self.funding_amount,
        }
