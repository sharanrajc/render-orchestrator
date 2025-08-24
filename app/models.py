
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time

class OrchestrateRequest(BaseModel):
    session_id: str
    last_user_utterance: str = ""
    metadata: Optional[Dict[str, Any]] = None

class OrchestrateResponse(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict)
    next_prompt: str = "Could you please repeat that?"
    completed: bool = False
    handoff: bool = False
    citations: List[int] = Field(default_factory=list)
    confidence: float = 0.7

class Slots(BaseModel):
    identity: Dict[str, Any] = Field(default_factory=dict)
    contact:  Dict[str, Any] = Field(default_factory=dict)
    case:     Dict[str, Any] = Field(default_factory=dict)
    attorney: Dict[str, Any] = Field(default_factory=dict)
    injury:   Dict[str, Any] = Field(default_factory=dict)
    funding:  Dict[str, Any] = Field(default_factory=dict)

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
    injury_type: Optional[str] = None      # auto accident, dog bite, slip & fall, etc.
    injury_details: Optional[str] = None   # short, empathetic capture
    funding_amount: Optional[str] = None   # USD string

    # Flow control
    retries: Dict[str, int] = Field(default_factory=dict)
    last_prompt: Optional[str] = None
    summary_read: bool = False             # have we read back the summary?
    awaiting_confirmation: bool = False    # waiting for “yes/no” after summary
    updated_at: float = Field(default_factory=lambda: time.time())
