
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

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
    stage: str = "CONSENT"
    slots: Slots = Slots()
    completed: bool = False
    handoff: bool = False
