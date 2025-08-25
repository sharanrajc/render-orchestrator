# app/db.py
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Text, Boolean, func
)
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

# Engine & Session
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String, index=True, nullable=False)
    status = Column(String, default="pending", nullable=False)  # "pending" | "completed"

    # who called
    caller_number = Column(String, nullable=True)

    # identity
    full_name = Column(String, nullable=True)
    best_phone = Column(String, nullable=True)
    email = Column(String, nullable=True)

    # address & state
    address = Column(Text, nullable=True)
    address_norm = Column(Text, nullable=True)
    address_verified = Column(Boolean, default=False)
    address_skipped = Column(Boolean, default=False)

    state = Column(String, nullable=True)
    state_eligible = Column(String, nullable=True)           # "yes" | "no" | "unknown"
    state_eligibility_note = Column(Text, nullable=True)     # explanatory note

    # attorney
    has_attorney = Column(Boolean, nullable=True)
    attorney_name = Column(String, nullable=True)
    attorney_phone = Column(String, nullable=True)
    law_firm = Column(String, nullable=True)
    law_firm_address = Column(Text, nullable=True)
    attorney_verified = Column(Boolean, nullable=True)

    # case
    injury_type = Column(String, nullable=True)
    injury_details = Column(Text, nullable=True)
    incident_date = Column(String, nullable=True)  # ISO date string

    # funding
    funding_type = Column(String, nullable=True)   # "fresh" | "extend"
    funding_amount = Column(String, nullable=True) # e.g., "$2,000"

    # audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), default=func.now())


def init_db():
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


# -------------------- Helpers --------------------

def get_latest_by_caller(db, caller_number: str) -> Optional[Application]:
    if not caller_number:
        return None
    return (
        db.query(Application)
        .filter(Application.caller_number == caller_number)
        .order_by(Application.updated_at.desc().nullslast(), Application.id.desc())
        .first()
    )


def upsert_application_from_state(db, state) -> Application:
    """
    Persist a SessionState into the applications table.
    Uses getattr(...) for forward compatibility, so new state fields won't crash older code.
    """
    row = (
        db.query(Application)
        .filter(Application.session_id == state.session_id)
        .one_or_none()
    )
    if row is None:
        row = Application(session_id=state.session_id)
        db.add(row)

    # status
    row.status = "completed" if getattr(state, "completed", False) else "pending"

    # caller
    row.caller_number = getattr(state, "caller_number", row.caller_number)

    # identity
    row.full_name = getattr(state, "full_name", row.full_name)
    # prefer best_phone; fallback to phone
    bp = getattr(state, "best_phone", None) or getattr(state, "phone", None)
    row.best_phone = bp or row.best_phone
    row.email = getattr(state, "email", row.email)

    # address & state
    row.address = getattr(state, "address", row.address)
    row.address_norm = getattr(state, "address_norm", row.address_norm)
    row.address_verified = bool(getattr(state, "address_verified", row.address_verified or False))
    row.address_skipped = bool(getattr(state, "address_skipped", row.address_skipped or False))

    row.state = getattr(state, "state", row.state)
    row.state_eligible = getattr(state, "state_eligible", row.state_eligible)
    row.state_eligibility_note = getattr(state, "state_eligibility_note", row.state_eligibility_note)

    # attorney
    row.has_attorney = getattr(state, "has_attorney", row.has_attorney)
    row.attorney_name = getattr(state, "attorney_name", row.attorney_name)
    row.attorney_phone = getattr(state, "attorney_phone", row.attorney_phone)
    row.law_firm = getattr(state, "law_firm", row.law_firm)
    row.law_firm_address = getattr(state, "law_firm_address", row.law_firm_address)
    row.attorney_verified = getattr(state, "attorney_verified", row.attorney_verified)

    # case
    row.injury_type = getattr(state, "injury_type", row.injury_type)
    row.injury_details = getattr(state, "injury_details", row.injury_details)
    row.incident_date = getattr(state, "incident_date", row.incident_date)

    # funding
    row.funding_type = getattr(state, "funding_type", row.funding_type)
    row.funding_amount = getattr(state, "funding_amount", row.funding_amount)

    # touch updated_at (SQLAlchemy onupdate triggers, but be explicit)
    row.updated_at = datetime.utcnow()

    return row
