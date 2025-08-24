from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL, DB_ECHO

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required (e.g., postgresql+psycopg2://user:pass@host:5432/dbname)")

engine = create_engine(DATABASE_URL, echo=DB_ECHO, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, future=True, expire_on_commit=False)
Base = declarative_base()

class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True, nullable=False)
    status = Column(String(16), default="pending", index=True)  # pending|completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Caller
    caller_number = Column(String(32), index=True, nullable=True)

    # Contact
    full_name = Column(String(256))
    phone = Column(String(32))
    best_phone = Column(String(32))
    email = Column(String(256))
    address = Column(Text)
    address_norm = Column(Text)
    address_verified = Column(Boolean, default=False)
    state = Column(String(2))
    state_eligible = Column(String(32))  # yes|no|unknown
    state_eligibility_note = Column(Text)

    # Attorney
    has_attorney = Column(Boolean)
    attorney_name = Column(String(256))
    attorney_phone = Column(String(32))
    law_firm = Column(String(256))
    law_firm_address = Column(Text)
    attorney_verified = Column(Boolean, default=False)

    # Case
    injury_type = Column(String(64))
    injury_details = Column(Text)
    incident_date = Column(String(16))   # ISO yyyy-mm-dd

    # Funding
    funding_type = Column(String(16))    # fresh|topup
    funding_amount = Column(String(32))  # "$5000"

def init_db():
    Base.metadata.create_all(bind=engine)

def upsert_application_from_state(db, state) -> Application:
    app = db.query(Application).filter_by(session_id=state.session_id).one_or_none()
    if app is None:
        app = Application(session_id=state.session_id)
        db.add(app)

    app.caller_number = getattr(state, "caller_number", None)

    app.full_name = state.full_name
    app.phone = state.phone
    app.best_phone = state.best_phone or state.phone
    app.email = state.email
    app.address = state.address
    app.address_norm = getattr(state, "address_norm", None)
    app.address_verified = bool(getattr(state, "address_verified", False))
    app.state = getattr(state, "state", None)
    app.state_eligible = getattr(state, "state_eligible", None)
    app.state_eligibility_note = getattr(state, "state_eligibility_note", None)

    app.has_attorney = state.has_attorney
    app.attorney_name = state.attorney_name
    app.attorney_phone = state.attorney_phone
    app.law_firm = state.law_firm
    app.law_firm_address = getattr(state, "law_firm_address", None)
    app.attorney_verified = bool(getattr(state, "attorney_verified", False))

    app.injury_type = state.injury_type
    app.injury_details = state.injury_details
    app.incident_date = getattr(state, "incident_date", None)

    app.funding_type = getattr(state, "funding_type", None)
    app.funding_amount = state.funding_amount

    app.status = "completed" if state.completed else "pending"
    return app

def get_latest_by_caller(db, caller_number: str) -> Optional[Application]:
    if not caller_number:
        return None
    return (
        db.query(Application)
        .filter(Application.caller_number == caller_number)
        .order_by(Application.updated_at.desc())
        .first()
    )
