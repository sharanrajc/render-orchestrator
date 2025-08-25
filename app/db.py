from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    status = Column(String, default="pending")
    caller_number = Column(String, index=True)

    full_name = Column(String)
    phone = Column(String)
    best_phone = Column(String)
    email = Column(String)
    address = Column(String)
    address_norm = Column(String)
    address_verified = Column(Boolean, default=False)
    address_skipped = Column(Boolean, nullable=True, default=False)
    state = Column(String)
    state_eligible = Column(String)
    state_eligibility_note = Column(String)

    has_attorney = Column(Boolean)
    attorney_name = Column(String)
    attorney_phone = Column(String)
    law_firm = Column(String)
    law_firm_address = Column(String)
    attorney_verified = Column(Boolean, default=False)

    injury_type = Column(String)
    injury_details = Column(String)
    incident_date = Column(String)

    funding_type = Column(String)
    funding_amount = Column(String)

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

def init_db():
    Base.metadata.create_all(bind=engine)

def upsert_application_from_state(db, state):
    row = db.query(Application).filter(Application.session_id == state.session_id).first()
    if not row:
        row = Application(session_id=state.session_id)
        db.add(row)
    row.status = "completed" if state.completed else "pending"
    row.caller_number = state.caller_number
    row.full_name = state.full_name
    row.phone = state.phone
    row.best_phone = state.best_phone or state.phone
    row.email = state.email
    row.address = state.address
    row.address_norm = state.address_norm
    row.address_verified = state.address_verified
    row.address_skipped = state.address_skipped
    row.state = state.state
    row.state_eligible = state.state_eligible
    row.state_eligibility_note = state.state_eligibility_note

    row.has_attorney = state.has_attorney
    row.attorney_name = state.attorney_name
    row.attorney_phone = state.attorney_phone
    row.law_firm = state.law_firm
    row.law_firm_address = state.law_firm_address
    row.attorney_verified = state.attorney_verified

    row.injury_type = state.injury_type
    row.injury_details = state.injury_details
    row.incident_date = state.incident_date

    row.funding_type = state.funding_type
    row.funding_amount = state.funding_amount
    return row

def get_latest_by_caller(db, phone: str):
    return db.query(Application).filter(Application.best_phone == phone).order_by(Application.updated_at.desc()).first()
