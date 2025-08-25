import os

# Security (optional)
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "")

# Postgres
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")

# KB retrieval (optional; safe to leave unset)
KB_URL = os.getenv("KB_URL", "")
KB_API_KEY = os.getenv("KB_API_KEY", "")

# OpenAI slot extraction
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_SLOT_MODEL = os.getenv("OPENAI_SLOT_MODEL", "gpt-4o-mini")

# Tunables
DEFAULT_LISTEN = int(os.getenv("DEFAULT_LISTEN", "7"))
LONG_LISTEN = int(os.getenv("LONG_LISTEN", "15"))
SLOT_CONFIDENCE_MIN = float(os.getenv("SLOT_CONFIDENCE_MIN", "0.65"))
