# app/config.py
import os

ORCH_API_KEY = os.environ.get("ORCH_API_KEY", "")

# timeouts (seconds)
DEFAULT_LISTEN = int(os.environ.get("DEFAULT_LISTEN", "7"))
LONG_LISTEN = int(os.environ.get("LONG_LISTEN", "15"))

# OpenAI (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # soft assist only

# KB retrieval (optional)
KB_URL = os.environ.get("KB_URL", "")
KB_API_KEY = os.environ.get("KB_API_KEY", "")

# Google verification (optional)
GOOGLE_MAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Logger
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
