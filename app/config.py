import os

ORCH_API_KEY = os.environ.get("ORCH_API_KEY", "")

# ---- Database (Postgres) ----
# Example: postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Optional SQL echo logging
DB_ECHO = os.environ.get("DB_ECHO", "0") == "1"

# ---- KB service ----
KB_URL = os.environ.get("KB_URL", "")
KB_API_KEY = os.environ.get("KB_API_KEY", "")
KB_MAX_K = int(os.environ.get("KB_MAX_K", "20"))

# ---- Optional external tools (leave blank to use fallbacks) ----
ADDRESS_VERIFY_URL = os.environ.get("ADDRESS_VERIFY_URL", "")
ADDRESS_VERIFY_TOKEN = os.environ.get("ADDRESS_VERIFY_TOKEN", "")

ATTORNEY_VERIFY_URL = os.environ.get("ATTORNEY_VERIFY_URL", "")
ATTORNEY_VERIFY_TOKEN = os.environ.get("ATTORNEY_VERIFY_TOKEN", "")
