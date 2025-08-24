import httpx
from typing import Tuple
from .config import (
    ADDRESS_VERIFY_URL, ADDRESS_VERIFY_TOKEN,
    ATTORNEY_VERIFY_URL, ATTORNEY_VERIFY_TOKEN
)

async def verify_address(raw_address: str) -> Tuple[bool, str]:
    if not raw_address:
        return (False, "")
    if ADDRESS_VERIFY_URL:
        try:
            headers = {"Authorization": f"Bearer {ADDRESS_VERIFY_TOKEN}"} if ADDRESS_VERIFY_TOKEN else {}
            async with httpx.AsyncClient(timeout=5.0) as cli:
                r = await cli.post(ADDRESS_VERIFY_URL, json={"address": raw_address}, headers=headers)
                r.raise_for_status()
                data = r.json()
                return (bool(data.get("verified", False)), data.get("normalized", "") or raw_address)
        except Exception:
            pass
    # fallback: optimistic normalization
    return (True, raw_address.title())

async def verify_attorney(attorney_name: str, firm: str, phone: str, firm_address: str) -> bool:
    if not (attorney_name or firm or phone):
        return False
    if ATTORNEY_VERIFY_URL:
        try:
            headers = {"Authorization": f"Bearer {ATTORNEY_VERIFY_TOKEN}"} if ATTORNEY_VERIFY_TOKEN else {}
            payload = {"attorney_name": attorney_name, "firm": firm, "phone": phone, "firm_address": firm_address}
            async with httpx.AsyncClient(timeout=6.0) as cli:
                r = await cli.post(ATTORNEY_VERIFY_URL, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return bool(data.get("verified", False))
        except Exception:
            return False
    return bool(attorney_name and firm)
