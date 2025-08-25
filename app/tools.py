# app/tools.py
"""
Google Maps Platform helpers:
- verify_address(): normalize & validate a US address via Geocoding API
- verify_attorney(): validate attorney/law firm via Places Text Search + Place Details

Both functions are async and safe to fail (return best-effort results).
"""

from __future__ import annotations
from typing import Optional, Tuple
import asyncio
import httpx
import re

from .config import GOOGLE_MAPS_API_KEY

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# --------- utils ---------
def _us_state_from_components(components) -> Optional[str]:
    """Extract 2-letter state code from geocoding/places address_components."""
    if not components:
        return None
    state = None
    country = None
    for c in components:
        types = c.get("types", [])
        if "administrative_area_level_1" in types:
            state = c.get("short_name")
        if "country" in types:
            country = c.get("short_name")
    if (country or "").upper() != "US":
        return None
    if state and re.fullmatch(r"[A-Z]{2}", state):
        return state
    return None

def _format_address(result) -> Tuple[str, Optional[str]]:
    """Return (formatted, state) from a geocode or place result."""
    formatted = result.get("formatted_address") or ""
    state = _us_state_from_components(result.get("address_components"))
    return formatted, state

def _norm_phone_e164(us_phone: str | None) -> Optional[str]:
    if not us_phone:
        return None
    digits = re.sub(r"\D", "", us_phone)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"+1{digits}"
    return None

# --------- public API ---------
async def verify_address(address: str) -> Tuple[bool, str]:
    """
    Verify/normalize an address using Geocoding API.
    Returns (verified_bool, normalized_address).
    Only verifies US addresses (we care about state-based eligibility).
    """
    if not GOOGLE_MAPS_API_KEY or not address:
        # Behave like "not verified, but keep as is"
        return (False, address)

    params = {"address": address, "key": GOOGLE_MAPS_API_KEY, "region": "us"}
    timeout = httpx.Timeout(8.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.get(GEOCODE_URL, params=params)
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status != "OK":
                # ZERO_RESULTS or OVER_QUERY_LIMIT etc.
                return (False, address)
            best = data.get("results", [])[0]
            formatted, state = _format_address(best)
            if not state:
                # Not a US address (or missing state)
                return (False, formatted or address)
            # High-level sanity: require geometry + partial_match False for tight verification
            verified = bool(best.get("geometry")) and not best.get("partial_match", False)
            return (verified, formatted or address)
        except Exception:
            return (False, address)

async def verify_attorney(
    attorney_name: Optional[str],
    law_firm: Optional[str],
    attorney_phone: Optional[str],
    law_firm_address: Optional[str]
) -> bool:
    """
    Validate attorney & law firm using Places:
      1) Text Search with the most specific query we can form
      2) Place Details to compare name/phone/address
    We accept as verified if we can match law firm name AND (phone OR address).
    """
    if not GOOGLE_MAPS_API_KEY:
        return False

    # Build a strong query
    q_parts = []
    if attorney_name: q_parts.append(attorney_name)
    if law_firm: q_parts.append(law_firm)
    if law_firm_address: q_parts.append(law_firm_address)
    query = " ".join(q_parts).strip() or (attorney_name or law_firm or "")

    if not query:
        return False

    timeout = httpx.Timeout(8.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # 1) Text search
            ts_params = {"query": query, "key": GOOGLE_MAPS_API_KEY, "region": "us"}
            ts = await client.get(TEXTSEARCH_URL, params=ts_params)
            ts.raise_for_status()
            ts_data = ts.json()
            if ts_data.get("status") not in ("OK", "ZERO_RESULTS"):
                return False
            results = ts_data.get("results", [])
            if not results:
                # Try a looser search with only law_firm
                if law_firm and law_firm != query:
                    ts_params["query"] = law_firm
                    ts = await client.get(TEXTSEARCH_URL, params=ts_params)
                    ts.raise_for_status()
                    ts_data = ts.json()
                    results = ts_data.get("results", [])

            if not results:
                return False

            # 2) Iterate top few candidates; call Place Details for stronger signal
            for cand in results[:3]:
                place_id = cand.get("place_id")
                if not place_id:
                    continue
                det_params = {
                    "place_id": place_id,
                    "key": GOOGLE_MAPS_API_KEY,
                    "fields": "name,formatted_address,international_phone_number,formatted_phone_number,address_components"
                }
                det = await client.get(DETAILS_URL, params=det_params)
                det.raise_for_status()
                det_data = det.json()
                if det_data.get("status") != "OK":
                    continue
                pd = det_data.get("result", {})

                # Normalize values
                firm_name = (pd.get("name") or "").strip().lower()
                formatted_addr, state = _format_address(pd)
                firm_phone = _norm_phone_e164(pd.get("international_phone_number") or pd.get("formatted_phone_number"))

                # Compare law firm name (fuzzy-ish containment)
                name_ok = True
                if law_firm:
                    lf = law_firm.strip().lower()
                    name_ok = lf in firm_name or firm_name in lf

                # Compare phone/address if provided
                phone_ok = True
                if attorney_phone:
                    phone_ok = _norm_phone_e164(attorney_phone) == firm_phone

                addr_ok = True
                if law_firm_address:
                    addr_ok = (law_firm_address.strip().lower().split(",")[0] in formatted_addr.lower())

                # Accept if name matches and (phone or address) matches
                if name_ok and (phone_ok or addr_ok):
                    return True

            return False
        except Exception:
            return False
