# Safe no-op stubs you can later replace with real verifiers

async def verify_address(address: str):
    # return (verified_bool, normalized_str)
    return (False, address)

async def verify_attorney(attorney_name: str | None, law_firm: str | None, attorney_phone: str | None, law_firm_address: str | None):
    return False
