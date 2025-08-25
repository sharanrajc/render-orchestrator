# app/prompts.py

PROMPTS = {
    # --- Opening / context ---
    "INTRO": (
        "Hi, I’m Anna with Oasis — your intake assistant for pre-settlement funding. "
        "I’ll collect a few details to start your application and confirm them with you."
    ),

    # --- Capture prompts (LLM parses free speech; we keep them short + instructive) ---
    "ASK_NAME": "To start, please say your full legal name as on your ID — for example: “Joe Smith.”",
    "ASK_PHONE": "What’s the best phone number to reach you?",
    "ASK_EMAIL": "What’s the best email address for updates?",
    "ASK_ADDRESS": (
        "What’s your residential address including city, state, and ZIP? "
        "You can say “skip” if you prefer not to share."
    ),
    "ASK_ATTORNEY_YN": "Do you currently have an attorney representing you? Please say yes or no.",
    "ASK_ATTORNEY_INFO": (
        "Please share the attorney’s name, phone number, the law firm, and the firm’s address."
    ),
    "ASK_CASE_TYPE": "Which best describes your case: auto accident, slip and fall, or dog bite?",
    "ASK_INJURY_DETAILS": "I’m sorry you’re going through this. Briefly, what happened?",
    "ASK_INCIDENT_DATE": "On what date did the incident occur? Please say the month, day, and year.",
    "ASK_FUNDING_TYPE": "Are you looking for fresh funding, or to extend or top up existing funding?",
    "ASK_FUNDING_AMOUNT": "About how much funding are you looking for, in U.S. dollars?",

    # --- Confirmations / spelling (read-back to minimize errors) ---
    "CONFIRM_NAME": "I have your full legal name as {name}. I’ll spell it: {spelled}. Is that correct?",
    "NAME_SPELL_PROMPT": "Please say your full legal name, spelling it slowly, letter by letter.",
    "CONFIRM_PHONE": "I have your preferred phone number as {phone}. Is that correct?",
    "CONFIRM_EMAIL_SPELL": "I have your email as {email}. Spelled as: {spelled}. Is that correct?",
    "EMAIL_SPELL_PROMPT": "Please say your full email address, spelling it slowly.",
    "CONFIRM_ADDRESS": "I have your address as: {address}. Is that correct?",
    "CONFIRM_ATTORNEY": "I captured attorney information as: {summary}. Is that correct?",
    "CONFIRM_INJURY_DETAILS": "Here’s the incident summary I noted: {details}. Is that correct?",

    # --- Address & state notes (best-effort informational) ---
    "ADDRESS_SKIPPED": "No problem — I’ll note that the address was not provided.",
    "ADDRESS_VERIFY_FAIL": "Thanks. I couldn’t verify that address. I’ll note it and a specialist will double-check.",
    "STATE_ELIGIBLE": "Based on our info, Oasis serves clients in {state}.",
    "STATE_INELIGIBLE": "It looks like Oasis does not serve clients in {state}. A specialist can confirm.",

    # --- Summary / correction / wrap-up ---
    "SUMMARY_INTRO": "Let me repeat what I captured.",
    "SUMMARY_CONFIRM": "Is everything correct?",
    "HANDOFF": "Okay, I’ll connect you to a specialist now and share the information I’ve captured.",
    "DONE": "Thank you. A case specialist will reach out shortly.",
    "CORRECT_SELECT": (
        "Which information would you like to change? "
        "You can say: name, phone, email, address, attorney, case, incident date, funding type, or funding amount."
    ),
    "CORRECT_ACK": "Got it — let’s update that.",

    # --- Q&A phase ---
    "QNA_OFFER": "Before we wrap up, do you have any questions for me?",
    "QNA_PROMPT": "Sure — what would you like to know?",
    "QNA_FOLLOWUP": "Anything else I can clarify?",
    "QNA_WRAP": "Great — I’ll pass your application to a case specialist now.",
}
