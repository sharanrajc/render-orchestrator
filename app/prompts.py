PROMPTS = {
    "INTRO": (
        "Hi, I’m Anna with Oasis, your intake assistant for pre-settlement funding. "
        "I’ll take your application, confirm details, and share next steps."
    ),
    "EXISTING": (
        "I see an existing application for this number. "
        "Would you like to continue a pending application, modify a completed one, or start a new application?"
    ),
    "ASK_NAME": "To start, may I have your full legal name as it appears on your ID?",
    "ASK_LAST_NAME": "Thanks {first}. And your legal last name?",
    "ASK_PREFERRED_PHONE_CONFIRM": "I can use the number you’re calling from: {caller}. Is that your preferred contact number?",
    "ASK_PHONE": "What’s the preferred phone number to reach you at? You can say it.",
    "ASK_EMAIL": "Great. What’s the best email address for updates?",
    "ASK_ADDRESS": "What’s your residential address, including city, state, and ZIP? You can say skip if you prefer not to share.",
    "ADDRESS_SKIPPED": "No problem, I’ll note that address was not provided.",
    "ADDRESS_VERIFY_FAIL": "Thanks. I couldn’t verify that address. I’ll note it and a specialist will double-check.",
    "STATE_ELIGIBLE": "Based on our info, Oasis serves clients in {state}.",
    "STATE_INELIGIBLE": "It looks like Oasis does not serve clients in {state}. A specialist can confirm.",
    "ASK_ATTORNEY_YN": "Do you currently have an attorney representing you? Please say yes or no.",
    "ASK_ATTORNEY_INFO": "Please share the attorney’s name, phone number, the law firm, and the firm’s address.",
    "ATTY_VERIFY_FAIL": "Thanks. I couldn’t verify the attorney or firm. I’ll note it and a specialist will confirm.",
    "ASK_CASE_TYPE": "Which best describes your case: auto accident, slip and fall, or dog bite?",
    "ASK_INJURY_DETAILS": "I’m sorry you’re going through this. Briefly, what happened?",
    "ASK_INCIDENT_DATE": "On what date did the incident occur? Please say the month, day, and year.",
    "ASK_FUNDING_TYPE": "Are you looking for fresh funding, or to extend or top up existing funding?",
    "ASK_FUNDING_AMOUNT": "About how much funding are you looking for, in U.S. dollars?",
    "SUMMARY_INTRO": "Let me repeat what I captured.",
    "SUMMARY_CONFIRM": "Is this correct?",
    "DONE": "Thanks for the confirmation. A case specialist will reach out shortly.",
    "HANDOFF": "Okay, I’ll connect you to a specialist and share the info I’ve captured.",
    "REPROMPT_SHORT": "Sorry, I didn’t catch that. ",
    "CORRECT_SELECT": "Which information would you like to change? You can say: name, phone, email, address, attorney, case, incident date, funding type, or funding amount.",
    "CORRECT_ACK": "Got it. Let’s update that.",
}
