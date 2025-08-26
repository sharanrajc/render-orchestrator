# app/prompts.py
PROMPTS = {
    "INTRO": "Hi, I’m Anna with Oasis — your intake assistant for pre-settlement funding. I’ll collect a few details to start your application and confirm them with you.",
    "EXISTING": "I see an application linked to this number. Please say “continue”, “modify”, or “start new application.”",
    "CONTINUE_ACK": "Got it — continuing your application.",
    "MODIFY_ACK": "Okay, I’ll read back your details and we can make changes.",
    "NEW_ACK": "Starting a new application.",
    "DEFAULT_CONTINUE": "I’ll continue your pending application. If that’s not right, just say “stop” anytime.",

    "ASK_NAME": "To start, please say your full legal name — for example, “Joe Smith.”",
    "ASK_PHONE": "What’s the best phone number to reach you? You can say the digits.",
    "ASK_CONFIRM_CALLER_PHONE": "I can use the number you’re calling from: {phone}. Is that your preferred contact number?",
    "ASK_EMAIL": "What’s the best email address for updates?",
    "ASK_ADDRESS": "What’s your residential address including city, state, and ZIP? You can say “skip” to skip.",
    "ASK_ATTORNEY_YN": "Do you currently have an attorney representing you? Please say yes or no.",
    "ASK_ATTORNEY_INFO": "Please share the attorney’s name, phone number, the law firm, and the firm’s address.",
    "ASK_CASE_TYPE": "Which best describes your case: auto accident, slip and fall, or dog bite?",
    "ASK_INJURY_DETAILS": "I’m sorry you’re going through this. Briefly, what happened?",
    "ASK_INCIDENT_DATE": "On what date did the incident occur? For example, “August 24, 2025.”",
    "ASK_FUNDING_TYPE": "Are you looking for fresh funding, or to extend or top up existing funding?",
    "ASK_FUNDING_AMOUNT": "About how much funding are you looking for, in U.S. dollars?",

    "CONFIRM_PHONE": "I have your preferred phone number as {phone}. Is that correct?",
    "CONFIRM_EMAIL": "I have your email as {email}. Spelled as: {spelled}. Is that correct?",
    "EMAIL_SPELL_PROMPT": "Please say your full email address, spelling it slowly.",

    "SUMMARY_INTRO": "Let me repeat what I captured.",
    "SUMMARY_CONFIRM": "Is everything correct?",
    "CORRECT_SELECT": "Which information would you like to change? You can say: name, phone, email, address, attorney, case, incident date, funding type, or funding amount.",
    "CORRECT_ACK": "Got it — let’s update that.",

    "QNA_OFFER": "Before we wrap up, do you have any questions for me?",
    "QNA_PROMPT": "Sure — what would you like to know?",
    "QNA_WRAP": "Great — I’ll pass your application to a case specialist now.",
    "DONE": "Thank you. A case specialist will reach out shortly."
}
