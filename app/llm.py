
import os, json
from typing import Dict, Any
from openai import OpenAI

MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a low-latency voice intake helper for personal-injury pre‑settlement funding.\n"
    "Rules:\n"
    "- Never give legal advice.\n"
    "- Work one stage at a time.\n"
    "- Update ONLY allowed keys for the current stage.\n"
    "- Keep next_prompt ≤ 140 chars, empathetic and concise.\n"
    "- If KB_context is present and user asks a factual question, you may add a short phrase with [1]/[2].\n"
    "- Return STRICT JSON: {updates,next_prompt,completed,handoff,confidence}.\n"
)

def call_llm_mini(stage: str, utterance: str, slots: Dict[str, Any], kb_context: str, allowed_keys: list) -> Dict[str, Any]:
    user_block = {"stage": stage, "utterance": utterance or "", "allowed_keys_for_stage": allowed_keys, "slots": slots}
    if kb_context: user_block["KB_context"] = kb_context
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.2, max_tokens=140,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": json.dumps(user_block, ensure_ascii=False)}
            ]
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        out = {
            "updates": data.get("updates", {}),
            "next_prompt": data.get("next_prompt", "Could you say that again?"),
            "completed": bool(data.get("completed", False)),
            "handoff": bool(data.get("handoff", False)),
            "confidence": float(data.get("confidence", 0.7)),
        }
        stage_key = stage.lower()
        if isinstance(out["updates"], dict) and stage_key in out["updates"]:
            filtered = {k:v for k,v in out["updates"][stage_key].items() if k in allowed_keys}
            out["updates"] = {stage_key: filtered}
        else:
            out["updates"] = {}
        return out
    except Exception:
        return {"updates": {}, "next_prompt": "Sorry, I missed that. Could you repeat?", "completed": False, "handoff": False, "confidence": 0.4}
