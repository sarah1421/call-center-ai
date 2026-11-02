"""Per-call summary: customer issue, agent action, outcome (LLM via Groq)."""
from .utils import format_turns

SYSTEM_PROMPT = """You are a call-center quality analyst. Summarize the phone call transcript below.
Rules:
- Use ONLY information stated in the transcript. Do not invent names, numbers or outcomes.
- If something is not stated, write "Not stated".
- The transcript comes from speech recognition, so it may contain small errors; ignore filler words.
Return JSON with exactly these keys:
{"customer_issue": "<one sentence>",
 "agent_action": "<one sentence>",
 "outcome": "<one sentence: resolved / unresolved / next step>",
 "summary": "<2-3 sentences, max 60 words>"}"""


class CallSummarizer:
    def __init__(self, llm, max_chars=12000):
        self.llm = llm
        self.max_chars = max_chars  # keeps long calls inside the model's token limit

    def summarize(self, turns):
        transcript = format_turns(turns, max_chars=self.max_chars)
        result = self.llm.chat_json(SYSTEM_PROMPT, f"Transcript:\n{transcript}")
        keys = ["customer_issue", "agent_action", "outcome", "summary"]
        return {k: str(result.get(k, "Not stated")).strip() for k in keys}
