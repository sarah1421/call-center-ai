"""Decide which diarized speaker is the Agent and which is the Customer.

Method (explained in the README):
1. Keyword evidence: agents typically greet, introduce themselves and offer help
   ("thank you for calling", "how can I help"); customers describe problems ("I have a problem").
2. LLM check (Groq): the first turns are sent to the LLM, which names the agent and gives a reason.
3. Final decision: LLM answer when available; otherwise keywords; if still tied, the first speaker
   is assumed to be the agent (agents usually open inbound calls).
"""
import re

from .utils import format_turns

AGENT_PHRASES = [
    "thank you for calling", "thanks for calling", "how can i help", "how may i help", "how can i assist",
    "how may i assist", "my name is", "this is .* speaking", "speaking with", "can i have your",
    "could you please confirm", "can you verify", "let me check", "let me look", "i can help",
    "is there anything else", "have a great day", "have a nice day", "for calling", "i apologize",
    "sorry for the inconvenience", "please hold", "account number", "date of birth",
]
CUSTOMER_PHRASES = [
    "i have a problem", "i have an issue", "i'm calling about", "i am calling about", "i'm calling because",
    "i want to", "i would like to", "i'd like to", "i need", "my order", "my account", "my bill",
    "it's not working", "it is not working", "i was charged", "can you help me",
]

SYSTEM_PROMPT = (
    "You are analysing a customer-service phone call transcript produced by speech recognition and "
    "speaker diarization. Two speakers appear as SPEAKER labels. Decide which label is the call-center "
    "AGENT and which is the CUSTOMER. Return JSON: "
    '{"agent": "<label>", "customer": "<label>", "confidence": <0-1>, "reason": "<one sentence>"}'
)


class RoleLabeler:
    def __init__(self, llm=None, n_turns=12):
        self.llm = llm          # optional GroqLLM; without it only keywords are used
        self.n_turns = n_turns  # only the opening of the call is needed (saves tokens)

    @staticmethod
    def keyword_scores(turns):
        scores = {}
        for t in turns:
            text = t["text"].lower()
            s = scores.setdefault(t["speaker"], 0)
            s += sum(1 for p in AGENT_PHRASES if re.search(p, text))
            s -= sum(1 for p in CUSTOMER_PHRASES if re.search(p, text))
            scores[t["speaker"]] = s
        return scores

    def label(self, turns):
        """Return (mapping {raw_speaker: 'Agent'|'Customer'}, method, reason)."""
        speakers = list(dict.fromkeys(t["speaker"] for t in turns))
        if len(speakers) == 1:
            return {speakers[0]: "Unknown"}, "single-speaker", "Diarization found only one speaker."

        scores = self.keyword_scores(turns)
        ranked = sorted(speakers, key=lambda s: scores.get(s, 0), reverse=True)
        kw_agent = ranked[0] if scores.get(ranked[0], 0) != scores.get(ranked[1], 0) else None

        agent, method, reason = None, None, None
        if self.llm is not None:
            try:
                opening = [{"speaker": t["speaker"], "text": t["text"]} for t in turns[: self.n_turns]]
                ans = self.llm.chat_json(SYSTEM_PROMPT, format_turns(opening, max_chars=4000))
                if ans.get("agent") in speakers:
                    agent, method = ans["agent"], "llm"
                    reason = f"LLM (confidence {ans.get('confidence')}): {ans.get('reason')}"
                    if kw_agent and kw_agent != agent:
                        reason += f" | Note: keyword heuristic disagreed (picked {kw_agent})."
            except Exception as e:
                reason = f"LLM failed ({e}); "
        if agent is None and kw_agent:
            agent, method = kw_agent, "keywords"
            reason = (reason or "") + f"Keyword scores {scores}"
        if agent is None:
            agent, method = speakers[0], "first-speaker"
            reason = (reason or "") + "No clear evidence; first speaker assumed to be the agent."

        mapping = {s: ("Agent" if s == agent else "Customer") for s in speakers}
        return mapping, method, reason
