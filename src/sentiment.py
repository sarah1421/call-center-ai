"""Customer sentiment: overall label, start vs. end of call, justification and evidence quotes."""
from .utils import format_turns

LABELS = ["Negative", "Neutral", "Positive"]

SYSTEM_PROMPT = """You analyse CUSTOMER sentiment in a call-center conversation.
You receive the customer's turns from the START of the call, from the END of the call, and the full
customer side. Judge only the customer's feelings (not the agent's politeness).
Labels must be one of: Negative, Neutral, Positive.
Return JSON:
{"overall_sentiment": "<label>",
 "start_sentiment": "<label>",
 "end_sentiment": "<label>",
 "justification": "<1-2 sentences explaining the labels>",
 "evidence": ["<short exact quote from the customer>", "<optional second quote>"],
 "confidence": <number between 0 and 1>}"""


def _normalize(label):
    label = str(label or "").strip().capitalize()
    return label if label in LABELS else "Neutral"


class SentimentAnalyzer:
    def __init__(self, llm, max_chars=8000):
        self.llm = llm
        self.max_chars = max_chars

    @staticmethod
    def split_customer_turns(turns):
        """Customer turns only, split into the first and last third of the call."""
        cust = [t for t in turns if t.get("role") == "Customer"]
        if not cust:  # roles unknown -> fall back to all turns
            cust = turns
        k = max(1, len(cust) // 3)
        return cust, cust[:k], cust[-k:]

    def analyze(self, turns):
        cust, start, end = self.split_customer_turns(turns)
        prompt = (f"START of call (customer):\n{format_turns(start)}\n\n"
                  f"END of call (customer):\n{format_turns(end)}\n\n"
                  f"FULL customer side:\n{format_turns(cust, max_chars=self.max_chars)}")
        r = self.llm.chat_json(SYSTEM_PROMPT, prompt)
        res = {
            "overall_sentiment": _normalize(r.get("overall_sentiment")),
            "start_sentiment": _normalize(r.get("start_sentiment")),
            "end_sentiment": _normalize(r.get("end_sentiment")),
            "justification": str(r.get("justification", "")).strip(),
            "evidence": [str(q) for q in (r.get("evidence") or [])][:2],
            "confidence": r.get("confidence"),
        }
        res["trend"] = sentiment_trend(res["start_sentiment"], res["end_sentiment"])
        return res


def sentiment_trend(start, end):
    """Computed in code (not by the LLM) so it is always consistent with the two labels."""
    diff = LABELS.index(end) - LABELS.index(start)
    return "Improved" if diff > 0 else "Worsened" if diff < 0 else "Unchanged"
