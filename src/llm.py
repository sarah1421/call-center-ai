"""Thin wrapper around the Groq API that returns parsed JSON and handles rate limits.

The model is chosen automatically from `config.GROQ_MODEL_PREFERENCE` among the models this API key
can access, so the code keeps working when Groq retires a model.
"""
import json
import re
import time

from . import config


class GroqLLM:
    def __init__(self, api_key=None, preference=None, max_retries=5):
        from groq import Groq

        api_key = api_key or config.get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to Colab Secrets or set the environment variable.")
        self.client = Groq(api_key=api_key)
        self.max_retries = max_retries
        self.candidates = self._available_models(preference or config.GROQ_MODEL_PREFERENCE)
        self.model = self.candidates[0]
        print(f"Groq model: {self.model} (fallbacks: {self.candidates[1:]})")

    def _available_models(self, preference):
        """Keep only preferred models that this API key can use (in preference order)."""
        preference = [m for m in preference if m]
        try:
            available = {m.id for m in self.client.models.list().data}
        except Exception as e:
            print(f"Could not list Groq models ({e}); trying the preference list as is.")
            return preference
        usable = [m for m in preference if m in available]
        if not usable:
            raise RuntimeError(f"None of {preference} is available. Available models: {sorted(available)}. "
                               "Set GROQ_MODEL to one of them in src/config.py.")
        return usable

    def _next_model(self, reason):
        """Move to the next model in the list (model retired, or daily quota used up)."""
        idx = self.candidates.index(self.model)
        if idx + 1 >= len(self.candidates):
            return False
        print(f"{reason} for {self.model}; switching to {self.candidates[idx + 1]}")
        self.model = self.candidates[idx + 1]
        return True

    def _request(self, system, user, temperature):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        if self.model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"  # reasoning model: keep thinking short (faster, fewer tokens)
        try:
            return self.client.chat.completions.create(**kwargs)
        except TypeError:  # older groq SDK without reasoning_effort
            kwargs.pop("reasoning_effort", None)
            return self.client.chat.completions.create(**kwargs)

    def chat_json(self, system, user, temperature=0.0):
        """Send a prompt and return the model's answer as a dict (JSON mode, deterministic)."""
        import groq

        wait, attempt = 10, 0
        while attempt < self.max_retries:
            try:
                resp = self._request(system, user, temperature)
                return self._parse(resp.choices[0].message.content)
            except groq.NotFoundError:  # model retired / not available: no point retrying it
                if not self._next_model("Model not found"):
                    raise
            except groq.RateLimitError as e:
                msg = str(e).lower()
                if ("per day" in msg or "tpd" in msg or "rpd" in msg) and self._next_model("Daily limit reached"):
                    continue
                attempt += 1
                print(f"Rate limited, waiting {wait}s (attempt {attempt}/{self.max_retries})")
                time.sleep(wait)
                wait = min(wait * 2, 90)
            except (groq.APIConnectionError, groq.InternalServerError) as e:
                attempt += 1
                print(f"Groq temporary error: {e}; retrying in {wait}s")
                time.sleep(wait)
            except json.JSONDecodeError:
                attempt += 1
                print("Model returned invalid JSON; retrying")
        raise RuntimeError("Groq request failed after retries")

    @staticmethod
    def _parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text or "", re.S)  # recover JSON wrapped in extra text
            if match:
                return json.loads(match.group(0))
            raise
