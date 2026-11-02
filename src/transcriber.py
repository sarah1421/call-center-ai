"""Speech-to-text with OpenAI Whisper (word-level timestamps)."""
from . import config


class WhisperTranscriber:
    """Wraps `openai-whisper`. Word timestamps are needed to match each word to a speaker later."""

    def __init__(self, model_name=config.WHISPER_MODEL, language=config.LANGUAGE, device=None):
        import torch
        import whisper  # imported here so the rest of the project works without it

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.language = language
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio_path):
        """Transcribe one call. Returns full text, segments and a flat list of timed words."""
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,  # reduces repeated / hallucinated text on long audio
            fp16=(self.device == "cuda"),
            verbose=None,
        )
        words, segments = [], []
        for seg in result["segments"]:
            segments.append({
                "start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip(),
                "avg_logprob": float(seg.get("avg_logprob", 0.0)),
                "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
            })
            for w in seg.get("words", []):
                words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                              "probability": float(w.get("probability", 1.0))})  # numpy -> float
        return {"text": result["text"].strip(), "segments": segments, "words": words,
                "model": f"whisper-{self.model_name}"}
