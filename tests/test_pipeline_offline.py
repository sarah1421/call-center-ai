"""End-to-end pipeline test with fake models (no GPU, no API key, no download)."""
import numpy as np
import pandas as pd
import soundfile as sf

from src import config
from src.data_loader import AppTekDataLoader, quality_report
from src.pipeline import CallCenterPipeline, build_final_table
from src.role_labeler import RoleLabeler
from src.sentiment import SentimentAnalyzer
from src.summarizer import CallSummarizer
from tests.test_core import DIAR, WORDS


class FakeTranscriber:
    def transcribe(self, path):
        return {"text": "".join(w["word"] for w in WORDS), "words": WORDS, "segments": [], "model": "fake"}


class FakeDiarizer:
    model_name = "fake-diarizer"

    def diarize(self, path):
        return DIAR


class FakeLLM:
    def chat_json(self, system, user):
        if "AGENT" in system:
            return {"agent": "SPEAKER_01", "customer": "SPEAKER_00", "confidence": 0.9, "reason": "greets"}
        if "Summarize" in system:
            return {"customer_issue": "Has a problem", "agent_action": "Offered help",
                    "outcome": "Not stated", "summary": "Customer reports a problem."}
        return {"overall_sentiment": "negative", "start_sentiment": "Negative", "end_sentiment": "Neutral",
                "justification": "Reports a problem.", "evidence": ["I have a problem"], "confidence": 0.7}


def test_full_offline_run(tmp_path, monkeypatch):
    for name, sub in [("TRANSCRIPTS_DIR", "results/transcripts"), ("RESULTS_DIR", "results"),
                      ("AUDIO_DIR", "data/raw/audio"), ("HF_CACHE_DIR", "data/raw/hf")]:
        monkeypatch.setattr(config, name, tmp_path / sub)
    for name, f in [("SUMMARIES_FILE", "results/s.jsonl"), ("SENTIMENT_FILE", "results/se.jsonl"),
                    ("EVALUATION_FILE", "results/e.csv"), ("FINAL_CSV", "results/f.csv"),
                    ("FINAL_JSON", "results/f.json"), ("SAMPLE_METADATA", "data/raw/meta.jsonl")]:
        monkeypatch.setattr(config, name, tmp_path / f)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.ensure_dirs()

    # sampling is stratified across accents and reproducible
    cat = pd.DataFrame({"accent_dir": ["a", "a", "b", "b", "c"], "duration": [60, 70, 80, 900, 50],
                        "file_name": [f"audio/{i}.wav" for i in range(5)],
                        "segments": [[{"start": 0, "end": 1.8, "role": "agent", "text": "x"}]] * 5})
    cat["duration_min"] = cat["duration"] / 60
    s1 = AppTekDataLoader.select_sample(cat, n_calls=3, max_minutes=8, seed=1)
    s2 = AppTekDataLoader.select_sample(cat, n_calls=3, max_minutes=8, seed=1)
    assert list(s1["file_name"]) == list(s2["file_name"])
    assert set(s1["accent_dir"]) == {"a", "b", "c"}

    audio = config.AUDIO_DIR / "call_001.wav"
    sf.write(audio, np.zeros(16000 * 3, dtype="float32"), 16000)
    meta = {"call_id": "call_001", "accent": "en-US", "domain": "Telecom", "duration": 3.0,
            "audio_path": "data/raw/audio/call_001.wav",
            "segments": [{"start": 0.0, "end": 1.8, "role": "agent", "text": "thank you for calling how can i help"},
                         {"start": 2.2, "end": 3.0, "role": "customer", "text": "i have a problem"}]}
    from src.utils import write_jsonl
    write_jsonl(config.SAMPLE_METADATA, [meta])
    q = quality_report([meta])
    assert bool(q.loc[0, "file_ok"]) and bool(q.loc[0, "has_agent_and_customer"])

    llm = FakeLLM()
    pipe = CallCenterPipeline(FakeTranscriber(), FakeDiarizer(), RoleLabeler(llm),
                              CallSummarizer(llm), SentimentAnalyzer(llm))
    rec = pipe.transcribe_and_segment("call_001", audio)
    assert rec["speaker_roles"] == {"Speaker 1": "Agent", "Speaker 2": "Customer"}
    pipe.summarize(rec)
    se = pipe.analyze_sentiment(rec)
    assert se["overall_sentiment"] == "Negative" and se["trend"] == "Improved"

    df = build_final_table()
    assert list(df["call_id"]) == ["call_001"]
    assert "Agent: Thank you for calling" in df.loc[0, "transcript"]
    assert config.FINAL_JSON.exists()
