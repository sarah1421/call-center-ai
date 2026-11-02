"""Unit tests for the logic that does not need a GPU or an API key.  Run:  python -m pytest -q"""
from src.aligner import assign_speakers, build_turns, number_speakers
from src.evaluator import normalize_text, role_accuracy, word_error_rate
from src.role_labeler import RoleLabeler
from src.sentiment import sentiment_trend

WORDS = [
    {"word": " Thank", "start": 0.0, "end": 0.3, "probability": 0.9},
    {"word": " you", "start": 0.3, "end": 0.5, "probability": 0.9},
    {"word": " for", "start": 0.5, "end": 0.7, "probability": 0.9},
    {"word": " calling,", "start": 0.7, "end": 1.0, "probability": 0.9},
    {"word": " how", "start": 1.0, "end": 1.2, "probability": 0.9},
    {"word": " can", "start": 1.2, "end": 1.4, "probability": 0.9},
    {"word": " I", "start": 1.4, "end": 1.5, "probability": 0.9},
    {"word": " help?", "start": 1.5, "end": 1.8, "probability": 0.9},
    {"word": " I", "start": 2.2, "end": 2.3, "probability": 0.3},
    {"word": " have", "start": 2.3, "end": 2.5, "probability": 0.3},
    {"word": " a", "start": 2.5, "end": 2.6, "probability": 0.3},
    {"word": " problem", "start": 2.6, "end": 3.0, "probability": 0.3},
]
DIAR = [{"start": 0.0, "end": 1.9, "speaker": "SPEAKER_01"}, {"start": 2.1, "end": 3.1, "speaker": "SPEAKER_00"}]


def make_turns():
    turns = build_turns(assign_speakers(WORDS, DIAR))
    number_speakers(turns)
    return turns


def test_turns_split_by_speaker():
    turns = make_turns()
    assert [t["speaker"] for t in turns] == ["SPEAKER_01", "SPEAKER_00"]
    assert turns[0]["text"] == "Thank you for calling, how can I help?"
    assert turns[0]["speaker_label"] == "Speaker 1"
    assert turns[1]["low_confidence"] is True


def test_role_labeler_keywords():
    mapping, method, _ = RoleLabeler(llm=None).label(make_turns())
    assert mapping == {"SPEAKER_01": "Agent", "SPEAKER_00": "Customer"}
    assert method == "keywords"


def test_wer_and_normalization():
    assert normalize_text("Hello, [noise] World!") == "hello world"
    assert word_error_rate("hello world", "hello word")["wer"] == 0.5


def test_role_accuracy():
    turns = make_turns()
    for t, r in zip(turns, ["Agent", "Customer"]):
        t["role"] = r
    gt = [{"start": 0.0, "end": 1.8, "role": "agent"}, {"start": 2.2, "end": 3.0, "role": "customer"}]
    assert role_accuracy(turns, gt) == 1.0


def test_sentiment_trend():
    assert sentiment_trend("Negative", "Positive") == "Improved"
    assert sentiment_trend("Neutral", "Neutral") == "Unchanged"
