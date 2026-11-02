"""End-to-end pipeline: audio -> transcript -> speaker turns -> roles -> summary -> sentiment.

Each stage saves its output to results/, so notebooks can run one stage at a time and a run that is
interrupted (Colab timeout, API limit) continues where it stopped.
"""
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .aligner import assign_speakers, build_turns, number_speakers
from .utils import append_jsonl, load_json, read_jsonl, save_json


@dataclass
class CallResult:
    """Everything produced for one call (the 'minimum expected output per call' + extras)."""
    call_id: str
    turns: list
    speaker_roles: dict
    role_method: str
    role_reason: str
    summary: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


class CallCenterPipeline:
    """Holds the models and runs the stages. Components are created lazily (only when needed)."""

    def __init__(self, transcriber=None, diarizer=None, role_labeler=None, summarizer=None, sentiment=None):
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.role_labeler = role_labeler
        self.summarizer = summarizer
        self.sentiment = sentiment

    # ------------------------------------------------------------ stage 1: transcription + speakers
    def transcribe_and_segment(self, call_id, audio_path, save=True):
        asr = self.transcriber.transcribe(audio_path)
        diar = self.diarizer.diarize(audio_path)
        words = assign_speakers(asr["words"], diar)
        turns = build_turns(words)
        number_speakers(turns)
        mapping, method, reason = self.role_labeler.label(turns)
        for t in turns:
            t["role"] = mapping.get(t["speaker"], "Unknown")
        record = {
            "call_id": call_id, "asr_model": asr["model"], "diarization_model": self.diarizer.model_name,
            "full_text": asr["text"], "diarization": diar, "turns": turns,
            "speaker_roles": {t["speaker_label"]: t["role"] for t in turns},
            "role_method": method, "role_reason": reason,
        }
        if save:
            save_json(config.TRANSCRIPTS_DIR / f"{call_id}.json", record)
        return record

    # ------------------------------------------------------------ stage 2: summary
    def summarize(self, record, save=True):
        res = {"call_id": record["call_id"], **self.summarizer.summarize(record["turns"])}
        if save:
            append_jsonl(config.SUMMARIES_FILE, res)
        return res

    # ------------------------------------------------------------ stage 3: sentiment
    def analyze_sentiment(self, record, save=True):
        res = {"call_id": record["call_id"], **self.sentiment.analyze(record["turns"])}
        if save:
            append_jsonl(config.SENTIMENT_FILE, res)
        return res

    # ------------------------------------------------------------ all stages (used by the Streamlit app)
    def process_call(self, call_id, audio_path, save=False):
        record = self.transcribe_and_segment(call_id, audio_path, save=save)
        summary = self.summarize(record, save=save) if self.summarizer else {}
        senti = self.analyze_sentiment(record, save=save) if self.sentiment else {}
        return CallResult(call_id, record["turns"], record["speaker_roles"], record["role_method"],
                          record["role_reason"], summary, senti, auto_notes(record))


# ---------------------------------------------------------------- helpers
def load_transcripts():
    """All saved transcript records, sorted by call_id."""
    return [load_json(p) for p in sorted(config.TRANSCRIPTS_DIR.glob("call_*.json"))]


def done_ids(path):
    """call_ids already present in a results file (used to skip finished calls)."""
    return {r["call_id"] for r in read_jsonl(path)}


def auto_notes(record):
    """Automatic notes / limitations for one call."""
    notes = []
    low = [t for t in record["turns"] if t.get("low_confidence")]
    if low:
        notes.append(f"{len(low)} low-confidence turn(s) (possible unclear audio)")
    if record["role_method"] != "llm":
        notes.append(f"roles assigned by {record['role_method']}")
    if len(set(record["speaker_roles"].values())) < 2:
        notes.append("only one speaker detected")
    notes.append("role-played call (AppTek dataset)")
    return notes


def build_final_table():
    """Join transcripts, summaries, sentiment and evaluation into the per-call results table.

    Columns follow the assessment's minimum output: call_id, speakers, transcript, summary,
    sentiment, notes / limitations (+ accent and metrics).
    """
    from .data_loader import load_sample_metadata

    meta = {r["call_id"]: r for r in load_sample_metadata()}
    summaries = {r["call_id"]: r for r in read_jsonl(config.SUMMARIES_FILE)}
    sentiments = {r["call_id"]: r for r in read_jsonl(config.SENTIMENT_FILE)}
    evaluation = (pd.read_csv(config.EVALUATION_FILE).set_index("call_id").to_dict("index")
                  if config.EVALUATION_FILE.exists() else {})

    rows = []
    for rec in load_transcripts():
        cid = rec["call_id"]
        s, se, ev = summaries.get(cid, {}), sentiments.get(cid, {}), evaluation.get(cid, {})
        rows.append({
            "call_id": cid,
            "accent": meta.get(cid, {}).get("accent"),
            "duration_min": round((meta.get(cid, {}).get("duration") or 0) / 60, 2),
            "speakers": ", ".join(f"{k} = {v}" for k, v in sorted(rec["speaker_roles"].items())),
            "transcript": "\n".join(f"[{t['start']:.1f}s] {t['role']}: {t['text']}" for t in rec["turns"]),
            "customer_issue": s.get("customer_issue"),
            "agent_action": s.get("agent_action"),
            "outcome": s.get("outcome"),
            "summary": s.get("summary"),
            "sentiment": se.get("overall_sentiment"),
            "sentiment_start": se.get("start_sentiment"),
            "sentiment_end": se.get("end_sentiment"),
            "sentiment_trend": se.get("trend"),
            "sentiment_reason": se.get("justification"),
            "sentiment_evidence": " | ".join(se.get("evidence", []) or []),
            "wer": ev.get("wer"),
            "wer_whisper_norm": ev.get("wer_whisper_norm"),
            "role_accuracy": ev.get("role_accuracy"),
            "der": ev.get("der"),
            "notes": "; ".join(auto_notes(rec)),
            "role_method": rec["role_method"],
            "role_reason": rec["role_reason"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.FINAL_CSV, index=False)
    full = []
    for row, rec in zip(rows, load_transcripts()):
        full.append({**row, "turns": rec["turns"]})
    save_json(config.FINAL_JSON, full)
    return df
