"""Combine Whisper words with diarization segments into speaker turns."""
from . import config


def assign_speakers(words, diar_segments):
    """Give each word the speaker whose diarization segment overlaps it most.

    Words falling in a gap between segments get the speaker of the nearest segment.
    """
    out = []
    for w in words:
        best, best_overlap = None, 0.0
        for s in diar_segments:
            overlap = min(w["end"], s["end"]) - max(w["start"], s["start"])
            if overlap > best_overlap:
                best, best_overlap = s["speaker"], overlap
        if best is None and diar_segments:
            mid = (w["start"] + w["end"]) / 2
            nearest = min(diar_segments, key=lambda s: min(abs(mid - s["start"]), abs(mid - s["end"])))
            best = nearest["speaker"]
        out.append({**w, "speaker": best or "UNKNOWN"})
    return out


def build_turns(words, max_gap=2.0, low_conf=config.LOW_CONFIDENCE_THRESHOLD):
    """Merge consecutive words of the same speaker into turns.

    A new turn starts when the speaker changes or after a pause longer than `max_gap` seconds.
    Each turn keeps its mean word probability; low values are flagged as possibly unclear audio.
    """
    turns = []
    for w in words:
        same = turns and turns[-1]["speaker"] == w["speaker"] and w["start"] - turns[-1]["end"] <= max_gap
        if same:
            t = turns[-1]
            t["text"] += w["word"]
            t["end"] = w["end"]
            t["_probs"].append(w.get("probability", 1.0))
        else:
            turns.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"],
                          "text": w["word"], "_probs": [w.get("probability", 1.0)]})
    for t in turns:
        probs = t.pop("_probs")
        t["text"] = " ".join(t["text"].split())
        t["confidence"] = round(sum(probs) / len(probs), 3)
        t["low_confidence"] = t["confidence"] < low_conf
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)
    return turns


def number_speakers(turns):
    """Map raw diarization labels to 'Speaker 1', 'Speaker 2' in order of first appearance."""
    order = []
    for t in turns:
        if t["speaker"] not in order:
            order.append(t["speaker"])
    mapping = {spk: f"Speaker {i + 1}" for i, spk in enumerate(order)}
    for t in turns:
        t["speaker_label"] = mapping[t["speaker"]]
    return mapping
