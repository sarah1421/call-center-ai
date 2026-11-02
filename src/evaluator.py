"""Validation against the dataset's human ground truth.

- WER (word error rate): how accurate the Whisper transcript is.
- Role accuracy: share of speech time where our Agent/Customer label matches the human label.
- DER (diarization error rate, optional): speaker-separation quality, independent of role names.
"""
import re

import jiwer


FILLERS = {"um", "uh", "umm", "uhm", "hmm", "mm", "mhm", "ah", "eh", "er", "erm"}


def normalize_text(text):
    """Lowercase; drop annotations like [noise], <unk> and fillers written as (Um) / (uh);
    drop punctuation (keep apostrophes) and bare filler words.

    The AppTek reference marks fillers in parentheses, while Whisper usually omits them,
    so keeping them would count as Whisper errors that are not real errors.
    """
    text = str(text).lower()
    text = re.sub(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(w for w in text.split() if w not in FILLERS)


def reference_text(segments):
    """Ground-truth transcript: human segments joined in time order."""
    return " ".join(s["text"] for s in sorted(segments, key=lambda s: s["start"]))


def whisper_normalize(text):
    """Whisper's official English normalizer (used in the Whisper paper): also unifies numbers
    ("nine fifteen" vs "9:15"), spellings (colour/color) and contractions. Falls back to the basic one."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
    except ImportError:
        return normalize_text(text)
    text = re.sub(r"\([^)]*\)", " ", str(text))  # AppTek fillers like (Um)
    return EnglishTextNormalizer()(text)


def word_error_rate(reference, hypothesis):
    """WER with basic normalization, plus `wer_whisper_norm` using Whisper's normalizer.

    The basic version counts formatting differences (e.g. '9.15' vs 'nine fifteen') as errors;
    the Whisper-normalized version is closer to the real recognition error.
    """
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return None
    out = jiwer.process_words(ref, hyp)
    result = {"wer": round(out.wer, 4), "substitutions": out.substitutions,
              "deletions": out.deletions, "insertions": out.insertions, "ref_words": len(ref.split())}
    ref_w, hyp_w = whisper_normalize(reference), whisper_normalize(hypothesis)
    if ref_w.strip():
        out_w = jiwer.process_words(ref_w, hyp_w)
        result.update(wer_whisper_norm=round(out_w.wer, 4),
                      errors_whisper_norm=out_w.substitutions + out_w.deletions + out_w.insertions,
                      ref_words_whisper_norm=len(ref_w.split()))
    return result


def role_accuracy(pred_turns, gt_segments):
    """Time-weighted agreement between predicted roles and human roles (agent/customer)."""
    matched = total = 0.0
    for g in gt_segments:
        g_role = str(g.get("role", "")).lower()
        for p in pred_turns:
            overlap = min(p["end"], g["end"]) - max(p["start"], g["start"])
            if overlap > 0:
                total += overlap
                if str(p.get("role", "")).lower() == g_role:
                    matched += overlap
    return round(matched / total, 4) if total else None


def diarization_error_rate(pred_turns, gt_segments):
    """DER with pyannote.metrics (optimal speaker mapping). Returns None if pyannote is unavailable."""
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        return None
    ref, hyp = Annotation(), Annotation()
    for g in gt_segments:
        if g["end"] > g["start"]:
            ref[Segment(g["start"], g["end"])] = g.get("role", "unk")
    for p in pred_turns:
        if p["end"] > p["start"]:
            hyp[Segment(p["start"], p["end"])] = p["speaker"]
    return round(float(DiarizationErrorRate()(ref, hyp)), 4)
