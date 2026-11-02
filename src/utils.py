"""Small file helpers shared by all modules."""
import json
from pathlib import Path


def _to_builtin(obj):
    """Convert numpy numbers/arrays (returned by Whisper and pyannote) into plain Python for JSON."""
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_jsonl(path):
    """Read a JSON-lines file into a list of dicts (empty list if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Overwrite a JSON-lines file with the given rows."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_to_builtin) + "\n")


def append_jsonl(path, row):
    """Append one row; used so a long run can resume after an interruption."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_to_builtin) + "\n")


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_to_builtin)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_turns(turns, max_chars=None):
    """Render speaker turns as plain text, e.g. 'Agent: Hello, how can I help?'."""
    lines = [f"{t.get('role') or t.get('speaker')}: {t['text']}" for t in turns]
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n[... transcript truncated ...]"
    return text
