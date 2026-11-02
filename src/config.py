"""Central configuration: paths, model names and run settings.

Every notebook and the Streamlit app import from here, so a setting is changed in one place only.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
AUDIO_DIR = RAW_DIR / "audio"                      # selected calls, renamed call_001.wav ...
HF_CACHE_DIR = RAW_DIR / "hf_download"             # original files as downloaded from Hugging Face
SAMPLE_METADATA = RAW_DIR / "sample_metadata.jsonl"  # one line per selected call (incl. ground truth)

RESULTS_DIR = PROJECT_ROOT / "results"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"      # one JSON per call: words, turns, roles
SUMMARIES_FILE = RESULTS_DIR / "summaries.jsonl"
SENTIMENT_FILE = RESULTS_DIR / "sentiment.jsonl"
EVALUATION_FILE = RESULTS_DIR / "evaluation.csv"
QUALITY_REPORT_FILE = RESULTS_DIR / "data_quality_report.csv"
FINAL_JSON = RESULTS_DIR / "final_results.json"
FINAL_CSV = RESULTS_DIR / "final_results.csv"

# ---------------------------------------------------------------- dataset
DATASET_REPO = "apptek-com/apptek_callcenter_dialogues"
DATASET_CONFIG_DIR = "diarization"   # one merged (both speakers) audio file per call
N_CALLS = 20                         # size of the processed sample
MAX_CALL_MINUTES = 8                 # skip very long calls to keep Colab run time reasonable
RANDOM_SEED = 42                     # makes the sample reproducible

# ---------------------------------------------------------------- models
WHISPER_MODEL = "turbo"              # OpenAI Whisper large-v3-turbo; use "small" on CPU
LANGUAGE = "en"
DIARIZATION_MODELS = [               # tried in order (pyannote 4.x model first, 3.x fallback)
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]
NUM_SPEAKERS = 2                     # every call is agent + customer
# Groq models change over time, so the code asks Groq which models this account can use and picks
# the first available one from this preference list (set GROQ_MODEL to force a specific model).
GROQ_MODEL_PREFERENCE = [
    os.getenv("GROQ_MODEL", ""),
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant",
]
LOW_CONFIDENCE_THRESHOLD = 0.5       # mean word probability below this => turn flagged as unclear


def get_secret(name: str):
    """Read a secret from Colab Secrets (key icon in the left bar) or from an environment variable.

    Keys are never written in the code or committed to the repository.
    """
    try:
        from google.colab import userdata  # only exists inside Colab
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


def ensure_dirs():
    """Create all output folders if they do not exist yet."""
    for d in (AUDIO_DIR, HF_CACHE_DIR, TRANSCRIPTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
