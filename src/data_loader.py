"""Dataset access: list the AppTek call-center dataset, pick a reproducible sample, download it
and run data-quality checks.

Only the selected calls are downloaded (the full dataset is ~52 GB).
"""
import random
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFolder

from . import config
from .utils import read_jsonl, write_jsonl


class AppTekDataLoader:
    """Selects and downloads calls from `apptek-com/apptek_callcenter_dialogues` (diarization config).

    Layout on Hugging Face:  diarization/<accent>/metadata.jsonl  +  diarization/<accent>/audio/*.wav
    Each metadata line holds the call's file_name, duration, domain, accent and the
    ground-truth `segments` (start, end, role, speaker_id, gender, text).
    """

    def __init__(self, repo_id=config.DATASET_REPO, config_dir=config.DATASET_CONFIG_DIR, token=None):
        self.repo_id = repo_id
        self.config_dir = config_dir
        self.token = token
        self.api = HfApi(token=token)

    # ---------------------------------------------------------- catalog
    def list_accent_dirs(self):
        """Return the accent folder names (e.g. 'en-US_General') under the diarization config."""
        items = self.api.list_repo_tree(self.repo_id, path_in_repo=self.config_dir, repo_type="dataset")
        return sorted(Path(i.path).name for i in items if isinstance(i, RepoFolder))

    def load_accent_metadata(self, accent_dir):
        """Download one small metadata.jsonl and return its rows, tagged with the accent folder."""
        path = hf_hub_download(
            self.repo_id, f"{self.config_dir}/{accent_dir}/metadata.jsonl",
            repo_type="dataset", token=self.token, local_dir=config.HF_CACHE_DIR,
        )
        rows = read_jsonl(path)
        for r in rows:
            r["accent_dir"] = accent_dir
        return rows

    def build_catalog(self):
        """Metadata of every call in the diarization config as a DataFrame (no audio downloaded)."""
        rows = []
        for accent_dir in self.list_accent_dirs():
            rows.extend(self.load_accent_metadata(accent_dir))
        df = pd.DataFrame(rows)
        df["duration_min"] = df["duration"] / 60
        df["n_segments"] = df["segments"].apply(len)
        return df

    # ---------------------------------------------------------- sampling
    @staticmethod
    def select_sample(catalog, n_calls=config.N_CALLS, max_minutes=config.MAX_CALL_MINUTES,
                      seed=config.RANDOM_SEED):
        """Stratified sample: take calls round-robin across accents so every accent is represented.

        Calls longer than `max_minutes` are excluded to keep processing time manageable.
        The fixed seed makes the same sample come out on every run.
        """
        rng = random.Random(seed)
        eligible = catalog[catalog["duration_min"] <= max_minutes]
        pools = {}
        for accent, group in eligible.groupby("accent_dir"):
            idx = list(group.index)
            rng.shuffle(idx)
            pools[accent] = idx
        chosen = []
        while len(chosen) < n_calls and any(pools.values()):
            for accent in sorted(pools):
                if pools[accent] and len(chosen) < n_calls:
                    chosen.append(pools[accent].pop())
        sample = eligible.loc[chosen].reset_index(drop=True)
        sample.insert(0, "call_id", [f"call_{i + 1:03d}" for i in range(len(sample))])
        return sample

    # ---------------------------------------------------------- download
    def download_sample(self, sample):
        """Download the audio of each selected call and save it as data/raw/audio/<call_id>.wav.

        Also writes data/raw/sample_metadata.jsonl, which keeps the ground-truth segments
        used later for evaluation.
        """
        config.ensure_dirs()
        records = []
        for row in sample.to_dict("records"):
            target = config.AUDIO_DIR / f"{row['call_id']}.wav"
            if not target.exists():  # skip files already downloaded (safe to re-run)
                src = hf_hub_download(
                    self.repo_id, f"{self.config_dir}/{row['accent_dir']}/{row['file_name']}",
                    repo_type="dataset", token=self.token, local_dir=config.HF_CACHE_DIR,
                )
                shutil.copy(src, target)
            records.append({
                "call_id": row["call_id"],
                "source_file": f"{self.config_dir}/{row['accent_dir']}/{row['file_name']}",
                "accent": row.get("accent"),
                "domain": row.get("domain"),
                "duration": row.get("duration"),
                "audio_path": str(target.relative_to(config.PROJECT_ROOT)),
                "segments": row["segments"],
            })
        write_jsonl(config.SAMPLE_METADATA, records)
        return records


def load_sample_metadata():
    """Selected calls with their ground truth (created by notebook 01)."""
    return read_jsonl(config.SAMPLE_METADATA)


def quality_report(records):
    """Data-quality checks for each downloaded call. Returns one row per call.

    Checks: file readable, sample rate / channels, audio length vs. metadata duration,
    segments with a missing role or empty text, and whether both agent and customer appear.
    """
    rows = []
    for r in records:
        path = config.PROJECT_ROOT / r["audio_path"]
        check = {"call_id": r["call_id"], "accent": r["accent"], "domain": r["domain"]}
        try:
            info = sf.info(path)
            check.update(sample_rate=info.samplerate, channels=info.channels,
                         audio_minutes=round(info.duration / 60, 2), file_ok=True)
            check["duration_mismatch_s"] = round(abs(info.duration - (r["duration"] or 0)), 2)
        except Exception as e:  # unreadable / missing file
            check.update(file_ok=False, error=str(e))
        segs = r["segments"]
        roles = {s.get("role") for s in segs}
        check.update(
            n_segments=len(segs),
            missing_role=sum(1 for s in segs if not s.get("role")),
            empty_text=sum(1 for s in segs if not str(s.get("text", "")).strip()),
            has_agent_and_customer={"agent", "customer"} <= roles,
        )
        rows.append(check)
    return pd.DataFrame(rows)
