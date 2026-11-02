"""Speaker diarization ("who spoke when") with pyannote.audio.

Requires a Hugging Face token and accepting the model terms once on the model pages
(see README > Setup). Works with pyannote.audio 3.x and 4.x.
"""
import numpy as np
import soundfile as sf

from . import config


class SpeakerDiarizer:
    def __init__(self, hf_token, models=config.DIARIZATION_MODELS, device=None):
        import torch
        from pyannote.audio import Pipeline

        errors = []
        self.pipeline, self.model_name = None, None
        for name in models:  # try the newest model first, then fall back
            try:
                try:
                    pipe = Pipeline.from_pretrained(name, token=hf_token)          # pyannote 4.x
                except TypeError:
                    pipe = Pipeline.from_pretrained(name, use_auth_token=hf_token)  # pyannote 3.x
                if pipe is not None:
                    self.pipeline, self.model_name = pipe, name
                    break
            except Exception as e:
                errors.append(f"{name}: {e}")
        if self.pipeline is None:
            raise RuntimeError("Could not load a diarization model. Did you accept the model terms "
                               "on Hugging Face and set HF_TOKEN?\n" + "\n".join(errors))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(torch.device(device))

    @staticmethod
    def _load_waveform(audio_path):
        """Load audio in memory as a mono (1, n) tensor; avoids audio-backend issues in pyannote."""
        import torch
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        return {"waveform": torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0), "sample_rate": sr}

    def diarize(self, audio_path, num_speakers=config.NUM_SPEAKERS):
        """Return a list of {start, end, speaker} segments, e.g. speaker = 'SPEAKER_00'."""
        output = self.pipeline(self._load_waveform(audio_path), num_speakers=num_speakers)
        # pyannote 4.x returns an object; "exclusive" output has no overlapping speech (best for alignment)
        if hasattr(output, "exclusive_speaker_diarization"):
            annotation = output.exclusive_speaker_diarization
        elif hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization
        else:
            annotation = output  # pyannote 3.x returns the Annotation directly
        return [{"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
                for turn, _, label in annotation.itertracks(yield_label=True)]
