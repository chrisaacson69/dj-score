"""Demucs stem separation and the separation folders it writes (the optional `stems` extra).

Converted from camelot `separate_stems.py` (separation) and the loading half of
`resum_check.py`. torch and demucs are imported inside the calls, so importing this
module costs nothing where the extra is not installed.

A folder holds one wav per source plus `separation_info.json` (input, model,
sample_rate, device, timing, stem paths), which is how a separation is read back.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dj_score.adapters import library


@dataclass
class Separation:
    input: Path
    model: str
    sample_rate: int
    mix: np.ndarray
    """(channels, samples) float64, decoded by Demucs' reader at the model rate."""
    stems: dict[str, np.ndarray]
    """Per source, (channels, samples) float64 as saved."""


def pick_device(prefer_gpu=True) -> str:
    import torch
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


class DemucsSeparator:
    """`StemSeparator` running Demucs in this process."""

    def __init__(self, model_name="htdemucs", device=None, segment_length=None, overlap=0.25,
                 shifts=1, mp3_output=False, float_output=False):
        self.model_name = model_name
        self.device = device
        self.segment_length = segment_length
        self.overlap = overlap
        self.shifts = shifts
        self.mp3_output = mp3_output
        self.float_output = float_output

    def separate(self, audio_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
        """Write one file per source into `output_dir` (default: the track's stems folder, `library.stems_dir`) plus
        `separation_info.json`; returns {source: path}."""
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio
        from demucs.pretrained import get_model

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        output_dir = Path(output_dir) if output_dir else library.stems_dir(audio_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        device = self.device or pick_device()

        model = get_model(self.model_name)
        model.to(device)
        model.eval()
        sample_rate = model.samplerate

        # Demucs' ffmpeg reader: resamples to the model rate, returns stereo
        # (torchaudio >= 2.9 load needs TorchCodec)
        waveform = AudioFile(audio_path).read(streams=0, samplerate=sample_rate, channels=2)
        duration = waveform.shape[1] / sample_rate
        waveform = waveform.unsqueeze(0).to(device)

        # Under 6 GB of VRAM, let Demucs split the track into segments
        use_split = (self.segment_length is None and device == "cuda"
                     and torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) < 6)

        start_time = time.time()
        with torch.no_grad():
            try:
                if self.segment_length:
                    estimates = apply_model(model, waveform, device=device, shifts=self.shifts, split=True,
                                            overlap=self.overlap, segment=self.segment_length, progress=True)
                elif use_split:
                    estimates = apply_model(model, waveform, device=device, shifts=self.shifts, split=True,
                                            overlap=self.overlap, progress=True)
                else:
                    estimates = apply_model(model, waveform, device=device, shifts=self.shifts, progress=True)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                    raise
                torch.cuda.empty_cache()
                device = "cpu"
                model, waveform = model.to("cpu"), waveform.to("cpu")
                estimates = apply_model(model, waveform, device="cpu", shifts=self.shifts, split=True,
                                        overlap=self.overlap, progress=True)
        elapsed = time.time() - start_time

        stem_paths = {}
        for i, source in enumerate(model.sources):
            stem_path = output_dir / f"{source}.{'mp3' if self.mp3_output else 'wav'}"
            if self.float_output:
                # 32-bit float, unclamped: stems re-sum exactly as Demucs produced them
                save_audio(estimates[0, i].cpu(), stem_path, samplerate=sample_rate, clip="none", as_float=True)
            else:
                save_audio(estimates[0, i].cpu(), stem_path, samplerate=sample_rate, clip="clamp")
            stem_paths[source] = stem_path

        info = {
            "input": str(audio_path),
            "model": self.model_name,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "device": device,
            "processing_time_seconds": elapsed,
            "stems": {k: str(v) for k, v in stem_paths.items()},
        }
        (output_dir / "separation_info.json").write_text(json.dumps(info, indent=2))
        return stem_paths


class SeparationFolder:
    """`SeparationReader` over the folders `DemucsSeparator` writes."""

    def read(self, stems_dir: Path) -> Separation:
        import soundfile as sf
        from demucs.audio import AudioFile

        stems_dir = Path(stems_dir)
        info = json.loads((stems_dir / "separation_info.json").read_text())
        sr = info["sample_rate"]
        mix_path = Path(info["input"])
        if not mix_path.exists():
            raise FileNotFoundError(f"separation_info.json input {mix_path} does not exist "
                                    "(the info file records absolute paths; was the folder moved?)")
        stems = {}
        for name, recorded in info["stems"].items():
            # Recorded paths go stale when a folder moves; the stem sits beside the info file
            p = Path(recorded) if Path(recorded).exists() else stems_dir / Path(recorded).name
            data, file_sr = sf.read(p, dtype="float64", always_2d=True)
            if file_sr != sr:
                raise ValueError(f"{p}: {file_sr} Hz, expected {sr} Hz")
            stems[name] = data.T
        channels = next(iter(stems.values())).shape[0]
        mix = AudioFile(mix_path).read(streams=0, samplerate=sr, channels=channels).numpy().astype(np.float64)
        return Separation(mix_path, info["model"], sr, mix, stems)
