"""Audio loading / saving helpers (soundfile + librosa, no ffmpeg required for wav/flac/ogg)."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path, sample_rate: int, mono: bool = False) -> np.ndarray:
    """Load `path` resampled to `sample_rate`. Returns float32 shaped (channels, samples)."""
    try:
        y, _ = librosa.load(str(path), sr=sample_rate, mono=mono)
    except Exception as e:  # librosa/soundfile/audioread raise many different types
        raise ValueError(f"could not read {path} as audio ({type(e).__name__}: {e})") from None
    y = np.atleast_2d(y).astype(np.float32)
    return y


def to_stereo(wave: np.ndarray) -> np.ndarray:
    """Demucs expects 2 channels: duplicate mono, drop extras."""
    if wave.shape[0] == 1:
        return np.repeat(wave, 2, axis=0)
    return wave[:2]


def save_wav(path, wave: np.ndarray, sample_rate: int) -> Path:
    """Write (channels, samples) float audio as 16-bit wav."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wave.T, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return path


def to_mono_16k(wave: np.ndarray, sample_rate: int, target: int = 16000) -> np.ndarray:
    """Mono float32 at `target` Hz, as YAMNet expects."""
    mono = wave.mean(axis=0) if wave.ndim == 2 else wave
    if sample_rate != target:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target)
    return mono.astype(np.float32)


def rms(wave: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wave)))) if wave.size else 0.0
