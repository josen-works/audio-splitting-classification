"""Stage 2: label each stem with YAMNet (521 AudioSet sound classes)."""
import csv

import numpy as np

from . import config
from .audio_io import rms, to_mono_16k

_yamnet = {}


def _load():
    """Lazy-load YAMNet from TensorFlow Hub (downloaded once, then cached)."""
    if not _yamnet:
        import tensorflow_hub as hub  # imported late: TensorFlow is slow to import

        model = hub.load(config.YAMNET_HANDLE)
        with open(model.class_map_path().numpy().decode("utf-8"), newline="") as f:
            names = [row["display_name"] for row in csv.DictReader(f)]
        _yamnet["model"], _yamnet["names"] = model, names
    return _yamnet["model"], _yamnet["names"]


def class_names() -> list:
    return _load()[1]


def classify_waveform(wave16k: np.ndarray) -> np.ndarray:
    """Run YAMNet. Returns per-frame scores shaped (frames, 521)."""
    model, _ = _load()
    scores, _emb, _spec = model(wave16k)
    return scores.numpy()


def summarize(scores: np.ndarray, names: list, top_k: int = config.TOP_K) -> dict:
    """Condense frame scores into top-k classes (mean score) and a per-frame timeline."""
    mean = scores.mean(axis=0)
    top = np.argsort(mean)[::-1][:top_k]
    timeline = [
        {"time_s": round(i * config.FRAME_HOP_S, 2),
         "label": names[int(f.argmax())],
         "score": round(float(f.max()), 4)}
        for i, f in enumerate(scores)
    ]
    return {
        "top_classes": [{"label": names[int(i)], "score": round(float(mean[i]), 4)} for i in top],
        "timeline": timeline,
    }


def stem_level(wave: np.ndarray) -> float:
    """RMS of the stem's mono mix-down."""
    return rms(wave.mean(axis=0) if wave.ndim == 2 else wave)


def classify_stem(wave: np.ndarray, sample_rate: int, reference_rms: float | None = None) -> dict:
    """Classify one stem (channels, samples).

    Stems that are near-silent in absolute terms, or more than RELATIVE_SILENCE_DB below
    `reference_rms` (the loudest stem), hold only separation bleed. They are flagged as
    silent rather than classified, since YAMNet would just label the residue.
    """
    level = stem_level(wave)
    rel_db = None
    if reference_rms:
        rel_db = round(20 * np.log10(max(level, 1e-12) / reference_rms), 1)
    base = {"rms": round(level, 6), "level_db_vs_loudest": rel_db}
    too_quiet = level < config.SILENCE_RMS or (rel_db is not None and rel_db < config.RELATIVE_SILENCE_DB)
    mono = to_mono_16k(wave, sample_rate, config.YAMNET_SAMPLE_RATE)
    if too_quiet or mono.size < config.YAMNET_SAMPLE_RATE:
        return {"silent": True, **base, "top_classes": [], "timeline": []}
    result = summarize(classify_waveform(mono), class_names())
    return {"silent": False, **base, **result}
