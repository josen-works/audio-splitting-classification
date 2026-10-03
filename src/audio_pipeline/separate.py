"""Stage 1: split a mixture into stems with Demucs."""
from pathlib import Path

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from . import config
from .audio_io import load_audio, save_wav, to_stereo

_models = {}


def _get(name: str):
    if name not in _models:
        model = get_model(name)
        model.eval()
        _models[name] = model
    return _models[name]


def model_sources(model_name: str = config.DEMUCS_MODEL) -> list:
    """Stem names the model produces, e.g. ['drums', 'bass', 'other', 'vocals']."""
    return list(_get(model_name).sources)


def separate(audio_path, out_dir=None, model_name: str = config.DEMUCS_MODEL,
             device: str | None = None, save: bool = True) -> dict:
    """Separate `audio_path` into stems.

    Returns {stem_name: ndarray (2, samples)} at the model's sample rate, plus
    writes `<out_dir>/<stem>.wav` when `save` is true.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _get(model_name)
    sr = model.samplerate

    wave = to_stereo(load_audio(audio_path, sr, mono=False))
    tensor = torch.from_numpy(wave)
    # Normalise as Demucs does, then undo it on the outputs.
    ref = tensor.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    tensor = (tensor - mean) / std

    with torch.no_grad():
        out = apply_model(model, tensor[None], device=device, shifts=config.DEMUCS_SHIFTS,
                          split=config.DEMUCS_SPLIT, overlap=config.DEMUCS_OVERLAP,
                          progress=False)[0]
    out = out * std + mean

    stems = {name: out[i].cpu().numpy() for i, name in enumerate(model.sources)}
    if save and out_dir is not None:
        for name, w in stems.items():
            save_wav(Path(out_dir) / f"{name}.wav", w, sr)
    return {"sample_rate": sr, "stems": stems}
