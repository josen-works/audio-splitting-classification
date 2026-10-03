"""End-to-end: audio file -> Demucs stems -> YAMNet labels -> reports."""
from pathlib import Path

import librosa

from . import config
from .classify import classify_stem, stem_level
from .report import print_summary, write_reports
from .separate import model_sources, separate


def run(audio_path, out_root=config.OUTPUT_DIR, model_name: str = config.DEMUCS_MODEL,
        device: str | None = None, only_stems: list | None = None, verbose: bool = True) -> dict:
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"no such audio file: {audio_path}")
    if only_stems:  # check before the (slow) separation, not after
        available = model_sources(model_name)
        unknown = [s for s in only_stems if s not in available]
        if unknown:
            raise ValueError(f"unknown stem(s) {unknown} for {model_name}; available: {available}")
    out_dir = Path(out_root) / audio_path.stem
    stem_dir = out_dir / "stems"

    if verbose:
        print(f"[1/3] Separating {audio_path.name} with {model_name} ...")
    sep = separate(audio_path, stem_dir, model_name=model_name, device=device)
    sr, stems = sep["sample_rate"], sep["stems"]

    if verbose:
        print("[2/3] Classifying stems with YAMNet ...")
    loudest = max(stem_level(w) for w in stems.values())
    classified = {name: classify_stem(w, sr, reference_rms=loudest) for name, w in stems.items()
                  if not only_stems or name in only_stems}

    results = {
        "input": str(audio_path),
        "duration_s": float(librosa.get_duration(path=str(audio_path))),
        "demucs_model": model_name,
        "stems": classified,
    }
    if verbose:
        print("[3/3] Writing reports ...")
    paths = write_reports(results, out_dir)
    results["outputs"] = {k: str(v) for k, v in paths.items()} | {"stems_dir": str(stem_dir)}
    if verbose:
        print_summary(results)
        print(f"\nResults in {out_dir}")
    return results
