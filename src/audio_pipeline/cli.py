import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow's startup log noise

from . import config


def main(argv=None):
    p = argparse.ArgumentParser(prog="audio_pipeline",
                                description="Demucs separation + YAMNet classification")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="separate and classify (full pipeline)")
    r.add_argument("audio", nargs="+", help="audio file(s): wav, mp3, flac, ogg ...")
    r.add_argument("--out", default=str(config.OUTPUT_DIR))
    r.add_argument("--model", default=config.DEMUCS_MODEL,
                   help="htdemucs (4 stems), htdemucs_6s (adds guitar/piano), htdemucs_ft, mdx_extra")
    r.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    r.add_argument("--stems", nargs="+", help="only classify these stems, e.g. vocals")

    s = sub.add_parser("separate", help="separation only")
    s.add_argument("audio")
    s.add_argument("--out", default=str(config.OUTPUT_DIR))
    s.add_argument("--model", default=config.DEMUCS_MODEL)
    s.add_argument("--device", default=None)

    c = sub.add_parser("classify", help="classify existing audio file(s) with YAMNet only")
    c.add_argument("audio", nargs="+")

    a = p.parse_args(argv)
    files = a.audio if isinstance(a.audio, list) else [a.audio]
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        p.error(f"no such audio file: {', '.join(missing)}")
    try:
        _run(a)
    except (FileNotFoundError, ValueError) as e:
        p.exit(1, f"error: {e}\n")


def _run(a):
    if a.cmd == "run":
        from .pipeline import run
        for f in a.audio:
            run(f, a.out, a.model, a.device, a.stems)
    elif a.cmd == "separate":
        from .separate import separate
        out = Path(a.out) / Path(a.audio).stem / "stems"
        separate(a.audio, out, a.model, a.device)
        print(f"Stems written to {out}")
    else:
        from .audio_io import load_audio
        from .classify import classify_stem
        for f in a.audio:
            wave = load_audio(f, config.YAMNET_SAMPLE_RATE, mono=True)
            r = classify_stem(wave, config.YAMNET_SAMPLE_RATE)
            top = ", ".join(f"{c['label']} {c['score']:.2f}" for c in r["top_classes"])
            print(f"{f}: {'silent' if r['silent'] else top}")


if __name__ == "__main__":
    main()
