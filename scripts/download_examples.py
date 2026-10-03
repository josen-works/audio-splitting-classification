"""Download two short, freely licensed recordings into input/ to try the pipeline on.

    python scripts/download_examples.py

- input/speech_libri.ogg  - LibriSpeech excerpt (CC BY 4.0)
- input/music_vibe_ace.ogg - "Vibe Ace" by Kevin MacLeod, incompetech.com (CC BY 3.0)

Both come from librosa's example data (https://librosa.org/data/audio/).
"""
import shutil
from pathlib import Path

import librosa

INPUT = Path(__file__).resolve().parents[1] / "input"
EXAMPLES = {"libri1": "speech_libri.ogg", "vibeace": "music_vibe_ace.ogg"}

INPUT.mkdir(exist_ok=True)
for key, name in EXAMPLES.items():
    src = librosa.ex(key)
    shutil.copy(src, INPUT / name)
    print(f"wrote {INPUT / name}")
