"""Create a short synthetic mixture (bass line + drum-like clicks + a vocal-ish tone)
so the pipeline can be tried without sourcing any copyrighted audio.

    python scripts/make_demo_audio.py [out.wav]   ->   input/demo_mix.wav by default

For meaningful results use a real song or speech recording.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR, DUR = 44100, 12
t = np.arange(SR * DUR) / SR
rng = np.random.default_rng(0)

bass = 0.35 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
beat = np.zeros_like(t)
for start in np.arange(0, DUR, 0.5):                      # kick/snare-like bursts
    i = int(start * SR)
    n = int(0.08 * SR)
    beat[i:i + n] += 0.6 * rng.standard_normal(n) * np.exp(-np.linspace(0, 8, n))
melody_f = 220 * 2 ** (np.floor(t * 2) % 5 / 12)          # wobbling "vocal" line
voice = 0.3 * sum(np.sin(2 * np.pi * melody_f * k * t) / k for k in (1, 2, 3, 4))

mix = bass + beat + voice
mix = 0.8 * mix / np.max(np.abs(mix))
default = Path(__file__).resolve().parents[1] / "input" / "demo_mix.wav"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else default   # optional output path
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, np.stack([mix, mix], axis=1), SR)
print("wrote", out)
