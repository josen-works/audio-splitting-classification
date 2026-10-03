"""Full pipeline on a synthetic clip. Downloads Demucs and YAMNet weights on first run."""
import runpy
from pathlib import Path

import pytest

from audio_pipeline.pipeline import run

pytestmark = pytest.mark.slow


def test_full_pipeline(tmp_path, monkeypatch):
    audio = tmp_path / "demo_mix.wav"
    monkeypatch.setattr("sys.argv", ["make_demo_audio.py", str(audio)])
    runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "make_demo_audio.py"))
    res = run(audio, out_root=tmp_path, verbose=False)
    assert set(res["stems"]) == {"drums", "bass", "other", "vocals"}
    for stem in res["stems"].values():
        assert stem["silent"] or len(stem["top_classes"]) == 5
    for stem_name in res["stems"]:
        assert (tmp_path / "demo_mix" / "stems" / f"{stem_name}.wav").exists()
    assert (tmp_path / "demo_mix" / "results.json").exists()
