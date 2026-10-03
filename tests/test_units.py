"""Fast tests that need neither Demucs nor YAMNet weights."""
import numpy as np

from audio_pipeline import config
from audio_pipeline.audio_io import rms, to_mono_16k, to_stereo
from audio_pipeline.classify import classify_stem, summarize


def test_to_stereo_duplicates_mono():
    assert to_stereo(np.zeros((1, 100), np.float32)).shape == (2, 100)
    assert to_stereo(np.zeros((6, 100), np.float32)).shape == (2, 100)


def test_to_mono_16k_resamples():
    wave = np.random.default_rng(0).standard_normal((2, 44100)).astype(np.float32)
    out = to_mono_16k(wave, 44100)
    assert out.ndim == 1 and abs(len(out) - 16000) <= 1 and out.dtype == np.float32


def test_summarize_ranks_by_mean_score():
    names = ["a", "b", "c"]
    scores = np.array([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2]])
    s = summarize(scores, names, top_k=2)
    assert s["top_classes"][0]["label"] == "b"
    assert len(s["timeline"]) == 2 and s["timeline"][1]["time_s"] == config.FRAME_HOP_S


def test_silent_stem_is_flagged_without_loading_model():
    r = classify_stem(np.zeros((2, 44100 * 3), np.float32), 44100)
    assert r["silent"] and r["top_classes"] == []
    assert rms(np.zeros(10)) == 0.0


def test_quiet_stem_relative_to_mix_is_flagged_as_bleed():
    t = np.arange(44100 * 2) / 44100
    faint = (0.003 * np.sin(2 * np.pi * 440 * t))[None].repeat(2, 0).astype(np.float32)
    r = classify_stem(faint, 44100, reference_rms=0.1)   # about -33 dB below the loudest stem
    assert r["silent"] and r["level_db_vs_loudest"] < config.RELATIVE_SILENCE_DB
