from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"

# Demucs
DEMUCS_MODEL = "htdemucs"          # 4 stems: drums, bass, other, vocals
# "htdemucs_6s" adds guitar and piano; "htdemucs_ft" is slower but higher quality.
DEMUCS_SHIFTS = 1                  # >1 averages random time shifts: better, slower
DEMUCS_OVERLAP = 0.25
DEMUCS_SPLIT = True                # chunk long tracks to bound memory use

# YAMNet
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16000         # the model requires 16 kHz mono
TOP_K = 5                          # classes reported per stem
FRAME_HOP_S = 0.48                 # YAMNet emits one prediction every 0.48 s
SILENCE_RMS = 1e-3                 # stems quieter than this are reported as silent
RELATIVE_SILENCE_DB = -26.0        # ...as are stems this far below the loudest stem (separation bleed)
