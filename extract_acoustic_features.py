"""
Extract acoustic features from TTS clips using librosa.
Batch-processes audio with validation.
Output: acoustic_features.csv (merge with survey CSV later by clip_id/filename).
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[misc, assignment]

SR = 22050
HOP = 512
N_FFT = 2048
N_MFCC = 13

REQUIRED_COLUMNS = [
    "clip_id", "filename", "duration_s", "rms_mean", "rms_std", "rms_max",
    "f0_mean_hz", "f0_std_hz", "tempo_bpm", "zcr_mean", "zcr_std",
    "spectral_centroid_mean", "spectral_rolloff_mean", "spectral_bandwidth_mean",
]


def extract_features_from_audio(
    y: np.ndarray,
    sr: int,
    clip_id: str = "clip",
    filename: Optional[str] = None,
) -> dict:
    """Extract acoustic features from raw audio arrays (for batch use and testing)."""
    if librosa is None:
        raise ImportError("Install librosa: pip install librosa soundfile")

    filename = filename or f"{clip_id}.mp3"

    duration_s = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms))

    f0, _ = librosa.piptrack(y=y, sr=sr, hop_length=HOP)
    f0_vals = f0[f0 > 0]
    f0_mean = float(np.mean(f0_vals)) if len(f0_vals) else np.nan
    f0_std = float(np.std(f0_vals)) if len(f0_vals) else np.nan
    f0_min = float(np.min(f0_vals)) if len(f0_vals) else np.nan
    f0_max = float(np.max(f0_vals)) if len(f0_vals) else np.nan

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    S = np.abs(librosa.stft(y, hop_length=HOP, n_fft=N_FFT))
    spectral_centroid = librosa.feature.spectral_centroid(S=S)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    sc_mean = float(np.mean(spectral_centroid))
    sc_std = float(np.std(spectral_centroid))
    rolloff_mean = float(np.mean(spectral_rolloff))
    rolloff_std = float(np.std(spectral_rolloff))
    bandwidth_mean = float(np.mean(spectral_bandwidth))
    bandwidth_std = float(np.std(spectral_bandwidth))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    out = {
        "clip_id": clip_id,
        "filename": filename,
        "duration_s": duration_s,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_max,
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_min_hz": f0_min,
        "f0_max_hz": f0_max,
        "tempo_bpm": tempo,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "spectral_centroid_mean": sc_mean,
        "spectral_centroid_std": sc_std,
        "spectral_rolloff_mean": rolloff_mean,
        "spectral_rolloff_std": rolloff_std,
        "spectral_bandwidth_mean": bandwidth_mean,
        "spectral_bandwidth_std": bandwidth_std,
    }
    for i in range(N_MFCC):
        out[f"mfcc{i+1}_mean"] = float(mfcc_means[i])
        out[f"mfcc{i+1}_std"] = float(mfcc_stds[i])
    return out


def extract_features(audio_path: Path) -> dict:
    """Load one audio file and extract per-clip acoustic features."""
    if librosa is None:
        raise ImportError("Install librosa: pip install librosa soundfile")

    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    return extract_features_from_audio(
        y, sr,
        clip_id=audio_path.stem,
        filename=audio_path.name,
    )


def validate_input(path: Path) -> tuple[bool, str]:
    """Validate an input audio file before processing. Returns (ok, message)."""
    if not path.exists():
        return False, "file not found"
    if not path.is_file():
        return False, "not a file"
    if path.stat().st_size == 0:
        return False, "empty file"
    if path.suffix.lower() not in (".mp3", ".wav", ".flac", ".ogg"):
        return False, f"unsupported extension {path.suffix}"
    return True, ""


def validate_output(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate extracted feature DataFrame. Returns (ok, list of error messages)."""
    errors = []
    if len(df) == 0:
        errors.append("empty DataFrame")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"missing column: {col}")
    if "clip_id" in df.columns and df["clip_id"].duplicated().any():
        errors.append("duplicate clip_id")
    return (len(errors) == 0, errors)


def main():
    if librosa is None:
        raise SystemExit("Install librosa: pip install librosa soundfile")

    parser = argparse.ArgumentParser(description="Extract acoustic features from audio (batch + validation).")
    parser.add_argument("--clips-dir", type=Path, default=None, help="Folder containing .mp3 files (default: ./clips)")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path (default: ./acoustic_features.csv)")
    parser.add_argument("--no-validate", action="store_true", help="Skip output validation")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    clips_dir = args.clips_dir or (base / "clips")
    out_path = args.out or (base / "acoustic_features.csv")

    if not clips_dir.is_dir():
        raise SystemExit(f"Clips folder not found: {clips_dir}")

    paths = sorted(clips_dir.glob("*.mp3"))
    if not paths:
        raise SystemExit(f"No .mp3 files in {clips_dir}")

    rows = []
    for p in paths:
        ok, msg = validate_input(p)
        if not ok:
            print(f"  SKIP {p.name}: validation failed ({msg})")
            continue
        try:
            rows.append(extract_features(p))
            print(f"  {p.name}")
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")

    df = pd.DataFrame(rows)
    if not args.no_validate:
        ok, errs = validate_output(df)
        if not ok:
            for e in errs:
                print(f"  Validation error: {e}")
            raise SystemExit("Output validation failed")

    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print("Merge with your survey CSV later on clip_id or filename.")


if __name__ == "__main__":
    main()
