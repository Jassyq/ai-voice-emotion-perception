"""
Unit tests for audio feature extraction.
Uses PyTest and mocked inputs for isolated testing (no real audio files).
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import from project root (run as: pytest tests/ or python -m pytest)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract_acoustic_features import (
    extract_features_from_audio,
    extract_features,
    validate_input,
    validate_output,
    REQUIRED_COLUMNS,
    SR,
    N_MFCC,
)


def _make_fake_audio(seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate a short synthetic signal (sine + noise) for isolated tests."""
    n = int(sr * seconds)
    t = np.linspace(0, seconds, n, dtype=np.float32)
    y = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(n).astype(np.float32)
    return y


class TestExtractFeaturesFromAudio:
    """Tests for extract_features_from_audio using synthetic arrays (no file I/O)."""

    def test_returns_dict_with_required_keys(self):
        y = _make_fake_audio(1.0)
        result = extract_features_from_audio(y, SR, clip_id="test_clip")
        assert isinstance(result, dict)
        for col in REQUIRED_COLUMNS:
            assert col in result, f"missing key {col}"

    def test_clip_id_and_filename_set_correctly(self):
        y = _make_fake_audio(0.5)
        result = extract_features_from_audio(y, SR, clip_id="angry_calm", filename="angry_calm.mp3")
        assert result["clip_id"] == "angry_calm"
        assert result["filename"] == "angry_calm.mp3"

    def test_default_filename_from_clip_id(self):
        y = _make_fake_audio(0.5)
        result = extract_features_from_audio(y, SR, clip_id="my_clip")
        assert result["filename"] == "my_clip.mp3"

    def test_duration_positive_and_reasonable(self):
        y = _make_fake_audio(2.0)
        result = extract_features_from_audio(y, SR, clip_id="t")
        assert result["duration_s"] > 0
        assert 1.5 <= result["duration_s"] <= 2.5

    def test_rms_and_tempo_numeric(self):
        y = _make_fake_audio(1.0)
        result = extract_features_from_audio(y, SR, clip_id="t")
        assert isinstance(result["rms_mean"], (int, float))
        assert result["rms_mean"] >= 0
        assert isinstance(result["tempo_bpm"], (int, float))
        assert result["tempo_bpm"] > 0

    def test_mfcc_columns_present(self):
        y = _make_fake_audio(1.0)
        result = extract_features_from_audio(y, SR, clip_id="t")
        for i in range(1, N_MFCC + 1):
            assert f"mfcc{i}_mean" in result
            assert f"mfcc{i}_std" in result

    def test_spectral_features_present(self):
        y = _make_fake_audio(1.0)
        result = extract_features_from_audio(y, SR, clip_id="t")
        assert "spectral_centroid_mean" in result
        assert "spectral_rolloff_mean" in result
        assert "spectral_bandwidth_mean" in result
        assert result["spectral_centroid_mean"] >= 0


class TestExtractFeaturesWithMockedLibrosa:
    """Tests for extract_features(audio_path) with mocked librosa.load."""

    @patch("extract_acoustic_features.librosa.load")
    def test_calls_librosa_load_with_path(self, mock_load):
        mock_load.return_value = (_make_fake_audio(1.0), SR)
        path = Path("/fake/clips/angryangry.mp3")
        extract_features(path)
        mock_load.assert_called_once()
        call_args = mock_load.call_args[0]
        assert str(path) in str(call_args[0]) or path == call_args[0]

    @patch("extract_acoustic_features.librosa.load")
    def test_returns_same_structure_as_from_audio(self, mock_load):
        y = _make_fake_audio(1.0)
        mock_load.return_value = (y, SR)
        path = Path("/fake/test.mp3")
        result = extract_features(path)
        assert result["clip_id"] == "test"
        assert result["filename"] == "test.mp3"
        for col in REQUIRED_COLUMNS:
            assert col in result


class TestValidateInput:
    """Tests for input validation (no real files required; use mocks or temp paths)."""

    def test_nonexistent_file_fails(self, tmp_path):
        p = tmp_path / "missing.mp3"
        ok, msg = validate_input(p)
        assert ok is False
        assert "not found" in msg or "missing" in msg.lower()

    def test_empty_file_fails(self, tmp_path):
        p = tmp_path / "empty.mp3"
        p.write_bytes(b"")
        ok, msg = validate_input(p)
        assert ok is False
        assert "empty" in msg.lower()

    def test_valid_extension_accepts(self, tmp_path):
        p = tmp_path / "clip.mp3"
        p.write_bytes(b"x")  # minimal non-empty
        ok, msg = validate_input(p)
        assert ok is True
        assert msg == ""


class TestValidateOutput:
    """Tests for output DataFrame validation."""

    def test_empty_dataframe_fails(self):
        df = pd.DataFrame()
        ok, errs = validate_output(df)
        assert ok is False
        assert any("empty" in e.lower() for e in errs)

    def test_missing_column_fails(self):
        df = pd.DataFrame([{"clip_id": "a", "duration_s": 1.0}])
        ok, errs = validate_output(df)
        assert ok is False
        assert any("missing" in e.lower() for e in errs)

    def test_duplicate_clip_id_fails(self):
        row = {
            "clip_id": "a", "filename": "a.mp3", "duration_s": 1, "rms_mean": 0.1,
            "rms_std": 0, "rms_max": 0.2, "f0_mean_hz": 200, "f0_std_hz": 10,
            "tempo_bpm": 120, "zcr_mean": 0.1, "zcr_std": 0,
            "spectral_centroid_mean": 1000, "spectral_rolloff_mean": 2000, "spectral_bandwidth_mean": 1500,
        }
        df = pd.DataFrame([row, {**row, "filename": "a2.mp3"}])
        ok, errs = validate_output(df)
        assert ok is False
        assert any("duplicate" in e.lower() for e in errs)

    def test_valid_output_passes(self):
        df = pd.DataFrame([{
            "clip_id": "x", "filename": "x.mp3", "duration_s": 1.0, "rms_mean": 0.05,
            "rms_std": 0.02, "rms_max": 0.1, "f0_mean_hz": 200, "f0_std_hz": 20,
            "tempo_bpm": 100, "zcr_mean": 0.1, "zcr_std": 0.05,
            "spectral_centroid_mean": 2000, "spectral_rolloff_mean": 4000, "spectral_bandwidth_mean": 2000,
        }])
        ok, errs = validate_output(df)
        assert ok is True
        assert len(errs) == 0
