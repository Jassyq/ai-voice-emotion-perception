"""Tests for Qualtrics parsing and filename tone/word split."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from survey_parser import (
    design_cell,
    merge_with_acoustic,
    parse_filename_tone_words,
    parse_likert_value,
    parse_qualtrics_export,
)


class TestParseLikert:
    def test_prefix_digit(self):
        assert parse_likert_value("3 = Both equally") == 3
        assert parse_likert_value("2  = Weak") == 2
        assert parse_likert_value("1 = Very weak") == 1

    def test_empty(self):
        assert parse_likert_value("") is None
        assert parse_likert_value(None) is None


class TestParseFilename:
    def test_neutral_pair_with_variant(self):
        assert parse_filename_tone_words("neutralneutral1.mp3") == ("neutral", "neutral", "1")

    def test_concat_pair(self):
        assert parse_filename_tone_words("angryhappy.mp3") == ("angry", "happy", None)
        assert parse_filename_tone_words("calmfearful.mp3") == ("calm", "fearful", None)

    def test_design_cell(self):
        assert design_cell("angry", "happy") == "tone_emotional_word_emotional"
        assert design_cell("angry", "neutral") == "tone_emotional_word_neutral"
        assert design_cell("neutral", "happy") == "tone_neutral_word_emotional"
        assert design_cell("neutral", "neutral") == "tone_neutral_word_neutral"


class TestQualtricsIntegration:
    def test_export_produces_rows(self):
        root = Path(__file__).resolve().parent.parent
        p = root / "HDSI+Research+Project_March+23,+2026_16.22.csv"
        if not p.exists():
            pytest.skip("survey CSV not in repo")
        rows = parse_qualtrics_export(p, only_finished=True)
        assert len(rows) > 0
        assert "perceived_emotion" in rows[0]
        assert "design_cell" in rows[0]
        assert all(1 <= r["form"] <= 7 for r in rows)
        from collections import Counter

        c = Counter(r["response_id"] for r in rows)
        assert all(v == 5 for v in c.values())


class TestMergeAcoustic:
    def test_merge_adds_columns(self, tmp_path):
        ratings = [
            {"stem": "angryangry", "response_id": "a", "form": 4, "filename": "angryangry.mp3"},
        ]
        ac = tmp_path / "ac.csv"
        ac.write_text(
            "clip_id,filename,duration_s,rms_mean\nangryangry,angryangry.mp3,1,0.1\n",
            encoding="utf-8",
        )
        merged = merge_with_acoustic(ratings, ac)
        assert merged[0]["rms_mean"] == "0.1"
