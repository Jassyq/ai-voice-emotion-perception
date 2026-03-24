"""
Parse Qualtrics survey export into long-format ratings (one row per clip × participant).
Joins with stimulus_groups.FORM_GROUPS so each rating is tied to filename, tone, and word content.
Uses stdlib only for portability (merge optional acoustic CSV without pandas).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Optional

from stimulus_groups import FL_COLUMN_TO_FORM, FORM_GROUPS

# Emotion lexicon in filenames (longest first for greedy split)
_EMOTION_LEXICON = (
    "neutral",
    "fearful",
    "disgust",
    "happy",
    "angry",
    "calm",
    "sad",
)


def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def parse_likert_value(raw: str) -> Optional[int]:
    """Map Qualtrics text like '3 = Both equally' or '2  = Weak' to integer 1–5."""
    if _is_blank(raw):
        return None
    s = str(raw).strip()
    m = re.match(r"^(\d)", s)
    if m:
        return int(m.group(1))
    return None


def parse_emotion_label(raw: str) -> Optional[str]:
    if _is_blank(raw):
        return None
    return str(raw).strip().lower()


def parse_filename_tone_words(filename: str) -> tuple[str, str, Optional[str]]:
    """
    Split stem into (tone_emotion, word_emotion, variant).
    e.g. neutralneutral1.mp3 -> neutral, neutral, '1'
    angryhappy.mp3 -> angry, happy, None
    """
    stem = Path(filename).stem.lower()
    variant: Optional[str] = None
    m = re.match(r"^(.+?)(\d+)$", stem)
    if m:
        stem, variant = m.group(1), m.group(2)
    for tone in sorted(_EMOTION_LEXICON, key=len, reverse=True):
        if stem.startswith(tone):
            rest = stem[len(tone) :]
            for word in sorted(_EMOTION_LEXICON, key=len, reverse=True):
                if rest == word:
                    return tone, word, variant
    raise ValueError(f"cannot parse tone/words from filename: {filename!r}")


def design_cell(tone: str, words: str) -> str:
    """Four-way manipulation for ANOVA / reporting (neutral = linguistically or prosodically baseline)."""
    tn = tone == "neutral"
    wn = words == "neutral"
    if not tn and not wn:
        return "tone_emotional_word_emotional"
    if not tn and wn:
        return "tone_emotional_word_neutral"
    if tn and not wn:
        return "tone_neutral_word_emotional"
    return "tone_neutral_word_neutral"


def infer_form_from_row(row: dict[str, Any], fl_columns: list[str]) -> Optional[int]:
    for col in fl_columns:
        v = row.get(col)
        if v is not None and str(v).strip() == "1":
            return FL_COLUMN_TO_FORM.get(col)
    return None


def _clip_column_groups(header: list[str]) -> tuple[list[str], list[list[str]]]:
    """Return emotion column name and list of 5 column names per clip (35 clips)."""
    start = header.index("QID1")
    end = header.index("FL_69_DO_FL_22")
    qcols = header[start:end]
    groups: list[list[str]] = []
    for i in range(0, len(qcols), 5):
        chunk = qcols[i : i + 5]
        if len(chunk) < 5:
            break
        groups.append(chunk)
    emotion_cols = [g[0] for g in groups]
    return emotion_cols, groups


def parse_qualtrics_export(
    csv_path: Path,
    *,
    only_finished: bool = True,
) -> list[dict[str, Any]]:
    """
    Read a Qualtrics CSV (with two metadata rows after the header) and return list of row dicts.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
        next(f)
        next(f)
        reader = csv.DictReader(f, fieldnames=header)
        rows = list(reader)

    fl_columns = list(FL_COLUMN_TO_FORM.keys())
    _, clip_groups = _clip_column_groups(header)

    out: list[dict[str, Any]] = []
    for row in rows:
        if only_finished:
            fin = str(row.get("Finished", "")).strip().lower()
            if fin not in ("true", "1", "yes"):
                continue
        form = infer_form_from_row(row, fl_columns)
        if form is None:
            continue
        filenames = FORM_GROUPS.get(form)
        if not filenames:
            continue
        slot0 = (form - 1) * 5
        for j, fname in enumerate(filenames):
            slot = slot0 + j
            if slot >= len(clip_groups):
                break
            cols = clip_groups[slot]
            emo_raw = row.get(cols[0], "")
            if _is_blank(emo_raw):
                continue
            tone, words, variant = parse_filename_tone_words(fname)
            out.append(
                {
                    "response_id": row.get("ResponseId", ""),
                    "start_date": row.get("StartDate", ""),
                    "duration_sec": row.get("Duration (in seconds)", ""),
                    "finished": row.get("Finished", ""),
                    "form": form,
                    "clip_order_in_form": j + 1,
                    "clip_slot_index": slot,
                    "filename": fname,
                    "stem": Path(fname).stem,
                    "intended_tone": tone,
                    "intended_words": words,
                    "filename_variant": variant or "",
                    "design_cell": design_cell(tone, words),
                    "perceived_emotion": parse_emotion_label(emo_raw),
                    "emotion_strength": parse_likert_value(row.get(cols[1], "")),
                    "tone_vs_words": parse_likert_value(row.get(cols[2], "")),
                    "naturalness": parse_likert_value(row.get(cols[3], "")),
                    "trust": parse_likert_value(row.get(cols[4], "")),
                }
            )

    return out


def _read_acoustic_dict(acoustic_csv: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Index acoustic features by clip_id (stem)."""
    with acoustic_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        if "clip_id" not in fieldnames:
            raise ValueError("acoustic CSV must have clip_id")
        by_id: dict[str, dict[str, str]] = {}
        for row in r:
            cid = row.get("clip_id", "")
            if cid:
                by_id[cid] = row
    return fieldnames, by_id


def merge_with_acoustic(
    ratings_long: list[dict[str, Any]],
    acoustic_csv: Path,
) -> list[dict[str, Any]]:
    """Left join long ratings to acoustic_features.csv on clip stem == clip_id."""
    _, by_id = _read_acoustic_dict(acoustic_csv)
    merged: list[dict[str, Any]] = []
    for row in ratings_long:
        mrow = dict(row)
        stem = row.get("stem", "")
        ac = by_id.get(stem, {})
        for k, v in ac.items():
            if k == "clip_id":
                continue
            key = k if k not in mrow else f"acoustic_{k}"
            mrow[key] = v
        merged.append(mrow)
    return merged


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualtrics export → long-format ratings CSV")
    parser.add_argument("survey_csv", type=Path, help="Qualtrics export .csv")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: survey_long_ratings.csv next to input)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include rows where Finished is not True",
    )
    parser.add_argument(
        "--acoustic",
        type=Path,
        default=None,
        help="Optional acoustic_features.csv to merge for modeling",
    )
    args = parser.parse_args()

    out_path = args.out or (args.survey_csv.parent / "survey_long_ratings.csv")
    rows = parse_qualtrics_export(args.survey_csv, only_finished=not args.include_incomplete)
    if args.acoustic:
        rows = merge_with_acoustic(rows, args.acoustic)
    write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
