"""
Canonical mapping: Qualtrics form (1–7) → ordered list of clip filenames.
Form k uses clip slots (k-1)*5 .. (k-1)*5+4 in the export (35 total slots).
First word in each stem = intended prosody (tone); second = linguistic content (words).
"""
from __future__ import annotations

# Filenames as used in acoustic batch / storage (.mp3 optional in lookups)
FORM_GROUPS: dict[int, tuple[str, ...]] = {
    1: (
        "neutralneutral1.mp3",
        "angryhappy.mp3",
        "calmfearful.mp3",
        "happyneutral.mp3",
        "sadneutral.mp3",
    ),
    2: (
        "neutralneutral2.mp3",
        "angrycalm.mp3",
        "disgustdisgust.mp3",
        "fearfulneutral.mp3",
        "happysad.mp3",
    ),
    3: (
        "neutralneutral3.mp3",
        "angryneutral.mp3",
        "calmangry.mp3",
        "fearfulfearful.mp3",
        "neutralhappy.mp3",
    ),
    4: (
        "neutralneutral4.mp3",
        "angryangry.mp3",
        "calmneutral.mp3",
        "happyhappy.mp3",
        "neutralsad.mp3",
    ),
    5: (
        "neutralneutral5.mp3",
        "disgustneutral.mp3",
        "fearfulcalm.mp3",
        "happyangry.mp3",
        "sadsad.mp3",
    ),
    6: (
        "neutralneutral6.mp3",
        "calmcalm.mp3",
        "neutraldisgust.mp3",
        "neutralfearful.mp3",
        "sadhappy.mp3",
    ),
    7: (
        "neutralcalm.mp3",
        "neutralangry.mp3",
        "angryhappy.mp3",
        "calmfearful.mp3",
        "happyneutral.mp3",
    ),
}

FL_COLUMN_TO_FORM = {
    "FL_69_DO_FL_22": 1,
    "FL_69_DO_FL_62": 2,
    "FL_69_DO_FL_63": 3,
    "FL_69_DO_FL_64": 4,
    "FL_69_DO_FL_65": 5,
    "FL_69_DO_FL_66": 6,
    "FL_69_DO_FL_67": 7,
}
