# ai-voice-emotion-study
Research Study for HDSI


## Project Overview
This project investigates how people perceive emotions in AI-generated speech by distinguishing the effects of linguistic content from vocal prosody (tone, pitch, rhythm). Using audio generated from ElevenLabs and Amazon Polly, we test whether emotional perception depends more on **what is said** or **how it is said**. Participants rate randomized audio clips on emotion, naturalness, and trust. Insights will guide the design of more human-centered and ethically responsible AI voice systems.

## Data pipeline (repo)

1. **Acoustic features** — Place `.mp3` clips in `clips/`, then run `python extract_acoustic_features.py` to produce `acoustic_features.csv` (librosa).
2. **Survey export** — Qualtrics CSVs include 35 clip slots (7 forms × 5 clips). The randomizer columns `FL_69_DO_FL_22` … `FL_69_DO_FL_67` encode which form was shown (form 1 = `FL_22`, …, form 7 = `FL_67`). Stimulus filenames per form are defined in `stimulus_groups.py`.
3. **Long-format ratings** — Run `python survey_parser.py your_export.csv --acoustic acoustic_features.csv -o survey_long_ratings.csv` to get one row per participant × clip, with `design_cell` encoding the four-way tone × words manipulation, plus merged acoustic columns when `--acoustic` is set.

**Status:** Feature extraction and parsing/merge are implemented.

**Analysis:** Run `python analysis/run_analysis.py` from the repo root (after `pip install -r requirements.txt`). This writes `results/summary.md`, `results/summary.html`, and figures (`xgb_confusion_matrix.png`, `xgb_feature_importance.png`). Open `results/summary.html` in a browser or read the Markdown. The script fits **linear mixed models** (participant random intercept) for Likert outcomes vs. `design_cell`, adds **one-way ANOVA / Kruskal–Wallis** as a clearly labeled supplementary check, and trains a **gradient boosting** classifier for `perceived_emotion` from acoustic features (sklearn by default; pass `--use-xgb` if XGBoost works on your machine).

**Not in repo:** pyAudioAnalysis-only features (librosa is used here).
