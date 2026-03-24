# How good is this analysis?

This file explains your current results in plain language.

## Short answer

- The analysis is **good for a pilot study** and the methods are appropriate.
- The evidence says **tone matters at least as much as words, and often more** in this dataset.
- The ML model is **useful but not final**: it shows real signal, but you should treat it as exploratory until you add stronger validation.

## What the stats are doing

The main tests are **linear mixed models**:

- Outcome = one rating (`naturalness`, `trust`, `emotion_strength`, `tone_vs_words`)
- Predictor = `design_cell` (the 4 tone/word conditions)
- Random effect = participant (`response_id`)

This is the correct idea because each person rated multiple clips, so rows are not independent.

The report compares each full model to a null model with a likelihood-ratio test (LRT).

## Is the analysis good?

### Strong parts

- Repeated-measures structure is handled (mixed models, not just plain ANOVA).
- Results are internally consistent: large effects appear in both mixed-model and supplementary tests.
- You have a complete long-format pipeline from raw survey export to analysis outputs.

### Limits to keep in mind

- One model has a warning: `naturalness` shows **Converged: No** in the mixed-model summary, so interpret that block cautiously.
- Dataset size is still moderate (`N=59` participants, `295` clip ratings), so some estimates will move as you collect more data.
- The supplementary ANOVA/Kruskal tests ignore repeated measures; they are helpful checks, but mixed models are the primary evidence.

## Do tone or words matter more?

### What your `tone_vs_words` rating indicates

The survey scale is:

- 1 = only words
- 3 = both equally
- 5 = only tone

In your current data:

- Overall mean `tone_vs_words` = **3.373** (leans toward tone over words)
- By condition:
  - `tone_emotional_word_neutral`: **3.655** (strongest tone-lean)
  - `tone_emotional_word_emotional`: **3.444**
  - `tone_neutral_word_neutral`: **3.292**
  - `tone_neutral_word_emotional`: **2.982** (roughly balanced / slight words lean)

Interpretation: when tone is emotional, people rely on tone more. When tone is neutral and words are emotional, the balance shifts toward words.

### What significance tests say

- Mixed model for `tone_vs_words`: LRT p = **0.0865** (suggestive, not conventionally significant at 0.05)
- Supplementary one-way tests:
  - ANOVA p = **0.0106**
  - Kruskal p = **0.0230**

Because mixed models are the better test for repeated measures, the safest claim is:

> The data trend toward tone playing a larger role, with strongest evidence when prosody is emotional. This pattern is promising but should be confirmed with more participants.

## What is the model doing exactly?

The ML model predicts **`perceived_emotion`** from acoustic features only.

### Inputs

- Numeric acoustic columns from `survey_long_ratings.csv` (RMS, F0 stats, spectral stats, MFCC stats, etc.)
- No participant ID and no text labels as predictors.

### Steps

1. Keep rows with valid `perceived_emotion`.
2. Convert features to numeric and drop rows with missing feature values.
3. Encode emotion labels into class IDs.
4. Stratified train/test split (75% train, 25% test).
5. Train gradient boosting classifier (default backend: `sklearn_hist_gbrt`).
6. Evaluate on held-out test set:
   - accuracy
   - per-class precision/recall/F1
   - confusion matrix
   - feature-importance plot

### Current model quality

- Test accuracy = **0.608** on 7 classes (above chance baseline ~0.143)
- Best recall in this run: `neutral` (0.83), `fearful` (0.73)
- Weaker classes include `calm` and `angry` recall in this split

This is a good pilot signal that acoustics carry emotional information, but it is not yet a production-level classifier.

## Practical conclusion for your project write-up

You can currently claim:

1. Your pipeline successfully disentangles tone/word conditions and analyzes repeated measures correctly.
2. Emotional prosody has a strong relationship with perceived intensity and trust.
3. Tone appears to matter at least as much as words on average, with condition-dependent effects.
4. Acoustic features can predict perceived emotion above chance, supporting the feasibility of your ML component.

And you should add this caveat:

> These are interim results from a moderate sample and one held-out split; further data collection and stronger validation (e.g., grouped cross-validation) are planned.

