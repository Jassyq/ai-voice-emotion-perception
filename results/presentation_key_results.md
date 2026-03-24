# Presentation Results (Easy-to-Explain Version)

## 1) Project question (simple)

This study asks:

- Do people judge emotion in AI voice mainly from **tone/prosody**?
- Or mainly from **words/content**?

We intentionally mixed these (emotional tone + neutral words, neutral tone + emotional words, etc.) to separate their effects.

## 2) Participants and data (important for slides)

- **70 total participants** completed or started the survey.
- **59 completed participants** were used in the main inferential analysis.
- Main analysis dataset: **295 clip ratings** (5 ratings per completed participant).

Why 59 in the main model?  
To keep statistical tests clean, we used completed responses with full clip blocks.

## 3) Outcomes we measured

Per clip, participants rated:

- perceived emotion category
- emotion strength
- tone vs words importance
- naturalness
- trustworthiness

## 4) Main findings (what matters most)

### A) Does condition matter?

Yes. The tone/word condition clearly affected:

- **Emotion strength** (strong effect)
- **Trust** (clear effect)

Naturalness did not show a strong overall condition effect in this run.

### B) Tone vs words: which mattered more?

On the `tone_vs_words` scale:

- 1 = only words
- 3 = both equally
- 5 = only tone

Observed means:

- Overall mean: **3.373** (leans toward tone)
- Emotional tone + neutral words: **3.655** (strongest tone reliance)
- Emotional tone + emotional words: **3.444**
- Neutral tone + neutral words: **3.292**
- Neutral tone + emotional words: **2.982** (closest to words mattering more)

Plain-language interpretation:

> People generally leaned toward **tone**, especially when tone carried emotion.  
> When tone was neutral but words were emotional, reliance shifted toward words.

## 5) Machine learning (what each model predicts)

We trained models using acoustic features (pitch, MFCC, intensity, spectral features, etc.).

### Model A: predict perceived emotion (human labels)

- Target: what participants said they heard (`perceived_emotion`)
- Split strategy: grouped by participant (`response_id`) to test new-participant generalization
- Accuracy: **0.640** (7 classes)
- Chance baseline: ~0.143

Interpretation:

> Acoustic features contain substantial signal for how humans perceive emotion.

### Model B: predict true clip tone

- Target: intended tone label (`intended_tone`)
- Split strategy: grouped by clip (`stem`) to avoid clip leakage
- Accuracy: **0.632**

Interpretation:

> Tone category is acoustically learnable from the generated voice clips.

### Model C: predict true words-emotion label

- Target: intended words label (`intended_words`)
- Split strategy: grouped by clip (`stem`)
- Accuracy: **0.803**

Interpretation:

> Acoustic patterns also correlate with intended word-emotion classes in this dataset.
> (Class balance in held-out groups is uneven, so this should be interpreted carefully.)

## 6) Important modeling fix (mention this briefly)

Earlier we saw unrealistically high performance (1.000) due to leakage:

- same clip appeared in both train and test through different rows

This has been fixed with grouped splits.  
Current numbers above are the corrected ones.

## 7) What we removed from this presentation version

To keep the presentation clear, we removed:

- long coefficient tables
- full per-class precision/recall tables
- extra supplementary tests that repeat the same story

Those details still exist in `results/summary.md` for appendix/Q&A.

## 8) Slide-ready conclusion

1. **Core answer:** Both tone and words matter, but tone often has the stronger influence.
2. **Strongest evidence:** Condition robustly changes perceived emotion strength and trust.
3. **ML support:** Acoustics predict perceived emotion well above chance and also predict intended labels.
4. **Practical implication:** Emotional prosody design is a key lever for trustworthy, human-centered AI voice systems.
5. **Next phase:** Add more complete participants and keep leakage-safe validation for final reporting.

