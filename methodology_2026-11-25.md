# Experiment notes — 2026-11-25

## Completed shared work

Baseline, fold-local Pearson selection, candidate search, nested evaluation and final submission reproduction. Candidate and outer-fold aggregate scores are archived in the dated CSV files. Source files are preserved; only the portfolio reproduction entry switches the reference CSV comparison to its recorded SHA-256, avoiding redistribution of predictions.

## Interpretation

Correlation filtering measures marginal association, not causality or all interactions. QDA accommodates class-specific covariance structures; good scores do not prove that the data follow its assumptions. `tol` is a numerical rank threshold, not regularization strength. A retained source comment mentioning a sigmoid is stale: the actual export uses `rankdata` on decision scores.

## Planned individual extension 4 — not completed

Question: are marginal feature rankings stable, and are stable features useful predictors?
Protocol proposal: 1,000 bootstrap resamples of 250 rows, feature counts 10/30/50, selection frequency plots and chance references. Define a reproducible cutoff before using model scores. Compare stable features, fold-local top-30 and all features with matched validation splits and fixed QDA settings. All supervised bootstrap selection must be repeated inside each training fold for predictive evaluation; a full-data descriptive plot cannot define the CV feature set.
Results: pending. Conclusion: pending. The eventual individual report must use Question, Protocol, Results (table/figure), Conclusion and occupy at most two pages. Do not claim a stability or performance improvement before running the study.
