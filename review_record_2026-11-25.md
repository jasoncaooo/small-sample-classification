# Author and reviewer checks — 2026-11-25

These are two review perspectives applied by the same assistant, not an independent teacher grade.

## Author pass

Implemented 1,000 bootstrap resamples, top counts 10/30/50, original-index tie handling, frequency plots for all 150 features, largest-drop selection and fold-local predictive evaluation. Retained failed initial output and documented the QDA/LR protocol amendment before predictive scores existed.

## Reviewer findings and corrections

1. Full-feature QDA cannot run in this environment at the fixed configuration. Added fixed LR for the required subset-versus-all comparison, and reported QDA failure explicitly. No silent estimator substitution.
2. Full-data feature set must not define validation subsets. Verified training/validation row isolation and inspected the code path; selector receives only training rows.
3. Five repeats are dependent. Reported descriptive SD/range, not inferential confidence intervals or significance.
4. Largest gap is a heuristic. Reported actual boundary frequencies and weakening gap instead of claiming a universal natural threshold.
5. Stability does not establish predictive optimality. Report includes lower QDA results and lack of size-matched control.
6. Publication must not claim actual independent teacher approval, an unseen private score, or unaided authorship. Disclosure and limitations included.

## Verification

Independent Pearson calculation agrees with NumPy correlation; bootstrap is deterministic; inclusion frequencies sum to k; cutoff ties and flat-frequency edge cases pass. 75 selector audits have disjoint 200-row training/50-row validation sets. All 225 scored fits are finite and warning-free. All aggregate means match saved outputs. Rendered PDF has exactly two pages, with readable figures/table and all four required sections; both pages visually inspected.

## Remaining limits

No independent untouched test set, no size-matched control, no teacher grading guarantee. These do not prevent answering the specified extension but limit stronger claims. Ron's individual extension is outside this deliverable.
