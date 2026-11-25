# Small-sample classification

Joint DSAI4203 coursework by **Cao Wei and Ron**. Portfolio prepared on **2026-11-25**.

A reproducible classification experiment on 250 labelled rows with 150 numerical features. The workflow studies feature filtering and model selection under limited training data, then generates scores for 19,750 test rows.

## Method

Absolute Pearson correlation selects 30 features → StandardScaler → quadratic discriminant analysis (QDA), with `reg_param=1e-6` and `tol=1e-10`. Feature selection and scaling are fitted inside each training fold. IDs are excluded from model inputs.

The final search compares an earlier RBF SVM against 20 QDA configurations (5 feature counts × 4 regularization values). Selection uses stratified 5-fold CV repeated 5 times. Nested evaluation uses 5 outer folds repeated twice, rerunning the full candidate search inside each outer training fold.

## Recorded results

| Evaluation | ROC-AUC | Interpretation |
|---|---:|---|
| Best candidate, selection CV | 0.99676 | Used for selection; potentially optimistic |
| Nested evaluation of search procedure | 0.99435 | Mean outer-fold AUC, not an untouched test set |
| Kaggle public leaderboard | 0.99241 | Archived 2026-11-23 submission; not rechecked live |

Private leaderboard score is unknown. AUC is not classification accuracy. Repeated folds are dependent; fold spread is not a confidence interval. Earlier experiments reused the same data, so nested CV does not erase all adaptive experimentation bias. The baseline all-feature logistic regression recorded mean CV AUC 0.75371 under its own splits; this is historical context, not a controlled estimate of the QDA improvement.

## Reproduce the final submission

Use Python 3.12 (original environment: 3.12.0). Obtain the competition data yourself from [Kaggle](https://www.kaggle.com/competitions/small-sample-classification-challenge/data), subject to its access requirements and rules. Put `small-sample-classification-challenge.zip` in this repository root; it must contain `train.csv`, `test.csv`, and `sample_submission.csv` at its archive root. Raw data and individual predictions are not distributed here.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_final_2026-11-25.py
```

On Windows activate with `.venv\Scripts\activate`. The script checks the generated CSV against the SHA-256 of the archived submission and refuses to overwrite existing output. Move prior output before rerunning. Hash equality verifies reproduction of that artifact, not leaderboard correctness.

To rerun the candidate search and nested evaluation:

```bash
python qda_precision_search_2026-11-23_v2.py
```

This is substantially slower than final-model reproduction. New results use dated, incremented directories. Scripts never submit predictions to Kaggle automatically. Historical filenames remain unchanged because imports depend on them. `public_auc=null` in a newly generated search summary means the run has not submitted anything, not that the archived public result is missing.

## Repository contents

- `reproduce_final_2026-11-25.py`: compact final-model reproduction and hash check.
- `qda_precision_search_2026-11-23_v2.py`: final candidate search and nested evaluation.
- `improve_2026-11-21.py`: reusable Pearson selector and earlier experiments.
- `baseline_2026-11-21.py`: initial baseline experiment.
- the dated CSV/JSON files: aggregate archived comparisons and score provenance.
- the dated Markdown files: methodology, contribution boundaries and planned extension.

## Contributions and AI assistance

The shared model-development project was completed jointly by Cao Wei and Ron. Individual ownership of historical components is not asserted. OpenAI Codex assisted with model design, code, experiment checking and documentation. Authors remain responsible for understanding and presenting the work.

Cao Wei's planned individual extension is bootstrap feature-ranking stability. **It has not been implemented or evaluated in this release**, and no performance improvement from it is claimed.

## Limitations

Small labelled sample; adaptive reuse of data; unknown private score; no calibrated probabilities. The export transforms decision scores to average ranks divided by `(n_test + 1)`, preserving ordering rather than estimating class probabilities. The generic QDA search includes small positive regularization, not zero regularization. No external training data are used.
