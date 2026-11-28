# Extension 1 Validation protocol sensitivity

**Ron’s assigned extension in the shared DSAI4203 project. Verified run: 2026-11-28.**

This experiment asks how much a fixed model’s estimated ROC-AUC changes when the same 250 labelled rows are split differently. It compares one stratified 20% hold-out with stratified 3-, 5- and 10-fold cross-validation, using 30 split seeds for each scheme. Ten-fold CV has the smallest observed spread here; this alone does **not** establish that a 0.01 AUC difference between two models is reliable.

中文导读：这里包含 Ron 的 Extension 1 对应代码、新版独立运行入口、实际运行结果和检查记录。原脚本已存在；本次补齐 GitHub 中缺失的内容并重新运行验证。结果是同一批数据反复划分的描述性统计，不是30份独立测试集，也不能把“极差小于0.01”当作模型差异显著的证明。

## Protocol

- Data: 250 rows, 150 numerical features; class counts 137 and 113. Only `train.csv` is read. IDs are excluded from the model.
- Fixed recipe: absolute Pearson top 30 → StandardScaler → QDA, with `reg_param=1e-6` and `tol=1e-10`. The recipe is fixed; fitted parameters and selected features are learned again from each training fold.
- Seeds: integers 0 through 29. The model has no random-state parameter; randomization comes from the row partitions. CV uses `shuffle=True`.
- Feature selection and scaling are inside the pipeline and fitted only on training rows.
- One estimate per seed: the hold-out validation AUC, or the arithmetic mean of that run’s fold AUCs. These are not pooled out-of-fold AUCs.
- Report the mean, sample SD (`ddof=1`), minimum, maximum and range of the 30 estimates. Fold scores are retained separately.

| Scheme | Training rows per fit | Validation rows per fit | Fits per seed | Total fits |
|---|---:|---:|---:|---:|
| 20% hold-out | 200 | 50 | 1 | 30 |
| 3-fold | 166–167 | 83–84 | 3 | 90 |
| 5-fold | 200 | 50 | 5 | 150 |
| 10-fold | 225 | 25 | 10 | 300 |

## Verified results

The 2026-11-28 run produced 120 estimates from 570 fits, with no captured estimator warnings. The four summaries match the figures used in the existing presentation when rounded to five decimals.

| Scheme | Mean AUC | SD across seeds | Minimum | Maximum | Range |
|---|---:|---:|---:|---:|---:|
| 20% hold-out | 0.99699 | 0.00480 | 0.98551 | 1.00000 | 0.01449 |
| 3-fold | 0.99188 | 0.00628 | 0.96851 | 0.99867 | 0.03015 |
| 5-fold | 0.99443 | 0.00426 | 0.97939 | 1.00000 | 0.02061 |
| 10-fold | 0.99679 | 0.00193 | 0.99291 | 0.99936 | 0.00645 |

Full precision: [summary CSV](results_2026-11-28/summary_2026-11-28.csv), [120 seed estimates](results_2026-11-28/estimates_2026-11-28.csv), [570 fold scores and counts](results_2026-11-28/fold_scores_2026-11-28.csv). See also the [run protocol and package versions](results_2026-11-28/protocol_2026-11-28.json) and [verification record](results_2026-11-28/verification_2026-11-28.json).

## Interpretation and the 0.01 question

Ten-fold CV has the smallest empirical SD and range for this fixed recipe, dataset and set of seeds. It is a reasonable choice when summarizing split sensitivity in this experiment, at a cost of ten model fits per seed. It also trains each model on more rows than the other schemes, so differences in means or spread cannot be attributed solely to averaging more folds. Hold-out and 5-fold use the same training size per fit, but the latter averages five validation AUCs.

The other three ranges exceed 0.01; this illustrates that changing splits can move an estimate by more than that amount in the observed runs. The 10-fold range of 0.00645 is **not** an error bound for future datasets, a confidence interval, a public-leaderboard interval, or a significance test of a 0.01 model difference. Thirty repetitions do not establish a maximum possible deviation.

To compare two models, evaluate both on the same outer splits and examine the paired differences; any tuning or supervised selection must remain inside training data. This extension evaluates one fixed recipe, so it does not contain that two-model comparison. Stratification also constrains class proportions and may reduce observed spread. The model recipe was selected in earlier work on these same rows, so this study does not provide an untouched estimate of generalization.

The [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) explains training-only preprocessing, averaging fold scores, and the limitations of stratification when interpreting variability.

## Run the experiment

Obtain the original competition ZIP through your own [Kaggle access](https://www.kaggle.com/competitions/small-sample-classification-challenge/data). Data and individual predictions are not included in this repository.

From the repository root, with Python 3.12:

```bash
python -m venv .venv_extension1
source .venv_extension1/bin/activate
python -m pip install -r extension1_2026-11-28/requirements_2026-11-28.txt
python extension1_2026-11-28/validation_protocol_2026-11-28.py \
  --zip /path/to/small-sample-classification-challenge.zip \
  --output extension1_local_results_2026-11-28
python extension1_2026-11-28/verify_results_2026-11-28.py \
  --results extension1_local_results_2026-11-28
```

On Windows activate with `.venv_extension1\Scripts\activate`. The output directory must not already exist. Change its date/version for another run. Output filenames use the actual run date in Asia/Hong_Kong. The default is 30 repeats, beginning at seed 0; `--repeats` and `--first-seed` are available for an explicitly different experiment. No command submits anything to Kaggle.

The aggregate-only checks can also run on the committed results without the competition ZIP:

```bash
python extension1_2026-11-28/verify_results_2026-11-28.py \
  --results extension1_2026-11-28/results_2026-11-28
```

To compare all estimates against the original implementation, add the following options to that command after installing the dependencies:

```bash
  --legacy-script validation_protocol_2026-11-25.py \
  --zip /path/to/small-sample-classification-challenge.zip
```

## What was improved

The [original script](../validation_protocol_2026-11-25.py) is preserved as a historical implementation; it imports the shared Pearson helper. The [new entry point](validation_protocol_2026-11-28.py) is self-contained and adds explicit input/output arguments, protection against overwriting results, actual-date filenames, dataset hashing, package versions, captured warnings, and checks of split isolation, validation coverage and scaler training counts.

The old implementation recorded the last fold’s row counts for a CV run. The new implementation records every fold and the min/max counts per run, correctly showing the uneven 3-fold sizes. The model, seeds, splitters and AUC aggregation are unchanged. Re-running the original functions gave identical AUCs for all 120 estimates, with maximum absolute difference **0.0**; see the [parity record](results_2026-11-28/legacy_parity_2026-11-28.json).

## Contribution and scope

Extension 1 is assigned to Ron in the group presentation. The original code was found in the shared project workspace; individual authorship of that historical file is not independently established here. On 2026-11-28, Codex assisted with packaging, code improvements, execution, verification and documentation at Cao Wei’s request. This is not a claim of unaided authorship or Ron’s separate approval. The existing presentation was not edited in this update; its stronger wording about trusting a 0.01 gap should be read with the limits above.
