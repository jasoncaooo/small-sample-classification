# Extension 4 — Feature ranking stability

**Current report: 2026-11-28.** Cao Wei's individual extension in the joint DSAI4203 project with Ron.

最新版已补充“同数量特征集合核对”和“七个一致入选特征与边界特征 49”。优先点下面的网页版；PDF 预览失败时可下载，Word 可编辑。原始预测结果仍来自 2026-11-25，本次补充核查没有训练新模型。

## Read and download

| Material | Read online | Download / edit |
|---|---|---|
| Revised two-page report | [Web report](Extension4_Report_2026-11-28.md) | [PDF](Extension4_Report_2026-11-28.pdf?raw=true) · [Word](Extension4_Report_2026-11-28.docx?raw=true) |
| Chinese presentation notes, v3 | [Web notes](Extension4_Speaker_Notes_2026-11-28_v3.md) | [PDF](Extension4_Speaker_Notes_2026-11-28_v3.pdf?raw=true) · [Word](Extension4_Speaker_Notes_2026-11-28_v3.docx?raw=true) |
| Experiment and post-hoc checks | [Original experiment](../feature_stability_2026-11-25_v2.py) · [New audit](selection_audit_2026-11-28.py) | [Complete reproduction bundle](Extension4_Reproducible_2026-11-28.zip?raw=true) |
| Evidence | [Audit summary](audit_summary_2026-11-28.json) · [75 set comparisons](matched_feature_sets_2026-11-28.csv) · [Fold inclusion counts](fold_feature_inclusion_2026-11-28.csv) | [Original AUC results](../extension4_summary_2026-11-25.csv) |

## What changed on 28 November

1. **Same-size comparison:** all 75 archived bootstrap-selected sets equal ordinary training-fold Pearson top-*m* sets, where *m* is the number selected by bootstrap in that fold. Set membership is compared, not column order. This is conditional on the bootstrap-selected count; it does not evaluate an independently chosen Pearson count or establish equal newly fitted AUCs.
2. **Boundary sensitivity:** at *k* = 10, features 17, 46, 77, 80, 98, 100 and 112 appear in all 25 training-fold sets. Feature 49 appears in only 6/25, versus 20/25 at *k* = 30 and 25/25 at *k* = 50. These overlapping-fold counts are descriptive, not independent inclusion probabilities.
3. The report and speaker notes now explain these findings and their limitations. The original experiment and original AUCs are unchanged. The new standard-library audit was rerun successfully for this publication: 75/75 set comparisons, row isolation, data hash, feature counts, and archived AUC means passed; zero model fits were performed.

The full-data curves have an eight-feature largest-drop boundary. This is compatible with a seven-feature core across varying training folds. Fixed LR improves from 0.7556 (all 150 features) to 0.8589 (stable *k* = 10), while ordinary top-30 QDA remains better than all stable-subset QDA means. Stability does not establish predictive optimality or a causal benefit from bootstrapping.

## Run from the repository root

Use Python 3.12. Obtain `small-sample-classification-challenge.zip` through your own Kaggle access; `train.csv` must be at the archive root. Data and row-level audit IDs are not distributed here.

```bash
python -m venv .venv-extension4
source .venv-extension4/bin/activate
python -m pip install -r extension4_2026-11-28/requirements_2026-11-28.txt
python feature_stability_2026-11-25_v2.py --zip /path/to/small-sample-classification-challenge.zip --output extension4_local_results_2026-11-28
python extension4_2026-11-28/selection_audit_2026-11-28.py --zip /path/to/small-sample-classification-challenge.zip --results extension4_local_results_2026-11-28 --output extension4_local_audit_2026-11-28
```

On Windows, activate with `.venv-extension4\Scripts\activate`. Both output directories must be new; choose another dated name if they exist. The experiment creates the private row-ID audit locally, which the new audit needs. The public aggregate CSVs alone cannot replace that input. No command submits to Kaggle. The dated scientific output filenames retain the original protocol date even when regenerated; the output directory identifies the rerun.

The original experiment remains at the repository root as a single source file. The audit accepts that layout and the standalone bundle layout, and records the original script hash in either case. The requirements pin the numerical packages used for the original experiment; the new audit itself needs only Python's standard library. The original protocol and pre-score LR amendment remain at the repository root and in the bundle. A full model rerun was not performed for this publishing-only refresh.

## Build editable documents

The bundle preserves the directory layout required by `build_revision_2026-11-28.py`, including the original figure, archived aggregates and new audit. Follow its `REPRODUCE_2026-11-28.md`. The DOCX builder needs `python-docx==1.2.0`; PDF export additionally requires a Word-compatible renderer and Chinese fonts. Published PDFs have already been checked (report: two pages; notes: three pages). Archive the shipped deliverables before rebuilding: the original document builder writes fixed filenames.

## Version cleanup and contribution boundaries

The superseded 25 November report PDF and reproduction ZIP have been replaced in the current branch. Earlier dated README snapshots are removed from the current file list to avoid conflicting entry points; Git history and local backups retain them. Original code, protocol and numerical results remain available because they support this revision.

This extension is prepared for Cao Wei; the underlying classification project is shared with Ron. OpenAI Codex assisted with design, implementation, execution, analysis and drafting. This is not a claim of unaided authorship. The Canva slides were not changed by this publication; the speaker notes distinguish the new audit from older slide screenshots.
