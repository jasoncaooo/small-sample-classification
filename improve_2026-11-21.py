"""Prespecified 34-candidate search with paired, nested cross-validation.

No external training data; no leaderboard-guided tuning; no Kaggle upload.
Run from this project with .venv/bin/python improve_2026-11-21.py --date YYYY-MM-DD.
The original ZIP and baseline remain untouched. Every run gets a new directory.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

INNER_SEED = 20261121
OUTER_SEED = 271828


class PearsonTopK(TransformerMixin, BaseEstimator):
    """Rank by absolute Pearson correlation using ONLY the rows passed to fit."""
    def __init__(self, k=150):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        if not 1 <= self.k <= self.n_features_in_:
            raise ValueError('Invalid feature count.')
        xc, yc = X - X.mean(axis=0), y - y.mean()
        denom = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
        corr = np.divide(xc.T @ yc, denom, out=np.zeros(X.shape[1]), where=denom > 0)
        self.scores_ = np.abs(corr)
        # Deterministic ties prefer the lower original feature index.
        ranked = np.argsort(-self.scores_, kind='stable')[:self.k]
        self.indices_ = np.sort(ranked)
        return self

    def transform(self, X):
        check_is_fitted(self, 'indices_')
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count mismatch.')
        return X[:, self.indices_]


def candidates():
    """Frozen before model scores; exact ties retain this order (smaller C first)."""
    specs = []
    for c in [.001, .01, .1, 1.0]:
        specs.append(dict(name=f'L2_all_C{c:g}', family='L2_all', k=150, C=c))
    for c in [.03, .1, .3, 1.0, 3.0, 10.0]:
        specs.append(dict(name=f'L1_all_C{c:g}', family='L1_all', k=150, C=c))
    for k in [3, 5, 10, 20, 30]:
        for c in [.01, .1, 1.0, 10.0]:
            specs.append(dict(name=f'Pearson{k}_L2_C{c:g}', family='Pearson_L2', k=k, C=c))
    for k in [5, 10, 20, 150]:
        specs.append(dict(name=f'LDA_shrinkage_k{k}', family='LDA_shrinkage', k=k, C=None))
    return specs


def build(spec):
    if spec['family'] == 'LDA_shrinkage':
        estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    else:
        estimator = LogisticRegression(C=spec['C'], l1_ratio=int(spec['family'] == 'L1_all'),
                                       solver='liblinear', random_state=42, max_iter=5000)
    return Pipeline([('select', PearsonTopK(spec['k'])),
                     ('scale', StandardScaler()), ('model', estimator)])


def probability(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def splits(X, y):
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                random_state=INNER_SEED).split(X, y))


def evaluate(specs, X, y, cv, progress=False):
    summary, details = [], []
    for spec in specs:
        aucs, nonzero = [], []
        for i, (train_idx, valid_idx) in enumerate(cv):
            model = build(spec).fit(X[train_idx], y[train_idx])
            score = float(roc_auc_score(y[valid_idx], probability(model, X[valid_idx])))
            nnz = int(np.count_nonzero(np.abs(model.named_steps['model'].coef_) > 1e-12))
            aucs.append(score)
            nonzero.append(nnz)
            details.append(dict(candidate=spec['name'], repeat=i // 5 + 1, fold=i % 5 + 1,
                                auc=score, nonzero_coefficients=nnz))
        summary.append(dict(**spec, mean_auc=float(np.mean(aucs)),
                            descriptive_sd=float(np.std(aucs, ddof=1)),
                            min_auc=min(aucs), max_auc=max(aucs), mean_nonzero=float(np.mean(nonzero))))
        if progress:
            print(f"{spec['name']}: mean AUC {np.mean(aucs):.6f}, SD {np.std(aucs, ddof=1):.6f}", flush=True)
    # np.argmax picks the first candidate on an exact tie, as specified above.
    best_index = int(np.argmax([r['mean_auc'] for r in summary]))
    return specs[best_index], summary, details


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, default=root / 'small-sample-classification-challenge.zip')
    parser.add_argument('--date', default=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat())
    args = parser.parse_args()
    datetime.strptime(args.date, '%Y-%m-%d')
    warnings.simplefilter('error', ConvergenceWarning)
    out = root / f'improved_{args.date}'
    v = 1
    while out.exists():
        v += 1
        out = root / f'improved_{args.date}_v{v}'
    out.mkdir()
    def file(stem, ext):
        return out / f'{stem}_{args.date}.{ext}'
    specs = candidates()
    plan = dict(candidates=specs, tie_rule='first in listed order on exact mean AUC tie',
                search='5 folds x 5 repeats; same splits for all candidates', inner_seed=INNER_SEED,
                nested='outer 5 folds x 2 repeats; inner exact same 5x5 selection protocol',
                outer_seed=OUTER_SEED, metric='mean validation-fold ROC AUC of probabilities',
                policy='No alteration of candidate pool after viewing this run; no external data; no leaderboard tuning')
    file('prespecified_plan', 'json').write_text(json.dumps(plan, indent=2) + '\n')
    with zipfile.ZipFile(args.zip) as z:
        tables = {n: list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'))))
                  for n in ['train.csv', 'test.csv', 'sample_submission.csv']}
    train, test, sample = [tables[n] for n in ['train.csv', 'test.csv', 'sample_submission.csv']]
    features = [str(i) for i in range(150)]
    if (len(train), len(test), len(sample)) != (250, 19750, 19750):
        raise ValueError('Unexpected data dimensions.')
    for rows, expected in [(train, ['id', 'target'] + features), (test, ['id'] + features),
                           (sample, ['id', 'target'])]:
        if set(rows[0]) != set(expected) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Unexpected columns or duplicate IDs.')
    if set(r['id'] for r in train) & set(r['id'] for r in test):
        raise ValueError('Train/test IDs overlap.')
    if set(r['id'] for r in test) != set(r['id'] for r in sample):
        raise ValueError('Sample and test IDs differ.')
    X = np.array([[float(r[f]) for f in features] for r in train])
    Xt = np.array([[float(r[f]) for f in features] for r in test])
    y = np.array([int(r['target']) for r in train])
    if set(y) != {0, 1} or not np.isfinite(X).all() or not np.isfinite(Xt).all():
        raise ValueError('Invalid data values.')
    print(f'Output: {out}\nFixed candidate count: {len(specs)}', flush=True)
    best, table, detail = evaluate(specs, X, y, splits(X, y), progress=True)
    write_csv(file('candidate_comparison', 'csv'), sorted(table, key=lambda r: -r['mean_auc']))
    write_csv(file('candidate_fold_scores', 'csv'), detail)
    print(f"Full-data search winner: {best['name']}; beginning nested CV", flush=True)

    baseline = next(s for s in specs if s['name'] == 'L2_all_C1')
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=OUTER_SEED)
    outer_scores, outer_predictions, inner_results = [], [], []
    for i, (train_idx, valid_idx) in enumerate(outer.split(X, y)):
        chosen, inner_table, _ = evaluate(specs, X[train_idx], y[train_idx], splits(X[train_idx], y[train_idx]))
        for row in inner_table:
            inner_results.append(dict(outer_repeat=i // 5 + 1, outer_fold=i % 5 + 1, **row))
        tuned = build(chosen).fit(X[train_idx], y[train_idx])
        base = build(baseline).fit(X[train_idx], y[train_idx])
        p, pb = probability(tuned, X[valid_idx]), probability(base, X[valid_idx])
        score, base_score = float(roc_auc_score(y[valid_idx], p)), float(roc_auc_score(y[valid_idx], pb))
        row = dict(repeat=i // 5 + 1, fold=i % 5 + 1, chosen=chosen['name'],
                   selected_auc=score, baseline_auc=base_score, paired_gain=score - base_score)
        outer_scores.append(row)
        outer_predictions.extend(dict(repeat=i // 5 + 1, fold=i % 5 + 1, id=train[j]['id'],
                                      target=int(y[j]), selected_probability=float(a), baseline_probability=float(b))
                                 for j, a, b in zip(valid_idx, p, pb))
        print(f"Nested outer {i + 1}/10: {chosen['name']} AUC {score:.6f}; baseline {base_score:.6f}", flush=True)
        write_csv(file('nested_outer_scores', 'csv'), outer_scores)
    write_csv(file('nested_inner_searches', 'csv'), inner_results)
    write_csv(file('nested_oof_predictions', 'csv'), outer_predictions)
    outer_repeat = [dict(repeat=r, selected_mean_auc=float(np.mean([s['selected_auc'] for s in outer_scores if s['repeat'] == r])),
                         baseline_mean_auc=float(np.mean([s['baseline_auc'] for s in outer_scores if s['repeat'] == r]))) for r in [1, 2]]
    write_csv(file('nested_repeat_scores', 'csv'), outer_repeat)

    # Final choice is from the prespecified full-data search, NOT outer results.
    final = build(best).fit(X, y)
    p = probability(final, Xt)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid predictions.')
    by_id = dict(zip([r['id'] for r in test], p))
    submission = file('submission_improved', 'csv')
    write_csv(submission, [dict(id=r['id'], target=format(float(by_id[r['id']]), '.17g')) for r in sample])
    indices = final.named_steps['select'].indices_
    coeff = final.named_steps['model'].coef_[0]
    feature_rows = [dict(feature=features[j], standardized_coefficient=float(c)) for j, c in zip(indices, coeff)]
    write_csv(file('final_model_coefficients', 'csv'), feature_rows)
    chosen_row = next(r for r in table if r['name'] == best['name'])
    aucs = np.array([s['selected_auc'] for s in outer_scores])
    base_aucs = np.array([s['baseline_auc'] for s in outer_scores])
    pkgs = ['numpy', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'narwhals', 'cloudpickle']
    dependencies = {n: version(n) for n in pkgs}
    metrics = dict(run_date=args.date, python=platform.python_version(), dependencies=dependencies,
                   source_zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
                   source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   best_configuration=best, selection_cv_auc=chosen_row['mean_auc'],
                   selection_cv_descriptive_sd=chosen_row['descriptive_sd'],
                   nested_mean_auc=float(aucs.mean()), nested_descriptive_sd=float(aucs.std(ddof=1)),
                   nested_min_auc=float(aucs.min()), nested_max_auc=float(aucs.max()),
                   paired_baseline_nested_mean_auc=float(base_aucs.mean()),
                   paired_gain_mean=float(np.mean(aucs - base_aucs)),
                   paired_gain_min=float(np.min(aucs - base_aucs)),
                   paired_gain_max=float(np.max(aucs - base_aucs)),
                   outer_winner_counts=dict(Counter(s['chosen'] for s in outer_scores)),
                   final_nonzero_features=[r['feature'] for r in feature_rows if abs(r['standardized_coefficient']) > 1e-12],
                   full_fit_training_auc=float(roc_auc_score(y, probability(final, X))),
                   submission_rows=len(p), submitted=False, public_auc=None, private_auc=None,
                   limitation='Nested CV evaluates the frozen search process on 200-row outer training sets, not a new independent test set; folds overlap. Final fit uses 250 rows.')
    file('metrics', 'json').write_text(json.dumps(metrics, indent=2) + '\n')
    file('requirements', 'txt').write_text(''.join(f'{n}=={v}\n' for n, v in dependencies.items()))
    write_csv(file('submission_log', 'csv'), [dict(experiment=best['name'], file=submission.name,
              selection_cv_auc=chosen_row['mean_auc'], nested_process_auc=float(aucs.mean()),
              status='NOT_SUBMITTED', public_auc='')])
    note = f'''改进实验 {args.date}

本次结果
最终配置：{best}
用于选择模型的重复五折 AUC：{chosen_row['mean_auc']:.6f}（34选1，存在选择乐观性）。
嵌套验证外层平均 AUC：{aucs.mean():.6f}；描述性折间标准差：{aucs.std(ddof=1):.6f}。
相同外层划分的第一版平均 AUC：{base_aucs.mean():.6f}。
配对平均提升：{np.mean(aucs-base_aucs):.6f}。
全量训练 AUC：{metrics['full_fit_training_auc']:.6f}（不是测试成绩）。
最终非零特征数：{len(metrics['final_nonzero_features'])}；见 final_model_coefficients 文件。
已生成 {len(p)} 行概率，保留样例 ID 顺序；未上传 Kaggle，公开和私有成绩未知。

预设实验设计
候选34个：全特征L2四个强度、全特征L1六个强度、Pearson前3/5/10/20/30特征乘四个L2强度、四个收缩LDA方案。
所有候选采用相同的五折、五次重复，种子20261121，与第一版划分一致。
每个Pipeline内部独立筛选特征和标准化，验证标签不参与训练。
精确并列时按预先列出的顺序选第一个；没有用公开榜或训练AUC决定胜者。
嵌套外层五折重复两次，种子271828；每个外层训练集内重新执行完整34候选、5x5搜索。
外层验证部分只评估选择流程；最终全量参数由全250行的预定搜索选择，不按外层结果改选。

解释与限制
1. 候选比较中的最高均分不等于无偏的最终测试成绩；优先用nested结果描述搜索流程的表现。
2. 外层每次训练200行，内层每次训练160行；最终在250行训练。训练量不同。
3. 重复划分共享样本，标准差不是置信区间；这也不是新增独立测试集。
4. 公榜只有约395行，私榜尚未知；不能承诺公榜达到同样分数或领先队伍水平。
5. L1非零系数可以压缩噪声特征，但被保留不等于已证明它是真实因果特征。
6. 本次未运行个人扩展的完整对照实验；核心调优与个人扩展须分别报告。

文件使用
提交：submission_improved 日期文件。建议说明：{best['name']}; fold-local preprocessing; nested CV evaluated。
保留第一版作实验对照，不必为了填日志而上传每个候选。日志只能记实际提交及得分。
复现：项目 .venv/bin/python improve_2026-11-21.py --date YYYY-MM-DD。
脚本默认读取同目录原始ZIP；新环境先 pip install -r requirements 日期文件。
再次运行自动新建v2、v3目录，保留旧结果。所有结果可以由脚本重建。

参考与AI披露
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
https://scikit-learn.org/stable/common_pitfalls.html
本次使用OpenAI Codex辅助设计候选实验、编写与检查脚本、运行分析；最终报告据实际用途披露。
'''
    file('experiment_note', 'txt').write_text(note, encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)
    print(f'COMPLETE: {submission}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
