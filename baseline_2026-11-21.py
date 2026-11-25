"""Fixed first baseline; no tuning and no Kaggle upload.

Read the original ZIP, evaluate a leakage-safe pipeline, then fit all labelled
rows and export probabilities in sample_submission order. Outputs never overwrite
earlier runs. See the generated experiment note for limitations and references.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import warnings
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def save_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_model():
    # Scaling learns ONLY from each fold's training rows. Default penalty is L2.
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver='liblinear', max_iter=5000, random_state=42))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--zip', type=Path, default=root / 'small-sample-classification-challenge.zip')
    parser.add_argument('--date', default=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat())
    args = parser.parse_args()
    datetime.strptime(args.date, '%Y-%m-%d')
    warnings.simplefilter('error', ConvergenceWarning)
    with zipfile.ZipFile(args.zip) as archive:
        tables = {name: list(csv.DictReader(io.StringIO(
            archive.read(name).decode('utf-8-sig')))) for name in
            ('train.csv', 'test.csv', 'sample_submission.csv')}
    train, test, sample = [tables[n] for n in ('train.csv', 'test.csv', 'sample_submission.csv')]
    features = [str(i) for i in range(150)]
    if (len(train), len(test), len(sample)) != (250, 19750, 19750):
        raise ValueError('Unexpected row counts.')
    for rows, expected in [(train, ['id', 'target'] + features),
                           (test, ['id'] + features), (sample, ['id', 'target'])]:
        if set(rows[0]) != set(expected) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Unexpected columns or duplicate IDs.')
    train_ids, test_ids, sample_ids = [[r['id'] for r in rows] for rows in (train, test, sample)]
    if set(train_ids) & set(test_ids) or set(test_ids) != set(sample_ids):
        raise ValueError('Train/test IDs overlap or submission IDs do not match.')
    X = np.array([[float(r[f]) for f in features] for r in train])
    X_test = np.array([[float(r[f]) for f in features] for r in test])
    y = np.array([int(r['target']) for r in train])
    if set(y) != {0, 1} or not np.isfinite(X).all() or not np.isfinite(X_test).all():
        raise ValueError('Invalid targets or non-finite features; handle before fitting.')

    # Parameters and split protocol are fixed before observing these scores.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=20261121)
    fold_rows, prediction_rows = [], []
    oof = np.full((5, len(y)), np.nan)
    for i, (fit_idx, val_idx) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(i, 5)
        model = make_model().fit(X[fit_idx], y[fit_idx])
        pos = list(model.classes_).index(1)
        p = model.predict_proba(X[val_idx])[:, pos]
        oof[repeat, val_idx] = p
        fold_rows.append(dict(repeat=repeat + 1, fold=fold + 1,
                              n_train=len(fit_idx), n_valid=len(val_idx),
                              auc=float(roc_auc_score(y[val_idx], p))))
        prediction_rows.extend(dict(repeat=repeat + 1, fold=fold + 1,
                                    id=train_ids[j], target=int(y[j]), probability=float(prob))
                               for j, prob in zip(val_idx, p))
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete out-of-fold predictions.')
    repeat_rows = [dict(repeat=r + 1,
                        mean_fold_auc=float(np.mean([v['auc'] for v in fold_rows if v['repeat'] == r + 1])),
                        pooled_oof_auc=float(roc_auc_score(y, oof[r]))) for r in range(5)]
    final = make_model().fit(X, y)
    pos = list(final.classes_).index(1)
    predictions = final.predict_proba(X_test)[:, pos]
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError('Invalid submission probabilities.')
    by_id = dict(zip(test_ids, predictions))
    scores = np.array([r['auc'] for r in fold_rows])
    run = root / f'baseline_{args.date}'
    suffix = 1
    while run.exists():
        suffix += 1
        run = root / f'baseline_{args.date}_v{suffix}'
    run.mkdir()
    def output(stem, ext):
        return run / f'{stem}_{args.date}.{ext}'
    submission = output('submission_baseline', 'csv')
    save_csv(submission, ['id', 'target'],
             [dict(id=ident, target=format(float(by_id[ident]), '.17g')) for ident in sample_ids])
    save_csv(output('fold_scores', 'csv'), list(fold_rows[0]), fold_rows)
    save_csv(output('repeat_scores', 'csv'), list(repeat_rows[0]), repeat_rows)
    save_csv(output('oof_predictions', 'csv'), list(prediction_rows[0]), prediction_rows)
    packages = ['numpy', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'narwhals', 'cloudpickle']
    dependencies = {name: version(name) for name in packages}
    metrics = dict(
        run_date=args.date, python=platform.python_version(), dependencies=dependencies,
        source_zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_rows=len(train), test_rows=len(test), n_features=len(features),
        target_counts={str(k): int((y == k).sum()) for k in [0, 1]},
        duplicate_train_feature_rows=int(len(X) - len(np.unique(X, axis=0))),
        constant_train_features=int((np.ptp(X, axis=0) == 0).sum()),
        model='StandardScaler + L2 LogisticRegression, C=1, liblinear, model seed=42',
        protocol='5-fold stratified CV, repeated 5 times, split seed=20261121; no tuning',
        mean_fold_auc=float(scores.mean()), descriptive_fold_auc_sd=float(scores.std(ddof=1)),
        min_fold_auc=float(scores.min()), max_fold_auc=float(scores.max()),
        repeat_scores=repeat_rows,
        full_fit_training_auc=float(roc_auc_score(y, final.predict_proba(X)[:, pos])),
        submission_rows=len(predictions), probability_min=float(predictions.min()),
        probability_max=float(predictions.max()), kaggle_submitted=False,
        kaggle_public_auc=None, kaggle_private_auc=None)
    output('metrics', 'json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n')
    output('requirements', 'txt').write_text(''.join(f'{n}=={v}\n' for n, v in dependencies.items()))
    save_csv(output('submission_log', 'csv'),
             ['experiment', 'file', 'local_mean_fold_auc', 'status', 'public_auc', 'notes'],
             [dict(experiment='fixed_L2_baseline', file=submission.name,
                   local_mean_fold_auc=float(scores.mean()), status='NOT_SUBMITTED',
                   public_auc='', notes='First fixed baseline, not a tuned final model')])
    note = f'''第一份基准实验：{args.date}

目的：跑通训练、验证、预测流程，建立后续实验的对照；不是最终最佳模型。
模型：150 个特征全部使用，排除 id 和 target；每折独立标准化；L2 逻辑回归 C=1。
验证：分层五折，重复五次，共 25 个验证分数；每折约 200 条训练、50 条验证。
平均折 AUC：{scores.mean():.6f}
折 AUC 标准差：{scores.std(ddof=1):.6f}（描述划分波动，不是标准误或置信区间）。
全量拟合后的训练 AUC：{metrics['full_fit_training_auc']:.6f}（不是泛化成绩）。
最终预测：用全部 250 条训练，生成 19,750 个类别 1 概率，按样例 ID 顺序输出。
状态：仅生成本地文件，没有向 Kaggle 提交；公开榜和私有榜分数未知。

如何理解分数：
1. 重复划分使用同一批数据，25 个分数并非独立实验，不能据此声称精确置信区间。
2. repeat_scores 同时记录每轮平均折 AUC 和汇总 OOF AUC；二者计算方式不同。
   汇总 OOF 混合多个模型的概率，仅作补充，不与平均折 AUC 混为一谈。
3. CV 每次训练 200 条，最终模型训练 250 条，因此两者不完全是同一训练条件。
4. 这次没有搜索参数。之后反复调参会使被选中模型的 CV 成绩偏乐观，须如实披露。
5. 公开榜只覆盖约 395 条测试数据，不能代替本地验证或视为最终成绩。

你现在要做：
1. 在 Kaggle 点 Submit Prediction，上传本目录中的 submission_baseline 日期文件。
2. 提交说明填写：Baseline: StandardScaler + L2 LogisticRegression; C=1; 5x5 CV。
3. 保存实际提交时间和公开榜 AUC，补到自己的实验记录；不要虚构成绩或提交经历。
4. 与队友确认各自扩展：建议一人做验证方案设计，一人做信息泄漏；姓名待填写。

后续实验：
核心任务：在同一组划分上对照少量候选模型与正则化设置，记录参数、均值、波动。
扩展 1：比较留出、3/5/10 折，多次改变划分种子，固定模型随机种子。
扩展 3：配对比较全数据相关性筛选与每折内相关性筛选，保持划分和模型相同。
相关性筛选中的保留特征数与交叉验证折数请用不同变量名，避免混淆。
故意泄漏的流程仅用于教学对照，不用于证明最终模型优越。

复现：在项目目录使用 .venv/bin/python baseline_2026-11-21.py --date YYYY-MM-DD。
脚本直接读取原 ZIP，无需手动解压；重跑会生成 v2、v3 目录，保留原结果。
若在其他电脑运行：创建 Python 虚拟环境，pip install -r 本目录的 requirements 日期文件。
代码和原始 ZIP 需要保留在同一父目录；也可用 --zip 指定 ZIP 的路径。

参考资料（报告中应列明实际使用的文献和软件）：
https://scikit-learn.org/stable/common_pitfalls.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

AI 使用记录：本次使用 OpenAI Codex 辅助编写、检查和运行基准脚本、解释验证方案。
最终报告需按实际使用情况披露；两名成员都应理解自己报告和代码中的内容。
'''
    output('experiment_note', 'txt').write_text(note, encoding='utf-8')
    print(json.dumps(dict(output_directory=str(run), mean_fold_auc=metrics['mean_fold_auc'],
                         fold_auc_sd=metrics['descriptive_fold_auc_sd'],
                         training_auc=metrics['full_fit_training_auc'],
                         submission=str(submission), submitted=False), indent=2))


if __name__ == '__main__':
    main()
