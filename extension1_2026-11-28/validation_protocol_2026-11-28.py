"""Extension 1: split sensitivity of a fixed Pearson-top-30 + QDA recipe.

Only train.csv is read. No raw rows, row IDs, predictions or submissions are
written. Results describe repeated splits of one dataset, not new test sets.
"""
import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import warnings
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

SCHEMES = (("holdout_20pct", None), ("3fold", 3), ("5fold", 5), ("10fold", 10))


class PearsonTopK(TransformerMixin, BaseEstimator):
    """Same training-only selector and tie policy as the shared model."""

    def __init__(self, k=30):
        self.k = k

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        if not 1 <= self.k <= self.n_features_in_:
            raise ValueError("Invalid feature count")
        xc, yc = X - X.mean(axis=0), y - y.mean()
        denominator = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
        corr = np.divide(xc.T @ yc, denominator, out=np.zeros(X.shape[1]), where=denominator > 0)
        self.scores_ = np.abs(corr)
        ranked = np.argsort(-self.scores_, kind="stable")[:self.k]
        self.indices_ = np.sort(ranked)
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count mismatch")
        return X[:, self.indices_]


def make_model():
    return make_pipeline(PearsonTopK(30), StandardScaler(),
                         QuadraticDiscriminantAnalysis(reg_param=1e-6, tol=1e-10))


def load_train(path):
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("train.csv").decode("utf-8-sig"))))
    X = np.array([[float(row[str(j)]) for j in range(150)] for row in rows], dtype="<f8")
    y = np.array([int(row["target"]) for row in rows], dtype="<i8")
    if X.shape != (250, 150) or set(y) != {0, 1} or not np.isfinite(X).all():
        raise ValueError("Expected 250 finite training rows, 150 features and binary labels")
    return X, y


def splits_for(X, y, folds, seed):
    if folds is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(X, y))


def summarize(scheme, estimates):
    values = np.array(estimates, dtype=float)
    return {"scheme": scheme, "n_repeats": len(values), "mean_auc": float(values.mean()),
            "sd": float(values.std(ddof=1)), "min_auc": float(values.min()),
            "max_auc": float(values.max()), "range": float(np.ptp(values))}


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(zip_path, output, repeats=30, first_seed=0):
    if repeats < 2 or first_seed < 0:
        raise ValueError("Use at least two repeats and a nonnegative seed")
    X, y = load_train(zip_path)
    output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(first_seed, first_seed + repeats))
    date = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    environment = {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "joblib", "narwhals", "cloudpickle")}
    protocol = {
        "run_date": date, "python": platform.python_version(), "platform": platform.platform(),
        "packages": environment, "data_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "shape": list(X.shape), "class_counts": np.bincount(y).tolist(), "seeds": seeds,
        "model": "PearsonTopK(30) + StandardScaler + QDA(reg_param=1e-6, tol=1e-10)",
        "holdout_fraction": 0.2, "cv_folds": [3, 5, 10], "shuffle": True,
        "statistic": "One estimate per seed: hold-out AUC or arithmetic mean of k fold AUCs.",
        "scope": "Fixed recipe; selection and scaling fitted on each training fold. No tuning or submissions.",
        "limitations": "Dependent reuse of the same data; training sizes differ across schemes; model was selected in earlier work. SD/range are descriptive, not confidence intervals or a test of a 0.01 model difference.",
    }
    (output / f"protocol_{date}.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model = make_model()
    estimates, fold_scores, warning_rows = [], [], []
    with threadpool_limits(limits=1):
        for scheme, k in SCHEMES:
            for seed in seeds:
                split_list = splits_for(X, y, k, seed)
                if k is not None:
                    validation_counts = np.zeros(len(y), dtype=int)
                    for _, va in split_list:
                        validation_counts[va] += 1
                    assert np.all(validation_counts == 1)
                scores, train_sizes, valid_sizes = [], [], []
                for fold, (tr, va) in enumerate(split_list, 1):
                    assert not np.intersect1d(tr, va).size
                    assert len(tr) + len(va) == len(y)
                    assert set(y[tr]) == {0, 1} and set(y[va]) == {0, 1}
                    fitted = clone(model)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fitted.fit(X[tr], y[tr])
                        prediction = fitted.decision_function(X[va])
                    assert np.isfinite(prediction).all()
                    selector = fitted.named_steps["pearsontopk"]
                    scaler = fitted.named_steps["standardscaler"]
                    assert int(scaler.n_samples_seen_) == len(tr)
                    assert np.allclose(scaler.mean_, X[tr][:, selector.indices_].mean(axis=0))
                    auc = float(roc_auc_score(y[va], prediction))
                    scores.append(auc)
                    train_sizes.append(len(tr))
                    valid_sizes.append(len(va))
                    fold_scores.append({"scheme": scheme, "seed": seed, "fold": fold,
                                        "n_train": len(tr), "n_valid": len(va),
                                        "train_class0": int(np.sum(y[tr] == 0)), "train_class1": int(np.sum(y[tr] == 1)),
                                        "valid_class0": int(np.sum(y[va] == 0)), "valid_class1": int(np.sum(y[va] == 1)),
                                        "fold_auc": auc})
                    warning_rows.extend({"scheme": scheme, "seed": seed, "fold": fold,
                                         "category": item.category.__name__, "message": str(item.message)} for item in caught)
                estimates.append({"scheme": scheme, "seed": seed, "estimate_auc": float(np.mean(scores)),
                                  "n_folds": len(scores), "n_train_min": min(train_sizes), "n_train_max": max(train_sizes),
                                  "n_valid_min": min(valid_sizes), "n_valid_max": max(valid_sizes)})
            print(json.dumps(summarize(scheme, [row["estimate_auc"] for row in estimates if row["scheme"] == scheme])), flush=True)
    summary = [summarize(scheme, [row["estimate_auc"] for row in estimates if row["scheme"] == scheme]) for scheme, _ in SCHEMES]
    write_csv(output / f"estimates_{date}.csv", estimates)
    write_csv(output / f"fold_scores_{date}.csv", fold_scores)
    write_csv(output / f"summary_{date}.csv", summary)
    (output / f"warnings_{date}.json").write_text(json.dumps(warning_rows, indent=2) + "\n")
    verification = {"data_sha256": protocol["data_sha256"], "n_estimates": len(estimates),
                    "n_model_fits": len(fold_scores), "expected_model_fits": repeats * 19,
                    "warning_count": len(warning_rows), "all_scores_finite": True,
                    "disjoint_train_validation_checked": True, "cv_validation_coverage_checked": True,
                    "scaler_training_rows_checked": True, "raw_rows_or_predictions_exported": False}
    assert verification["n_estimates"] == repeats * 4 and len(fold_scores) == repeats * 19
    (output / f"verification_{date}.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2), flush=True)
    return estimates, fold_scores, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="Original competition ZIP containing train.csv")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=0)
    args = parser.parse_args()
    run(args.zip, args.output, args.repeats, args.first_seed)


if __name__ == "__main__":
    main()
