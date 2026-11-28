"""Validation protocol study for the fixed final model.

Compare one hold-out split with 3-fold, 5-fold, and 10-fold cross-validation.
Each scheme is repeated with a new split seed. The model is deterministic, so
the spread of AUC comes from the split alone.

One estimate means:
- hold-out: ROC AUC on the single validation block
- k-fold: the average of the k fold AUCs
"""
import csv
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
N_REPEATS = 30
HOLDOUT_FRACTION = 0.2
K_VALUES = (3, 5, 10)
SEEDS = list(range(N_REPEATS))


def load_helper():
    spec = importlib.util.spec_from_file_location("helper", ROOT / "improve_2026-11-21.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def load_train():
    with zipfile.ZipFile(ROOT / "small-sample-classification-challenge.zip") as z:
        rows = list(csv.DictReader(io.StringIO(z.read("train.csv").decode("utf-8-sig"))))
    features = [str(j) for j in range(150)]
    X = np.array([[float(r[j]) for j in features] for r in rows], dtype=float)
    y = np.array([int(r["target"]) for r in rows], dtype=int)
    if X.shape != (250, 150) or set(y) != {0, 1}:
        raise ValueError("Unexpected training data.")
    return X, y


def make_model(pearson_top_k):
    """Same pipeline as the submitted model. QDA has no random state."""
    return make_pipeline(
        pearson_top_k(30),
        StandardScaler(),
        QuadraticDiscriminantAnalysis(reg_param=1e-6, tol=1e-10),
    )


def auc_on_split(model, X, y, train_idx, valid_idx):
    fitted = clone(model).fit(X[train_idx], y[train_idx])
    scores = fitted.decision_function(X[valid_idx])
    return float(roc_auc_score(y[valid_idx], scores))


def holdout_estimate(model, X, y, seed):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=HOLDOUT_FRACTION, random_state=seed
    )
    train_idx, valid_idx = next(splitter.split(X, y))
    auc = auc_on_split(model, X, y, train_idx, valid_idx)
    return auc, [auc], len(train_idx), len(valid_idx)


def kfold_estimate(model, X, y, k, seed):
    # shuffle=True is required. With the default shuffle=False, random_state
    # is ignored and every repeat would reuse the same fold assignment.
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    sizes = None
    for train_idx, valid_idx in splitter.split(X, y):
        folds.append(auc_on_split(model, X, y, train_idx, valid_idx))
        sizes = (len(train_idx), len(valid_idx))
    return float(np.mean(folds)), folds, sizes[0], sizes[1]


def summarize(name, estimates):
    values = np.array(estimates, dtype=float)
    return {
        "scheme": name,
        "n_repeats": int(values.size),
        "mean_auc": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min_auc": float(values.min()),
        "max_auc": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    helper = load_helper()
    X, y = load_train()
    model = make_model(helper.PearsonTopK)
    out = ROOT / "validation_protocol_2026-11-25"
    out.mkdir(exist_ok=True)

    class_counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    print(f"rows={len(y)} features={X.shape[1]} class_counts={class_counts}", flush=True)

    estimates, fold_rows = [], []
    schemes = [("holdout_20pct", None)] + [(f"{k}fold", k) for k in K_VALUES]
    for name, k in schemes:
        seed_aucs = []
        train_n = valid_n = None
        for seed in SEEDS:
            if k is None:
                auc, folds, train_n, valid_n = holdout_estimate(model, X, y, seed)
            else:
                auc, folds, train_n, valid_n = kfold_estimate(model, X, y, k, seed)
            seed_aucs.append(auc)
            estimates.append({
                "scheme": name,
                "seed": seed,
                "estimate_auc": auc,
                "n_train": train_n,
                "n_valid": valid_n,
            })
            for fold_id, fold_auc in enumerate(folds, start=1):
                fold_rows.append({
                    "scheme": name,
                    "seed": seed,
                    "fold": fold_id,
                    "fold_auc": fold_auc,
                })
            print(f"{name} seed={seed} estimate={auc:.6f}", flush=True)
        print(summarize(name, seed_aucs), flush=True)

    summary = [summarize(name, [r["estimate_auc"] for r in estimates if r["scheme"] == name])
               for name, _ in schemes]
    write_csv(out / "estimates_2026-11-25.csv", estimates)
    write_csv(out / "fold_scores_2026-11-25.csv", fold_rows)
    write_csv(out / "summary_2026-11-25.csv", summary)
    (out / "summary_2026-11-25.json").write_text(
        json.dumps({
            "model": "PearsonTopK(30) + StandardScaler + QDA(reg_param=1e-6, tol=1e-10)",
            "repeats": N_REPEATS,
            "seeds": SEEDS,
            "holdout_fraction": HOLDOUT_FRACTION,
            "class_counts": class_counts,
            "what_varies": "Only the random seed of the row split. The model has no random seed.",
            "estimate": "Hold-out uses one validation AUC. K-fold uses the mean of its fold AUCs.",
            "summary": summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
