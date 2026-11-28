"""Check saved aggregates; optionally compare every estimate with the old code."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--legacy-script", type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    protocols = list(args.results.glob("protocol_*.json"))
    assert len(protocols) == 1
    protocol = json.loads(protocols[0].read_text())
    date = protocol["run_date"]
    seeds = protocol["seeds"]
    estimates = read_csv(args.results / f"estimates_{date}.csv")
    folds = read_csv(args.results / f"fold_scores_{date}.csv")
    summary = read_csv(args.results / f"summary_{date}.csv")
    assert len(estimates) == 4 * len(seeds) and len(folds) == 19 * len(seeds)
    expected_folds = {"holdout_20pct": 1, "3fold": 3, "5fold": 5, "10fold": 10}
    assert {(r["scheme"], int(r["seed"])) for r in estimates} == {(s, n) for s in expected_folds for n in seeds}
    assert len({(r["scheme"], r["seed"], r["fold"]) for r in folds}) == len(folds)
    for row in estimates:
        selected = [r for r in folds if r["scheme"] == row["scheme"] and r["seed"] == row["seed"]]
        assert len(selected) == expected_folds[row["scheme"]]
        values = [float(r["fold_auc"]) for r in selected]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        assert math.isclose(statistics.mean(values), float(row["estimate_auc"]), abs_tol=1e-12)
        for sample in selected:
            assert int(sample["n_train"]) + int(sample["n_valid"]) == 250
            for part in ("train", "valid"):
                assert int(sample[f"{part}_class0"]) + int(sample[f"{part}_class1"]) == int(sample[f"n_{part}"])
                assert min(int(sample[f"{part}_class0"]), int(sample[f"{part}_class1"])) > 0
        for part in ("train", "valid"):
            counts = [int(r[f"n_{part}"]) for r in selected]
            assert int(row[f"n_{part}_min"]) == min(counts)
            assert int(row[f"n_{part}_max"]) == max(counts)
    for row in summary:
        values = [float(r["estimate_auc"]) for r in estimates if r["scheme"] == row["scheme"]]
        actual = {"mean_auc": statistics.mean(values), "sd": statistics.stdev(values),
                  "min_auc": min(values), "max_auc": max(values), "range": max(values) - min(values)}
        assert int(row["n_repeats"]) == len(seeds)
        assert all(math.isclose(value, float(row[key]), abs_tol=1e-12) for key, value in actual.items())
    assert {(int(r["n_train"]), int(r["n_valid"])) for r in folds if r["scheme"] == "3fold"} == {(166, 84), (167, 83)}
    warning_rows = json.loads((args.results / f"warnings_{date}.json").read_text())
    result = {"estimates_checked": len(estimates), "fold_scores_checked": len(folds),
              "summary_recomputed": True, "three_fold_sizes_correct": True,
              "warning_count": len(warning_rows)}
    if args.legacy_script:
        if args.zip is None:
            parser.error("--zip is required with --legacy-script")
        from threadpoolctl import threadpool_limits
        revised = load_module("revised_validation", Path(__file__).with_name("validation_protocol_2026-11-28.py"))
        legacy = load_module("legacy_validation", args.legacy_script.resolve())
        X, y = revised.load_train(args.zip)
        model = legacy.make_model(legacy.load_helper().PearsonTopK)
        differences = []
        with threadpool_limits(limits=1):
            for row in estimates:
                if row["scheme"] == "holdout_20pct":
                    value, *_ = legacy.holdout_estimate(model, X, y, int(row["seed"]))
                else:
                    k = expected_folds[row["scheme"]]
                    value, *_ = legacy.kfold_estimate(model, X, y, k, int(row["seed"]))
                differences.append(abs(value - float(row["estimate_auc"])))
        assert max(differences) < 1e-12
        result["legacy_estimates_checked"] = len(differences)
        result["max_absolute_auc_difference"] = max(differences)
    print(json.dumps(result, indent=2))
    print("PASS: repeat means, sample SD, ranges, every fold size and class counts.")


if __name__ == "__main__":
    main()
