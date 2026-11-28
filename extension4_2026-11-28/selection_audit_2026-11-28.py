"""Post-hoc checks of archived Extension 4 selections; no model fitting.

Run with Python 3.12; uses only the standard library. Input files are read-only.
The matched Pearson count comes from the archived bootstrap selector, so this
checks feature identity conditional on count, not a bootstrap-free count rule.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pearson_order(features, targets, indices):
    """Centered Pearson calculation, independent of the original NumPy code."""
    n = len(indices)
    mean_y = sum(targets[i] for i in indices) / n
    centered_y = [targets[i] - mean_y for i in indices]
    norm_y = math.sqrt(sum(value * value for value in centered_y))
    correlations = []
    for feature in range(150):
        mean_x = sum(features[i][feature] for i in indices) / n
        centered_x = [features[i][feature] - mean_x for i in indices]
        denominator = math.sqrt(sum(value * value for value in centered_x)) * norm_y
        value = abs(sum(x * y for x, y in zip(centered_x, centered_y)) / denominator) if denominator else 0.0
        correlations.append(value)
    return sorted(range(150), key=lambda feature: (-correlations[feature], feature))


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=base.parent / "small-sample-classification-challenge.zip")
    parser.add_argument("--results", type=Path, default=base / "results_2026-11-25_v2")
    parser.add_argument("--output", type=Path, default=base / "audit_2026-11-28")
    args = parser.parse_args()
    source_script = base / "feature_stability_2026-11-25_v2.py"
    if not source_script.is_file():
        source_script = base.parent / "feature_stability_2026-11-25_v2.py"
    if not source_script.is_file():
        parser.error("Original feature_stability_2026-11-25_v2.py must be beside this audit or in its parent directory.")
    inputs = sorted(args.results.glob("*")) + [source_script, args.zip]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.is_file()}
    with zipfile.ZipFile(args.zip) as archive:
        data = list(csv.DictReader(io.StringIO(archive.read("train.csv").decode("utf-8"))))
    features = [[float(row[str(j)]) for j in range(150)] for row in data]
    targets = [int(row["target"]) for row in data]
    assert len(data) == 250 and all(math.isfinite(value) for row in features for value in row)
    indices = {row["id"]: i for i, row in enumerate(data)}
    assert len(indices) == 250 and set(targets) == {0, 1}
    canonical = b"".join(struct.pack("<150d", *row) for row in features) + struct.pack("<250q", *targets)
    data_hash = hashlib.sha256(canonical).hexdigest()
    archived_verification = json.loads((args.results / "verification_2026-11-25.json").read_text())
    assert data_hash == archived_verification["data_sha256"]
    records = read_csv(args.results / "selection_audit_private_2026-11-25.csv")
    assert len(records) == 75
    expected = {(repeat, fold, k) for repeat in range(1, 6) for fold in range(1, 6) for k in (10, 30, 50)}
    assert {(int(r["repeat"]), int(r["fold"]), int(r["top_count"])) for r in records} == expected
    cache, sets_by_k, checks = {}, {k: [] for k in (10, 30, 50)}, []
    for record in records:
        train_ids = record["train_ids"].split(";")
        validation_ids = record["validation_ids"].split(";")
        assert len(set(train_ids)) == 200 and len(set(validation_ids)) == 50
        assert not set(train_ids) & set(validation_ids)
        assert set(train_ids) | set(validation_ids) == set(indices)
        key = tuple(train_ids)
        if key not in cache:
            cache[key] = pearson_order(features, targets, [indices[item] for item in train_ids])
        selected = set(map(int, record["features"].split(";")))
        count = int(record["n_selected"])
        assert len(selected) == count
        matched = set(cache[key][:count])
        k = int(record["top_count"])
        sets_by_k[k].append(selected)
        checks.append({
            "repeat": int(record["repeat"]), "fold": int(record["fold"]), "top_count": k,
            "selected_count": count, "stable_features": ";".join(map(str, sorted(selected))),
            "matched_pearson_features": ";".join(map(str, sorted(matched))),
            "same_feature_set": selected == matched,
        })
    summary, inclusion = [], []
    for k, sets in sets_by_k.items():
        counts = {size: sum(len(selected) == size for selected in sets) for size in sorted({len(s) for s in sets})}
        always = sorted(set.intersection(*sets))
        summary.append({"top_count": k, "fold_checks": len(sets),
                        "same_feature_set_count": sum(r["same_feature_set"] for r in checks if r["top_count"] == k),
                        "size_distribution": counts, "always_selected": always,
                        "feature49_selected_count": sum(49 in selected for selected in sets)})
        inclusion.extend({"top_count": k, "feature": feature, "folds_selected": sum(feature in selected for selected in sets),
                          "total_folds": len(sets)} for feature in range(150))
    assert all(row["same_feature_set"] for row in checks)
    assert summary[0]["always_selected"] == [17, 46, 77, 80, 98, 100, 112]
    assert summary[0]["feature49_selected_count"] == 6
    assert summary[0]["size_distribution"] == {7: 19, 8: 5, 9: 1}
    # Verify that every existing score is present and agrees with the archived summary.
    scores = read_csv(args.results / "fold_scores_2026-11-25.csv")
    archived_summary = read_csv(args.results / "summary_2026-11-25.csv")
    assert len(scores) == 225
    for row in archived_summary:
        values = [float(score["auc"]) for score in scores if score["model"] == row["model"]]
        assert len(values) == 25 and all(math.isfinite(value) for value in values)
        assert math.isclose(sum(values) / 25, float(row["mean_auc"]), abs_tol=1e-12)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in before.items())
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "matched_feature_sets_2026-11-28.csv", checks)
    write_csv(args.output / "fold_feature_inclusion_2026-11-28.csv", inclusion)
    result = {"date": "2026-11-28", "analysis_type": "post-hoc identity and inclusion audit",
              "data_sha256": data_hash, "identical_sets": 75, "total_checks": 75,
              "model_fits_performed": 0, "new_auc_estimates": False,
              "matching_count_source": "archived bootstrap selected count in each training fold",
              "interpretation": "All observed sets equal Pearson top-m conditional on m. This does not test a bootstrap-free count-selection rule.",
              "fold_dependence": "Overlapping repeated-CV training folds; inclusion counts are descriptive.",
              "summaries": summary}
    (args.output / "audit_summary_2026-11-28.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "input_hashes_2026-11-28.json").write_text(json.dumps(before, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("PASS: data hash, row isolation, 75 matched sets, seven-feature core, feature 49 count, size distribution, archived AUC means, unchanged inputs.")


if __name__ == "__main__":
    main()
