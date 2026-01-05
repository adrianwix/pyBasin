import json
from pathlib import Path


def compare_with_expected_by_size(
    basin_stability: dict[str, float],
    expected_file_path: Path,
) -> None:
    """Compare basin stability results with expected values by matching cluster sizes.

    Sorts both actual and expected results by size (descending) and compares them
    in order, avoiding issues with arbitrary cluster ID ordering.
    """
    if not expected_file_path.exists():
        print(f"Expected file not found: {expected_file_path}")
        return

    with open(expected_file_path) as f:
        expected_data = json.load(f)

    actual_sorted = sorted(basin_stability.items(), key=lambda x: x[1], reverse=True)
    expected_filtered = [item for item in expected_data if item["label"] != "NaN"]
    expected_sorted = sorted(expected_filtered, key=lambda x: x["basinStability"], reverse=True)

    print("\n" + "=" * 60)
    print("COMPARISON WITH EXPECTED RESULTS (sorted by size)")
    print("=" * 60)

    for _, ((cluster_id, actual), expected_item) in enumerate(
        zip(actual_sorted, expected_sorted, strict=False)
    ):
        expected = expected_item["basinStability"]
        label = expected_item["label"]
        diff = actual - expected
        print(
            f"Cluster {cluster_id} -> {label:15s}: "
            f"Actual={actual:.1%}  Expected={expected:.1%}  Diff={diff:+.1%}"
        )

    if len(actual_sorted) != len(expected_sorted):
        print(f"Warning: {len(actual_sorted)} clusters found, {len(expected_sorted)} expected")

    print("=" * 60 + "\n")


def compare_with_expected(
    basin_stability: dict[str, float],
    label_mapping: dict[str, str],
    expected_file_path: Path,
) -> None:
    mapped_results = {label_mapping.get(k, k): float(v) for k, v in basin_stability.items()}

    if not expected_file_path.exists():
        print(f"Expected file not found: {expected_file_path}")
        return

    with open(expected_file_path) as f:
        expected_data = json.load(f)

    expected_results = {
        item["label"]: item["basinStability"] for item in expected_data if item["label"] != "NaN"
    }

    labels_to_compare = sorted([label for label in set(label_mapping.values()) if label != "NaN"])

    print("\n" + "=" * 60)
    print("COMPARISON WITH EXPECTED RESULTS")
    print("=" * 60)
    for label in labels_to_compare:
        actual = mapped_results.get(label, 0.0)
        expected = expected_results.get(label, 0.0)
        diff = actual - expected
        print(f"{label:15s}: Actual={actual:.1%}  Expected={expected:.1%}  Diff={diff:+.1%}")
    print("=" * 60 + "\n")
