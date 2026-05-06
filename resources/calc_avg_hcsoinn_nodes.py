import argparse
import ast
import re
from pathlib import Path


def extract_prototypes_dict(log_text: str) -> dict:
    """
    Extract the last 'HC-SOINN prototypes per class: {...}' dict from log text.
    """
    pattern = re.compile(r"HC-SOINN prototypes per class:\s*(\{.*\})")
    matches = pattern.findall(log_text)
    if not matches:
        raise ValueError("No 'HC-SOINN prototypes per class' entry found in log.")

    # Use the last occurrence in case the log contains multiple tasks.
    raw_dict = matches[-1]
    parsed = ast.literal_eval(raw_dict)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed prototypes entry is not a dict.")
    return parsed


def extract_refined_by_class(log_text: str) -> dict:
    """
    Extract latest soinn_refined count per class from lines like:
    [HC-SOINN] class 199: hierarchical_clusters=60 -> soinn_refined=60 (reduction: 0)
    """
    pattern = re.compile(
        r"\[HC-SOINN\]\s+class\s+(\d+):\s+hierarchical_clusters=\d+\s+->\s+soinn_refined=(\d+)"
    )
    refined = {}
    for cls_str, val_str in pattern.findall(log_text):
        refined[int(cls_str)] = int(val_str)
    return refined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate average HC-SOINN prototypes per class from a log file."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to log file containing 'HC-SOINN prototypes per class: {...}'.",
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=None,
        help="Optional expected class count (e.g., 200) for sanity check.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="refined",
        choices=["refined", "dict"],
        help="Data source: 'refined' uses [HC-SOINN] class ... soinn_refined=..., "
        "'dict' uses 'HC-SOINN prototypes per class: {...}'. Default: refined.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    if args.source == "dict":
        proto_dict = extract_prototypes_dict(log_text)
    else:
        proto_dict = extract_refined_by_class(log_text)
        if not proto_dict:
            raise ValueError(
                "No '[HC-SOINN] class ... soinn_refined=...' entries found in log."
            )

    class_ids = sorted(int(k) for k in proto_dict.keys())
    values = [int(proto_dict[c]) for c in class_ids]

    class_count = len(values)
    avg_nodes = sum(values) / float(class_count) if class_count > 0 else 0.0
    min_nodes = min(values) if values else 0
    max_nodes = max(values) if values else 0

    print(f"log: {log_path}")
    print(f"source: {args.source}")
    print(f"class_count: {class_count}")
    print(f"avg_nodes_per_class: {avg_nodes:.6f}")
    print(f"min_nodes: {min_nodes}")
    print(f"max_nodes: {max_nodes}")
    if args.expected_classes is not None:
        if class_count == args.expected_classes:
            print(f"expected_classes_check: PASS ({class_count})")
        else:
            print(
                f"expected_classes_check: FAIL (expected={args.expected_classes}, actual={class_count})"
            )


if __name__ == "__main__":
    main()
