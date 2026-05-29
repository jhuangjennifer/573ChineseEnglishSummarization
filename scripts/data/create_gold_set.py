import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_data import load_json_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a reproducible gold set from pair-level evaluation scores."
    )
    parser.add_argument("--scores_path", required=True, help="CSV with pair-level scores.")
    parser.add_argument("--test_path", default="data/raw/test.json", help="Original test JSON.")
    parser.add_argument("--output_csv", required=True, help="Path for sampled score rows.")
    parser.add_argument("--output_json", required=True, help="Path for sampled dialogue records.")
    parser.add_argument("--n_gold", type=int, default=50, help="Number of examples to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--index_col", default=0, help="CSV index column, or none.")
    parser.add_argument("--reference_col", default="reference", help="Reference column in scores CSV.")
    parser.add_argument(
        "--test_reference_col",
        default="summary_zh",
        help="Reference field in the original test JSON.",
    )
    parser.add_argument(
        "--drop_fields",
        nargs="*",
        default=["summary_de"],
        help="Fields to remove from output JSON records.",
    )
    parser.add_argument(
        "--skip_alignment_check",
        action="store_true",
        help="Do not verify that score row indices align with test JSON positions.",
    )

    return parser.parse_args()


def read_scores(path: str, index_col: str):
    index = None if index_col.lower() == "none" else int(index_col)
    return pd.read_csv(path, index_col=index)


def validate_alignment(gold_df, test_records, reference_col, test_reference_col):
    missing_cols = [col for col in [reference_col] if col not in gold_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required score columns: {missing_cols}")

    mismatches = []
    for index in gold_df.index:
        test_index = int(index)
        if test_index >= len(test_records):
            mismatches.append(test_index)
            continue
        if test_records[test_index].get(test_reference_col) != gold_df.loc[index, reference_col]:
            mismatches.append(test_index)

    if mismatches:
        sample = mismatches[:5]
        raise ValueError(
            f"Found {len(mismatches)} alignment mismatches. "
            f"First mismatched indices: {sample}"
        )


def build_gold_records(gold_df, test_records, drop_fields):
    gold_records = []
    drop_fields = set(drop_fields)

    for index in gold_df.index:
        test_index = int(index)
        record = {
            key: value
            for key, value in test_records[test_index].items()
            if key not in drop_fields
        }
        gold_records.append({"test_index": test_index, **record})

    return gold_records


def main():
    args = parse_args()

    scores = read_scores(args.scores_path, args.index_col)
    if len(scores) < args.n_gold:
        raise ValueError(f"Cannot sample {args.n_gold} rows from only {len(scores)} rows.")

    gold_scores = scores.sample(n=args.n_gold, random_state=args.seed).sort_index()
    test_records = load_json_file(args.test_path).to_dict(orient="records")

    if not args.skip_alignment_check:
        validate_alignment(
            gold_scores,
            test_records,
            args.reference_col,
            args.test_reference_col,
        )

    gold_records = build_gold_records(gold_scores, test_records, args.drop_fields)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gold_scores.to_csv(output_csv)
    output_json.write_text(
        json.dumps(gold_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Sampled {len(gold_scores)} rows from {len(scores)} with seed={args.seed}.")
    metric_cols = [col for col in ["rougeL", "bs_f1_raw"] if col in gold_scores.columns]
    if metric_cols:
        print("Sample metric summary:")
        print(gold_scores[metric_cols].describe().round(2))
    print(f"Wrote sampled scores to: {output_csv}")
    print(f"Wrote gold records to: {output_json}")


if __name__ == "__main__":
    main()
