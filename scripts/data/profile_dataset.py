import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_data import load_json_file
from src.data.stats import (
    add_compression_features,
    add_length_features,
    add_turn_features,
    duplicate_summary,
    missing_value_summary,
    numeric_profile,
    split_overlap_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile XSAMSum train/validation/test splits."
    )
    parser.add_argument("--train_path", default="data/raw/train.json")
    parser.add_argument("--val_path", default="data/raw/val.json")
    parser.add_argument("--test_path", default="data/raw/test.json")
    parser.add_argument("--output_dir", default="analysis_results/dataset_profile")

    return parser.parse_args()


def load_splits(args):
    return {
        "train": load_json_file(args.train_path),
        "validation": load_json_file(args.val_path),
        "test": load_json_file(args.test_path),
    }


def profile_split(df: pd.DataFrame) -> pd.DataFrame:
    profiled = add_length_features(df)
    profiled = add_compression_features(profiled)
    profiled = add_turn_features(profiled)
    return profiled


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dfs = load_splits(args)
    profiled_splits = {split: profile_split(df) for split, df in split_dfs.items()}

    profile_rows = []
    for split, df in profiled_splits.items():
        profile_rows.append(numeric_profile(df, split))
    profile_df = pd.concat(profile_rows, ignore_index=True)

    missing_df = missing_value_summary(split_dfs)
    duplicates_df = duplicate_summary(split_dfs)
    overlap_df = split_overlap_summary(split_dfs, field="dialogue")

    profile_df.to_csv(output_dir / "numeric_profile.csv", index=False)
    missing_df.to_csv(output_dir / "missing_values.csv", index=False)
    duplicates_df.to_csv(output_dir / "duplicates.csv", index=False)
    overlap_df.to_csv(output_dir / "split_dialogue_overlaps.csv", index=False)

    print("Split sizes:")
    for split, df in split_dfs.items():
        print(f"  {split}: {len(df)}")
    print(f"Wrote dataset profile tables to: {output_dir}")


if __name__ == "__main__":
    main()
