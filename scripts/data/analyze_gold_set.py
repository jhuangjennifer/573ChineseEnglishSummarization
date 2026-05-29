import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_data import load_json_file
from src.data.stats import add_compression_features, add_length_features, add_turn_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a sampled gold set against the full test set."
    )
    parser.add_argument("--gold_path", required=True, help="Gold set JSON or JSONL.")
    parser.add_argument("--test_path", default="data/raw/test.json", help="Full test JSON.")
    parser.add_argument("--output_dir", default="analysis_results/gold_set_analysis")
    parser.add_argument(
        "--ks_test",
        action="store_true",
        help="Also run scipy.stats.ks_2samp for each feature.",
    )

    return parser.parse_args()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_length_features(df)
    df = add_compression_features(df)
    df = add_turn_features(df)
    return df


def compare_feature(gold_df, test_df, feature):
    gold = gold_df[feature].dropna()
    test = test_df[feature].dropna()
    return {
        "feature": feature,
        "gold_n": len(gold),
        "test_n": len(test),
        "gold_mean": gold.mean(),
        "test_mean": test.mean(),
        "gold_std": gold.std(),
        "test_std": test.std(),
        "mean_diff": gold.mean() - test.mean(),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_df = load_json_file(args.gold_path)
    test_df = load_json_file(args.test_path)

    gold_features = prepare_features(gold_df)
    test_features = prepare_features(test_df)

    feature_cols = [
        "dialogue_chars",
        "dialogue_words",
        "summary_chars",
        "summary_words",
        "summary_zh_chars",
        "summary_zh_words",
        "summary_zh_cjk_chars",
        "zh_chars_per_en_word",
        "zh_words_per_en_word",
        "num_turns",
        "num_speakers",
        "compression_en",
        "compression_zh",
    ]
    feature_cols = [
        col
        for col in feature_cols
        if col in gold_features.columns and col in test_features.columns
    ]

    rows = [compare_feature(gold_features, test_features, feature) for feature in feature_cols]
    comparison_df = pd.DataFrame(rows)

    if args.ks_test:
        from scipy.stats import ks_2samp

        ks_stats = []
        for feature in feature_cols:
            gold = gold_features[feature].replace([np.inf, -np.inf], np.nan).dropna()
            test = test_features[feature].replace([np.inf, -np.inf], np.nan).dropna()
            stat, p_value = ks_2samp(gold, test)
            ks_stats.append({"feature": feature, "ks_stat": stat, "p_value": p_value})
        comparison_df = comparison_df.merge(pd.DataFrame(ks_stats), on="feature")

    comparison_path = output_dir / "gold_vs_test_feature_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(f"Gold set size: {len(gold_df)}")
    print(f"Test set size: {len(test_df)}")
    print("Feature comparison:")
    print(comparison_df.round(4))
    print(f"Wrote comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()
