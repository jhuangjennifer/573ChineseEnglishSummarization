import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


def load_json_file(file_path: str) -> pd.DataFrame:
    """Load a JSON file that may be either a JSON array or JSON Lines."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            return pd.DataFrame([data])

        raise ValueError(f"Unsupported JSON structure in {file_path}")

    except json.JSONDecodeError:
        records = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    records.append(json.loads(line))

        return pd.DataFrame(records)


def load_train_val_test(
    train_path: str,
    val_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test JSON files."""
    train_df = load_json_file(train_path)
    val_df = load_json_file(val_path)
    test_df = load_json_file(test_path)

    return train_df, val_df, test_df


def create_dataset_dict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> DatasetDict:
    """Convert pandas DataFrames into a Hugging Face DatasetDict."""
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )