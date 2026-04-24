import re

import pandas as pd


def clean_dialogue_text(text: str) -> str:
    """Clean dialogue text while preserving speaker and turn information."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_summary_text(text: str) -> str:
    """Clean summary text."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_summarization_dataframe(
    df: pd.DataFrame,
    source_column: str = "dialogue",
    target_column: str = "summary",
) -> pd.DataFrame:
    """Prepare dataframe for dialogue summarization."""
    required_columns = [source_column, target_column]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()

    df[source_column] = df[source_column].apply(clean_dialogue_text)
    df[target_column] = df[target_column].apply(clean_summary_text)

    df = df[(df[source_column] != "") & (df[target_column] != "")]
    df = df.reset_index(drop=True)

    return df


def get_token_lengths(texts, tokenizer) -> list[int]:
    """Return token lengths for a list of texts."""
    return [len(tokenizer.encode(text, truncation=False)) for text in texts]