import re
from collections import Counter
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

try:
    import jieba

    jieba.setLogLevel(20)
except ImportError:  # pragma: no cover - handled by callers that need zh tokens.
    jieba = None


SPEAKER_RE = re.compile(r"^([^:\n]{1,30}):")


def tokenize_english(text: str) -> list[str]:
    """Tokenize English text with the same whitespace strategy used in the notebooks."""
    if not isinstance(text, str) or not text:
        return []
    return text.split()


def tokenize_chinese(text: str) -> list[str]:
    """Tokenize Chinese text with jieba."""
    if not isinstance(text, str) or not text:
        return []
    if jieba is None:
        raise ImportError("jieba is required for Chinese tokenization")
    return [token for token in jieba.lcut(text) if token.strip()]


def chinese_char_count(text: str) -> int:
    """Count CJK unified ideograph characters in a string."""
    if not isinstance(text, str):
        return 0
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add character and token length features for dialogue, summary, and summary_zh."""
    df = df.copy()
    field_tokenizers: dict[str, Callable[[str], list[str]]] = {
        "dialogue": tokenize_english,
        "summary": tokenize_english,
        "summary_zh": tokenize_chinese,
    }

    for field, tokenizer in field_tokenizers.items():
        if field not in df.columns:
            continue
        df[f"{field}_chars"] = df[field].map(lambda text: len(text) if isinstance(text, str) else 0)
        df[f"{field}_words"] = df[field].map(lambda text: len(tokenizer(text)))

    if "summary_zh" in df.columns:
        df["summary_zh_cjk_chars"] = df["summary_zh"].map(chinese_char_count)

    return df


def add_compression_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add summary/dialogue compression ratios for English and Chinese summaries."""
    df = df.copy()

    if "dialogue_words" not in df.columns:
        df = add_length_features(df)

    dialogue_words = df["dialogue_words"].replace(0, np.nan)

    if "summary_words" in df.columns:
        df["compression_en"] = df["summary_words"] / dialogue_words
    if "summary_zh_words" in df.columns:
        df["compression_zh"] = df["summary_zh_words"] / dialogue_words
    if "summary_zh_cjk_chars" in df.columns and "summary_words" in df.columns:
        summary_words = df["summary_words"].replace(0, np.nan)
        df["zh_chars_per_en_word"] = df["summary_zh_cjk_chars"] / summary_words
        df["zh_words_per_en_word"] = df["summary_zh_words"] / summary_words

    return df


def parse_dialogue_turns(dialogue_text: str) -> list[dict[str, str]]:
    """Parse a dialogue into speaker/utterance turns using `speaker: utterance` lines."""
    if not isinstance(dialogue_text, str):
        return []

    normalized = dialogue_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    turns = []

    for line in lines:
        match = SPEAKER_RE.match(line)
        if match:
            speaker = match.group(1).strip()
            utterance = line[match.end() :].strip()
        else:
            speaker = ""
            utterance = line
        turns.append({"speaker": speaker, "utterance": utterance})

    return turns


def dialogue_turn_features(dialogue_text: str) -> dict[str, float]:
    """Compute speaker-turn features used by the exploration notebook."""
    turns = parse_dialogue_turns(dialogue_text)
    if not turns:
        return {
            "num_turns": 0,
            "num_speakers": 0,
            "avg_turn_char_len": 0.0,
            "avg_turn_word_len": 0.0,
            "max_turn_char_len": 0,
            "max_turn_word_len": 0,
            "dominant_speaker_turn_ratio": 0.0,
            "same_speaker_consecutive_ratio": 0.0,
        }

    utterances = [turn["utterance"] for turn in turns]
    turn_char_lengths = [len(text) for text in utterances]
    turn_word_lengths = [len(tokenize_english(text)) for text in utterances]
    speakers = [turn["speaker"] for turn in turns if turn["speaker"]]
    speaker_counts = Counter(speakers)
    same_speaker_pairs = sum(
        1
        for prev, cur in zip(turns, turns[1:])
        if prev["speaker"] and prev["speaker"] == cur["speaker"]
    )

    return {
        "num_turns": len(turns),
        "num_speakers": len(set(speakers)),
        "avg_turn_char_len": float(np.mean(turn_char_lengths)),
        "avg_turn_word_len": float(np.mean(turn_word_lengths)),
        "max_turn_char_len": max(turn_char_lengths),
        "max_turn_word_len": max(turn_word_lengths),
        "dominant_speaker_turn_ratio": (
            max(speaker_counts.values()) / len(turns) if speaker_counts else 0.0
        ),
        "same_speaker_consecutive_ratio": same_speaker_pairs / max(len(turns) - 1, 1),
    }


def add_turn_features(df: pd.DataFrame, dialogue_col: str = "dialogue") -> pd.DataFrame:
    """Add dialogue turn features to a dataframe."""
    df = df.copy()
    if dialogue_col not in df.columns:
        return df

    features = df[dialogue_col].map(dialogue_turn_features).apply(pd.Series)
    return pd.concat([df, features], axis=1)


def missing_value_summary(
    split_dfs: dict[str, pd.DataFrame],
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Count missing-like values by split and field."""
    fields = fields or ["dialogue", "summary", "summary_zh"]
    rows = []

    for split, df in split_dfs.items():
        for field in fields:
            if field not in df.columns:
                continue
            values = df[field]
            missing_count = values.map(
                lambda value: value is None
                or (isinstance(value, float) and pd.isna(value))
                or (isinstance(value, str) and value.strip() == "")
            ).sum()
            rows.append(
                {
                    "split": split,
                    "field": field,
                    "missing_count": int(missing_count),
                    "n": len(df),
                    "missing_rate": missing_count / len(df) if len(df) else 0.0,
                }
            )

    return pd.DataFrame(rows)


def duplicate_summary(split_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count duplicate dialogues and full samples within each split."""
    rows = []

    for split, df in split_dfs.items():
        dialogue_duplicates = 0
        full_sample_duplicates = 0

        if "dialogue" in df.columns:
            dialogue_duplicates = int(df["dialogue"].duplicated(keep=False).sum())

        full_fields = [field for field in ["dialogue", "summary", "summary_zh"] if field in df.columns]
        if full_fields:
            full_sample_duplicates = int(df[full_fields].duplicated(keep=False).sum())

        rows.append(
            {
                "split": split,
                "dialogue_duplicate_rows": dialogue_duplicates,
                "full_sample_duplicate_rows": full_sample_duplicates,
                "n": len(df),
            }
        )

    return pd.DataFrame(rows)


def split_overlap_summary(
    split_dfs: dict[str, pd.DataFrame],
    field: str = "dialogue",
) -> pd.DataFrame:
    """Count overlaps for a field across split pairs."""
    rows = []

    for split_a, split_b in combinations(split_dfs.keys(), 2):
        if field not in split_dfs[split_a].columns or field not in split_dfs[split_b].columns:
            continue
        overlap = set(split_dfs[split_a][field]) & set(split_dfs[split_b][field])
        rows.append(
            {
                "split_a": split_a,
                "split_b": split_b,
                "field": field,
                "overlap_count": len(overlap),
            }
        )

    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Describe numeric profiling columns for one split."""
    numeric_df = df.select_dtypes(include=[np.number])
    rows = []

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "split": split,
                "feature": column,
                "count": int(series.count()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
            }
        )

    return pd.DataFrame(rows)
