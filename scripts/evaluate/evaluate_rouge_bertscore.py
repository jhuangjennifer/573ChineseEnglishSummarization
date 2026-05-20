# Install dependencies
import argparse
import nltk
nltk.download('stopwords')
nltk.download('punkt')      # often needed by the same code path
nltk.download('punkt_tab')  # NLTK 3.8.2+ split punkt into punkt_tab

import os
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from pathlib import Path

pd.set_option('display.max_colwidth', 120)

# Config
DATASET_NAME = "XSAMSum"
# Per-language config: (reference field, prediction field, predictions file, BERTScore model, BERTScore num_layers,
#                       report_rescaled, rouge_lang)
# - ClidSum uses chinese-bert-wwm-ext for Chinese; num_layers=8 matches bert-base-chinese optimal layer per BERTScore paper.
# - BERTScore paper recommends roberta-large for English; num_layers=None lets bert_score pick the recommended layer.
# - report_rescaled=True adds a rescaled-with-baseline F1 column. bert_score ships baselines for roberta-large
#   but not for chinese-bert-wwm-ext, so this is only enabled for English.
# - rouge_lang is the language NAME expected by the multilingual ROUGE toolkit (full name, not ISO code).
#   BERTScore uses the ISO code (dict key) while ROUGE needs the full name.
LANG_CONFIG = {
    "zh": ("hfl/chinese-bert-wwm-ext", 8, False, "zh"),
}

# ClidSum paper uses R-1 / R-2 / R-L
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
TOP_N_WORST = 10

import pandas as pd

# Helper functions
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str)
    parser.add_argument("--reference_col", type=str)
    parser.add_argument("--prediction_col", type=str)

    return parser.parse_args()

def load_summaries(col, path):
    """Load summaries from a csv file, one summary per line."""
    df = pd.read_csv(path)
    return df[col].to_list()


def compute_rouge(predictions, references, rouge_types, language):
    """Compute per-pair and corpus-level ROUGE F1 scores.

    `language` is the full language name expected by the multilingual ROUGE toolkit,
    e.g. "chinese" or "english" — not an ISO code.
    """
    scorer = rouge_scorer.RougeScorer(
        rouge_types=rouge_types,
        lang=language,
    )

    pair_scores = []
    aggregated = {rt: 0.0 for rt in rouge_types}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # (reference, hypothesis)
        pair = {rt: round(scores[rt].fmeasure * 100, 2) for rt in rouge_types}
        pair_scores.append(pair)
        for rt in rouge_types:
            aggregated[rt] += scores[rt].fmeasure

    n = len(predictions)
    corpus_scores = {rt: round(aggregated[rt] / n * 100, 2) for rt in rouge_types}
    return corpus_scores, pair_scores


def compute_bertscore(
    predictions, references, model_type, lang,
    num_layers=None, rescale_with_baseline=False,
    batch_size=32, verbose=True,
):
    """Compute per-pair and corpus-level BERTScore F1.

    `lang` here is the ISO code ("zh", "en") that the bert_score library expects.
    """
    predictions = list(predictions)
    references = list(references)
    scorer_kwargs = dict(
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
    )
    if num_layers is not None:
        scorer_kwargs["num_layers"] = num_layers
    scorer = BERTScorer(**scorer_kwargs)

    # Fix OverflowError on long inputs
    scorer._tokenizer.model_max_length = 512

    P, R, F1 = scorer.score(
        predictions, references,
        verbose=verbose, batch_size=batch_size,
    )

    pair_scores = [round(F1[i].item() * 100, 2) for i in range(len(predictions))]
    corpus_scores = {"f1": round(F1.mean().item() * 100, 2)}
    return corpus_scores, pair_scores

"""## 2. Chinese evaluation

### 2a. Load Chinese references & predictions
"""
args = parse_args()
ref_col_zh, pred_col_zh, results_path_zh = args.reference_col, args.prediction_col, args.results_path
bertscore_model_zh, num_layers_zh, report_rescaled_zh, rouge_lang_zh = LANG_CONFIG["zh"]

references_zh = load_summaries(ref_col_zh, results_path_zh)
predictions_zh = load_summaries(pred_col_zh, results_path_zh)
assert len(predictions_zh) == len(references_zh), (
    f"length mismatch: {len(predictions_zh)} preds vs {len(references_zh)} refs"
)
print(f"Loaded {len(predictions_zh)} Chinese prediction/reference pairs")

"""### 2b. Chinese ROUGE

Multilingual ROUGE toolkit with `lang="chinese"` — this enables Chinese word segmentation (jieba) before n-gram counting. Passing an unrecognized language (e.g. `"zh"`) silently falls back to whitespace tokenization, which for Chinese means character-level matching and substantially inflated scores.
"""

rouge_corpus_zh, rouge_pairs_zh = compute_rouge(
    predictions_zh, references_zh,
    rouge_types=ROUGE_TYPES, language=rouge_lang_zh,
    #use_stemmer=True
)

print("Chinese ROUGE:")
print(f"  ROUGE-1 (R1) : {rouge_corpus_zh['rouge1']:.2f}")
print(f"  ROUGE-2 (R2) : {rouge_corpus_zh['rouge2']:.2f}")
print(f"  ROUGE-L (R-L): {rouge_corpus_zh['rougeL']:.2f}")

"""### 2c. Chinese BERTScore (raw)

Uses `hfl/chinese-bert-wwm-ext` with `num_layers=8`, matching the ClidSum evaluation protocol. Raw F1 only — the `bert_score` library doesn't ship a precomputed baseline for this model, so rescaling isn't available.
"""

bs_raw_corpus_zh, bs_raw_pairs_zh = compute_bertscore(
    predictions_zh, references_zh,
    model_type=bertscore_model_zh,
    lang="zh",
    num_layers=num_layers_zh,
    rescale_with_baseline=False,
    batch_size=32,
)
print(f"Chinese BERTScore F1 (raw): {bs_raw_corpus_zh['f1']}")

"""### 2d. Combine Chinese results, inspect, and save"""

# Corpus-level scores
corpus_scores_zh = rouge_corpus_zh | {"bs_f1_raw": bs_raw_corpus_zh["f1"]}
corpus_df_zh = pd.DataFrame(corpus_scores_zh, index=[0])
print(f"Corpus Eval Scores (Chinese) of {DATASET_NAME}")
print(corpus_df_zh)

# Pair-level scores
pair_cols_zh = (
    {"reference": references_zh, "prediction": predictions_zh}
    | pd.DataFrame(rouge_pairs_zh).to_dict("list")
    | {"bs_f1_raw": bs_raw_pairs_zh}
)
pair_results_df_zh = pd.DataFrame(pair_cols_zh)

print("── BERTScore F1 distribution (Chinese) ──")
print(pair_results_df_zh[["bs_f1_raw"]].describe().round(2))

print("\n── rougeL distribution (Chinese) ──")
print(pair_results_df_zh["rougeL"].describe().round(2))

worst_zh = pair_results_df_zh.nsmallest(TOP_N_WORST, "rougeL")
print(f"── Top {TOP_N_WORST} worst Chinese examples by ROUGE-L of {DATASET_NAME} ──")
worst_zh

OUT_DIR = Path("./rouge_bertscore_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

corpus_df_zh.to_csv(OUT_DIR / f"rouge_bertscore_corpus_scores_zh_{DATASET_NAME}.csv", index=False, encoding="utf-8-sig")
pair_results_df_zh.to_csv(OUT_DIR / f"rouge_bertscore_corpus_pair_scores_zh_{DATASET_NAME}.csv", index=True, encoding="utf-8-sig")
print("Saved Chinese scores to CSV.")
