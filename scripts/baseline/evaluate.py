"""Evaluate Chinese and English summaries with ROUGE and BERTScore."""

import os
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import BERTScorer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_NAME = "XSAMSum"
BASE_DIR = "sample_data"

# Per-language config: (reference field, predictions file, BERTScore model, BERTScore num_layers,
#                       report_rescaled, rouge_lang)
# - ClidSum uses chinese-bert-wwm-ext for Chinese; num_layers=8 matches bert-base-chinese optimal layer per BERTScore paper.
# - BERTScore paper recommends roberta-large for English; num_layers=None lets bert_score pick the recommended layer.
# - report_rescaled=True adds a second rescaled-with-baseline F1 column (only supported for models with a precomputed
#   baseline; bert_score ships baselines for roberta-large but not for chinese-bert-wwm-ext).
# - rouge_lang is the language NAME expected by the multilingual ROUGE toolkit (full name, not ISO code).
#   BERTScore uses the ISO code (dict key) while ROUGE needs the full name — hence the separate field.
LANG_CONFIG = {
    "zh": ("summary_zh", "mbart_predictions.txt",    "hfl/chinese-bert-wwm-ext", 8,    False, "chinese"),
    "en": ("summary",    "mbart_predictions_en.txt", "roberta-large",           None, True,  "english"),
}

# ClidSum paper uses R1/R2/R-L
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
TOP_N_WORST = 10


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_files = {
    "train": os.path.join(BASE_DIR, "train.json"),
    "validation": os.path.join(BASE_DIR, "val.json"),
    "test": os.path.join(BASE_DIR, "test.json"),
}
dataset_dict = load_dataset("json", data_files=data_files)


def load_predictions(path):
    """Load predictions from a text file, one summary per line."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------
def compute_rouge(predictions, references, rouge_types, language="en"):
    scorer = rouge_scorer.RougeScorer(
        rouge_types=rouge_types,
        lang=language,
        use_stemmer=True
    )

    pair_scores = []
    aggregated = {rt: 0.0 for rt in rouge_types}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # (reference, hypothesis)
        pair = {rt: round(scores[rt].fmeasure * 100, 2) for rt in rouge_types}
        pair_scores.append(pair)
        for rt in rouge_types:
            aggregated[rt] += scores[rt].fmeasure  # extract only F1 score

    n = len(predictions)
    corpus_scores = {rt: round(aggregated[rt] / n * 100, 2) for rt in rouge_types}

    return corpus_scores, pair_scores


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------
def compute_bertscore(
    predictions,
    references,
    model_type,
    lang="zh",
    num_layers=None,
    rescale_with_baseline=False,
    batch_size=32,
    verbose=True,
):
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

    # Fix the OverflowError
    scorer._tokenizer.model_max_length = 512

    P, R, F1 = scorer.score(
        predictions,
        references,
        verbose=verbose,
        batch_size=batch_size,
    )

    pair_scores = [round(F1[i].item() * 100, 2) for i in range(len(predictions))]
    corpus_scores = {"f1": round(F1.mean().item() * 100, 2)}

    return corpus_scores, pair_scores


# ---------------------------------------------------------------------------
# Evaluation pipeline (ROUGE + BERTScore + combine/save) for one language
# ---------------------------------------------------------------------------
def evaluate(lang):
    ref_field, preds_path, bertscore_model, num_layers, report_rescaled, rouge_lang = LANG_CONFIG[lang]
    label = {"zh": "Chinese", "en": "English"}[lang]

    # Load aligned references and predictions
    references = dataset_dict["test"][ref_field]
    predictions = load_predictions(preds_path)
    assert len(predictions) == len(references)

    # ROUGE — note: multilingual ROUGE toolkit expects the full language name
    # ("chinese", "english"), not the ISO code ("zh", "en"). An unrecognized language
    # silently falls back to whitespace tokenization, which inflates Chinese scores.
    rouge_corpus, rouge_pairs = compute_rouge(predictions, references, ROUGE_TYPES, language=rouge_lang)
    print(f"{label}:")
    print(f"  ROUGE-1 (R1) : {rouge_corpus['rouge1']:.2f}")
    print(f"  ROUGE-2 (R2) : {rouge_corpus['rouge2']:.2f}")
    print(f"  ROUGE-L (R-L): {rouge_corpus['rougeL']:.2f}")

    # BERTScore (raw)
    bs_raw_corpus, bs_raw_pairs = compute_bertscore(
        predictions, references,
        model_type=bertscore_model, lang=lang, num_layers=num_layers,
        rescale_with_baseline=False, batch_size=32,
    )
    print(f"  F1 (B-S raw)     : {bs_raw_corpus['f1']}")

    # BERTScore (rescaled with baseline) — only where a baseline is available
    bs_rescaled_pairs = None
    if report_rescaled:
        bs_rescaled_corpus, bs_rescaled_pairs = compute_bertscore(
            predictions, references,
            model_type=bertscore_model, lang=lang, num_layers=num_layers,
            rescale_with_baseline=True, batch_size=32,
        )
        print(f"  F1 (B-S rescaled): {bs_rescaled_corpus['f1']}")

    # Combine corpus-level scores
    corpus_scores = rouge_corpus | {"bs_f1_raw": bs_raw_corpus["f1"]}
    if bs_rescaled_pairs is not None:
        corpus_scores["bs_f1_rescaled"] = bs_rescaled_corpus["f1"]
    corpus_df = pd.DataFrame(corpus_scores, index=[0])
    print(f"\nCorpus Eval Scores ({label}) of {DATASET_NAME}")
    print(corpus_df)

    # Combine pair-level scores
    pair_cols = (
        {"reference": references, "prediction": predictions}
        | pd.DataFrame(rouge_pairs).to_dict("list")
        | {"bs_f1_raw": bs_raw_pairs}
    )
    if bs_rescaled_pairs is not None:
        pair_cols["bs_f1_rescaled"] = bs_rescaled_pairs
    pair_results_df = pd.DataFrame(pair_cols)

    # Distribution + worst examples
    print(f"\n── BERTScore F1 distribution ({label}) ──")
    bs_cols = ["bs_f1_raw"] + (["bs_f1_rescaled"] if bs_rescaled_pairs is not None else [])
    print(pair_results_df[bs_cols].describe().round(2))

    print(f"\n── rougeL distribution ({label}) ──")
    print(pair_results_df['rougeL'].describe().round(2))

    worst = pair_results_df.nsmallest(TOP_N_WORST, "rougeL")
    print(f"\n── Top {TOP_N_WORST} worst {label} examples by ROUGE-L of {DATASET_NAME} ──")
    print(worst)

    # Save
    corpus_df.to_csv(f"corpus_scores_{lang}_{DATASET_NAME}.csv", index=False, encoding="utf-8-sig")
    pair_results_df.to_csv(f"pair_scores_{lang}_{DATASET_NAME}.csv", index=True, encoding="utf-8-sig")

    return corpus_df, pair_results_df


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
corpus_df_zh, pair_results_df_zh = evaluate("zh")
corpus_df_en, pair_results_df_en = evaluate("en")