import argparse
import json
import time
import torch
import pandas as pd
import random
import numpy as np
from datetime import datetime, timezone
from enum import StrEnum
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Optional

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str)
    parser.add_argument("--results_type", choices=['baseline', 'agentic'])

    return parser.parse_args()

# Reproducibility — OmniScore is deterministic at inference, but we still
# fix seeds so that any sampling we do (e.g. selecting a slice) is stable.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# Load OmniScore
REPO_ID = "QCRI/OmniScore-deberta-v3"
MAX_LEN = 512  # the model's own max sequence length

tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO_ID, trust_remote_code=True).to(DEVICE).eval()

SCORE_NAMES = list(model.config.score_names)
print("Score dimensions:", SCORE_NAMES)
print("Output range: [1, 5] (sigmoid-scaled in the regression head)")

@dataclass
class GeneralResult:
    test_index: str
    dialogue: str
    summary: str
    summary_zh: str
    predicted_en: str
    predicted_zh: str

# The baseline and agentic model results are formatted differently.
# Use BASELINE for baseline model results, and AGENTIC for agentic model results.
class ResultType(StrEnum):
    BASELINE = "baseline"
    AGENTIC = "agentic"

# Read a JSON with format matching the dataset_type into a list of GeneralResult
def load_dataset_slice(dataset_type, path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
      if (dataset_type == ResultType.BASELINE):
        json_items = json.load(f)
        for json_item in json_items:
          if not "predicted_en" in json_item:
            json_item["predicted_en"] = ""
          general_result = GeneralResult(**json_item)
          items.append(general_result)
      else:
        for line in f:
          json_item = json.loads(line)
          #agentic_result = AgenticResult(**json_item)
          general_result = GeneralResult(
              test_index = json_item["test_index"],
              dialogue = json_item["dialogue"],
              summary = json_item["reference_english_summary"],
              summary_zh = json_item["reference_chinese_summary"],
              predicted_en = "", # No predicted English summary for agentic model
              predicted_zh = json_item["final_summary"]
          )
          items.append(general_result)
    if limit is not None:
        items = items[:limit]
    return items

args = parse_args()
EXAMPLES = load_dataset_slice(args.results_type, path=args.results_path, limit=None)
print(f"Loaded {len(EXAMPLES)} examples.")

# Format inputs for OmniScore
TASK_TAG = "summarization"

def format_source_grounded(ex):
    return (
        f"Task: {TASK_TAG}\n"
        f"Source: {ex.dialogue}\n"
        f"Candidate: {ex.predicted_zh}"
    )

def format_reference_based(ex):
    return (
        f"Task: {TASK_TAG}\n"
        f"Reference: {ex.summary_zh}\n"
        f"Candidate: {ex.predicted_zh}"
    )

# Sanity check on the first example
print("=== Source-grounded input ===")
print(format_source_grounded(EXAMPLES[0])[:400], "...")
print()
print("=== Reference-based input ===")
print(format_reference_based(EXAMPLES[0])[:400], "...")

# Run batched scoring
BATCH_SIZE = 8  # safe default for a 0.2B model on a single GPU; tune for your hardware

def _was_truncated(text):
    # Did the tokenizer cut this input at MAX_LEN?
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    return len(ids) > MAX_LEN

def score_batch(texts):
    # Score a list of inputs. Returns (scores_per_input, truncation_flags).
    trunc_flags = [_was_truncated(t) for t in texts]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    preds = out.predictions.detach().cpu()  # shape (B, len(SCORE_NAMES))
    rows = []
    for row in preds:
        rows.append({name: float(row[i]) for i, name in enumerate(SCORE_NAMES)})
    return rows, trunc_flags

def score_all(examples, formatter, mode_name):
    # Run scoring for every example using the given input formatter.
    all_rows = []
    n = len(examples)
    t0 = time.time()
    for start in tqdm(range(0, n, BATCH_SIZE), desc=f"OmniScore [{mode_name}]"):
        chunk = examples[start : start + BATCH_SIZE]
        texts = [formatter(ex) for ex in chunk]
        scores, trunc_flags = score_batch(texts)
        for ex, sc, tr in zip(chunk, scores, trunc_flags):
            row = {"example_id": ex.test_index, "mode": mode_name, "truncated": tr}
            row.update(sc)
            all_rows.append(row)
    elapsed = time.time() - t0
    print(f"[{mode_name}] scored {n} examples in {elapsed:.1f}s "
          f"({n / max(elapsed, 1e-9):.1f} ex/s)")
    return pd.DataFrame(all_rows)

df_source = score_all(EXAMPLES, format_source_grounded, "source_grounded")
df_ref    = score_all(EXAMPLES, format_reference_based, "reference_based")
df_all = pd.concat([df_source, df_ref], ignore_index=True)
df_all

# Aggregate and report
def summarize(df):
    g = df.groupby("mode")
    means = g[SCORE_NAMES].mean().add_suffix("_mean")
    stds  = g[SCORE_NAMES].std().add_suffix("_std")
    trunc = g["truncated"].mean().rename("truncation_rate").to_frame()
    count = g.size().rename("n").to_frame()
    out = pd.concat([count, trunc, means, stds], axis=1)
    # Reorder: count, trunc, then mean/std interleaved per dim
    cols = ["n", "truncation_rate"]
    for s in SCORE_NAMES:
        cols.extend([f"{s}_mean", f"{s}_std"])
    return out[cols]

summary = summarize(df_all)
summary

# Per-example view — easier to inspect outliers (e.g. low faithfulness scores)
view = df_all.pivot_table(
    index="example_id",
    columns="mode",
    values=SCORE_NAMES,
    aggfunc="first",
)
# Flatten the column MultiIndex: (score, mode) -> "score__mode"
view.columns = [f"{s}__{m}" for s, m in view.columns]
view = view.reset_index()
view

# Flag examples worth eyeballing: low faithfulness in source-grounded mode
LOW_FAITH_THRESHOLD = 3.0
flagged = df_source[df_source["faithfulness"] < LOW_FAITH_THRESHOLD].copy()
if len(flagged):
    print(f"{len(flagged)} example(s) below faithfulness {LOW_FAITH_THRESHOLD} "
          f"in source-grounded mode — likely hallucination candidates:")
    print(flagged[["example_id", "faithfulness", "informativeness", "clarity", "plausibility"]])
else:
    print(f"No examples below faithfulness {LOW_FAITH_THRESHOLD} in source-grounded mode.")

# Save results
OUT_DIR = Path("./omniscore_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

per_example_path = OUT_DIR / "omniscore_per_example.csv"
summary_path     = OUT_DIR / "omniscore_summary.csv"
meta_path        = OUT_DIR / "omniscore_run_meta.json"

df_all.to_csv(per_example_path, index=False, encoding="utf-8")
summary.to_csv(summary_path, encoding="utf-8")

meta = {
    "model": REPO_ID,
    "max_seq_len": MAX_LEN,
    "score_names": SCORE_NAMES,
    "score_range": [1.0, 5.0],
    "device": DEVICE,
    "batch_size": BATCH_SIZE,
    "n_examples": len(EXAMPLES),
    "modes": ["source_grounded", "reference_based"],
    "task_tag": TASK_TAG,
    "seed": SEED,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print("Wrote:")
for p in (per_example_path, summary_path, meta_path):
    print(f"  {p}  ({p.stat().st_size} bytes)")
