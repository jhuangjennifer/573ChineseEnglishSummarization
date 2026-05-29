# Team XLS

Group 4 team project repository for **LING 573**.

## Project Overview

This project investigates **English-to-Chinese cross-lingual dialogue summarization**.

The goal is to build a system that takes **multi-turn English dialogues** as input and generates **concise Chinese summaries**.

The task requires the system to handle two challenges:

- **Dialogue summarization**: identifying important information across speaker turns and conversational context.
- **Cross-lingual generation**: producing the final summary in Chinese while the source dialogue is in English.

---

## Task

The main task is:

```text
English dialogue → Chinese summary
```

Our summarize-then-translate pipelines perform this task in two stages:

1. **Summarization**
   - English dialogue → English summary

2. **Translation**
   - English summary → Chinese summary

Our translate-then-summarize pipelines perform this task in two stages:

1. **Translation**
   - English dialogue → Chinese dialogue

2. **Summarization**
   - Chinese dialogue → Chinese summary

Our direct pipelines perform this task in one stage:

1. English dialogue → Chinese summary

---

## Dataset

This project uses the **XSAMSum** subset of the **ClidSum** benchmark.

The dataset contains English dialogues, English summaries, and Chinese summaries.

| Field | Description | Usage in This Project |
|---|---|---|
| `dialogue` | English multi-turn dialogue | Source input |
| `summary` | English reference summary | Target for intermediate English summarization |
| `summary_zh` | Chinese reference summary | Reference for final Chinese summary |

Raw dataset files are **not included in this repository** because of licensing and access restrictions.

---

## Reproducibility Instructions

This section explains how to reproduce the full project pipeline, including data preparation, model loading or training, inference, and evaluation.

---

## 1. Clone the Repository

```bash
git clone https://github.com/jhuangjennifer/573ChineseEnglishSummarization.git
cd 573ChineseEnglishSummarization
```

---

## 2. Set Up the Environment

We recommend using Python 3.10 or later. The commands below assume that they
are run from the repository root.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Source and Prepare the Data

This project uses **XSAMSum only** from the ClidSum benchmark.

ClidSum repository:

```text
https://github.com/krystalan/ClidSum
```

The `train.json`, `val.json`, and `test.json` files are **not directly included** in this repository.

To obtain the data:

1. Go to the ClidSum repository.
2. Follow the ClidSum README instructions for obtaining **XSAMSum**.
3. Request access from the dataset authors if required.
4. After receiving the XSAMSum files, place the train, validation, and test split files under `data/raw/`.
5. Rename the files if necessary so that the final file structure is:

```text
data/raw/train.json
data/raw/val.json
data/raw/test.json
```

Each JSON file should contain the following fields:

```text
dialogue
summary
summary_zh
```

The raw `.json` files should not be committed to GitHub. They are ignored by Git, while `data/raw/.gitkeep` preserves the folder structure.

For the pipeline, the system uses:

```text
Input:           dialogue
English target:  summary
Chinese target:  summary_zh
```

No heavy preprocessing is required for the current pipeline. We preserve the dialogue format as much as possible because speaker names, turn boundaries, informal language, and emojis may contain useful dialogue information.

Verify that the files are readable and inspect basic dataset statistics:

```bash
python scripts/data/profile_dataset.py \
  --train_path data/raw/train.json \
  --val_path data/raw/val.json \
  --test_path data/raw/test.json \
  --output_dir analysis_results/dataset_profile
```

Expected outputs:

```text
analysis_results/dataset_profile/numeric_profile.csv
analysis_results/dataset_profile/missing_values.csv
analysis_results/dataset_profile/duplicates.csv
analysis_results/dataset_profile/split_dialogue_overlaps.csv
```

---

## 4. Dataset Analysis and Gold Set Utilities

The exploratory notebooks under `notebooks/` have script equivalents for reproducible data checks and gold-set creation.

If you want to evaluate on the full test split, you can skip gold-set sampling
and run evaluation directly on full-test result files. If you want the same
50-example gold-set setup used in this repository, first create or obtain a
pair-level score CSV. That CSV should contain one row per test example, use the
original test-set index as the CSV index, and include a `reference` column whose
values match `summary_zh` in `data/raw/test.json`.

One way to create this score CSV is to run baseline inference on the full test
set, build an evaluation CSV from the predictions, and run ROUGE/BERTScore as
shown in Sections 7 and 10. The resulting pair-score file can then be passed to
`--scores_path`.

Create a reproducible 50-example gold set from pair-level evaluation scores:

```bash
python scripts/data/create_gold_set.py \
  --scores_path <PAIR_LEVEL_SCORE_CSV> \
  --test_path data/raw/test.json \
  --output_csv data/gold_results/gold_set_50_zh_XSAMSum_bart.csv \
  --output_json data/gold_results/gold_set_50_zh_XSAMSum_bart.json \
  --n_gold 50 \
  --seed 42
```

Compare the sampled gold set against the full test split:

```bash
python scripts/data/analyze_gold_set.py \
  --gold_path data/gold_results/gold_set_50_zh_XSAMSum_bart.json \
  --test_path data/raw/test.json \
  --output_dir analysis_results/gold_set_analysis \
  --ks_test
```

Expected output:

```text
analysis_results/gold_set_analysis/gold_vs_test_feature_comparison.csv
```

These scripts write derived analysis tables under `analysis_results/`, which is ignored by Git.

---

## 5. Baseline Models

The project uses two fine-tuned baseline summarization models, one direct
English-to-Chinese summarization model, and one pretrained translation model.
The Hugging Face models are downloaded automatically by `transformers` the first
time the relevant training, inference, or evaluation command is run.

| Component | Model |
|---|---|
| BART summarizer | `yunu919/bart-large-dialogue-summarization` |
| mBART English summarizer | `yunu919/mbart-large-dialogue-summarization` |
| mBART English-to-Chinese summarizer | `jjnhuang/mbart-large-50-en-dialogue-to-zh-summary` |
| English-to-Chinese translator | `Helsinki-NLP/opus-mt-en-zh` |

The summarization models can either be loaded directly from Hugging Face or trained locally using the scripts in this repository.

Hugging Face model links:

```text
https://huggingface.co/yunu919/bart-large-dialogue-summarization
https://huggingface.co/yunu919/mbart-large-dialogue-summarization
https://huggingface.co/jjnhuang/mbart-large-50-en-dialogue-to-zh-summary
https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
```

---

## 6. Train the Baseline Summarization Models

The summarization models are trained for:

```text
English dialogue → English summary
```

### 6.1 Train BART

```bash
python scripts/baseline/train_bart.py \
  --train_path data/raw/train.json \
  --val_path data/raw/val.json \
  --test_path data/raw/test.json \
  --output_dir outputs/bart_model
```

The trained BART model will be saved to:

```text
outputs/bart_model/
```

### 6.2 Train mBART

```bash
python scripts/baseline/train_mbart.py \
  --train_path data/raw/train.json \
  --val_path data/raw/val.json \
  --test_path data/raw/test.json \
  --output_dir outputs/mbart_model
```

The trained mBART model will be saved to:

```text
outputs/mbart_model/
```

For mBART, both the source and target language should be set to English:

```text
source language = en_XX
target language = en_XX
```

This ensures that mBART generates intermediate English summaries instead of drifting into another language.

---

## 7. Run the Baseline Inference Pipelines

The full baseline inference pipeline performs:

```text
English dialogue → English summary → Chinese summary
```

It first generates intermediate English summaries using a fine-tuned summarization model. Then, it translates those English summaries into Chinese using `Helsinki-NLP/opus-mt-en-zh`.

---

### 7.1 Run Pipeline with BART

Using the Hugging Face BART checkpoint:

```bash
python scripts/baseline/run_inference_pipeline.py \
  --summary_model yunu919/bart-large-dialogue-summarization \
  --model_tag bart \
  --input_path data/raw/test.json \
  --output_dir outputs
```

Expected outputs:

```text
outputs/bart_predictions_en.txt
outputs/bart_predictions_zh.txt
```

To run the same pipeline only on the 50-example gold set, pass the gold-set JSON:

```bash
python scripts/baseline/run_inference_pipeline.py \
  --summary_model yunu919/bart-large-dialogue-summarization \
  --model_tag bart_gold_50 \
  --input_path data/gold_results/gold_set_50_zh_XSAMSum_bart.json \
  --output_dir outputs
```

---

### 7.2 Run Pipeline with mBART

Using the Hugging Face mBART checkpoint:

```bash
python scripts/baseline/run_inference_pipeline.py \
  --summary_model yunu919/mbart-large-dialogue-summarization \
  --model_tag mbart \
  --input_path data/raw/test.json \
  --output_dir outputs
```

Expected outputs:

```text
outputs/mbart_predictions_en.txt
outputs/mbart_predictions_zh.txt
```

To run the same pipeline only on the 50-example gold set:

```bash
python scripts/baseline/run_inference_pipeline.py \
  --summary_model yunu919/mbart-large-dialogue-summarization \
  --model_tag mbart_gold_50 \
  --input_path data/gold_results/gold_set_50_zh_XSAMSum_bart.json \
  --output_dir outputs
```

---

### 7.3 Run Pipeline with a Locally Trained Model

If the model was trained locally, use the local model directory instead of the Hugging Face model ID.

Example:

```bash
python scripts/baseline/run_inference_pipeline.py \
  --summary_model outputs/bart_model \
  --model_tag bart_local \
  --input_path data/raw/test.json \
  --output_dir outputs
```

Expected outputs:

```text
outputs/bart_local_predictions_en.txt
outputs/bart_local_predictions_zh.txt
```

### 7.4 Build Evaluation Files from Baseline Predictions

The baseline inference script writes one prediction per line. The evaluation
scripts expect a CSV for ROUGE/BERTScore and a JSON file for OmniScore. Build
those files by joining the raw test or gold-set records with the prediction
files:

```bash
python scripts/baseline/build_results_file.py \
  --input_path data/gold_results/gold_set_50_zh_XSAMSum_bart.json \
  --predicted_en_path outputs/bart_gold_50_predictions_en.txt \
  --predicted_zh_path outputs/bart_gold_50_predictions_zh.txt \
  --output_csv data/gold_results/bart_helsinki/bart_helsinki_gold_50_results.csv \
  --output_json data/gold_results/bart_helsinki/bart_helsinki_gold_50_results.json
```

For mBART gold-set predictions, change the input prediction paths and output
directory:

```bash
python scripts/baseline/build_results_file.py \
  --input_path data/gold_results/gold_set_50_zh_XSAMSum_bart.json \
  --predicted_en_path outputs/mbart_gold_50_predictions_en.txt \
  --predicted_zh_path outputs/mbart_gold_50_predictions_zh.txt \
  --output_csv data/gold_results/mbart/mbart_gold_50_results.csv \
  --output_json data/gold_results/mbart/mbart_gold_50_results.json
```

Expected outputs:

```text
data/gold_results/bart_helsinki/bart_helsinki_gold_50_results.csv
data/gold_results/bart_helsinki/bart_helsinki_gold_50_results.json
data/gold_results/mbart/mbart_gold_50_results.csv
data/gold_results/mbart/mbart_gold_50_results.json
```

---

## 8. Agentic Models

The project uses three agentic models.

| Model |
|---|
| `aya-expanse:32b` |
| `gemma3:27b` |
| `qwen3.5:27b` |

The agentic models can be downloaded directly from Ollama.

Ollama model links:
```text
https://ollama.com/library/aya-expanse:32b
https://ollama.com/library/gemma3:27b
https://ollama.com/library/qwen3.5:27b 
```

Install Ollama by following the instructions at `https://ollama.com/`, then pull
the local models:

```bash
ollama pull aya-expanse:32b
ollama pull gemma3:27b
ollama pull qwen3.5:27b
```

---

## 9. Run the Agentic Inference Pipelines

To run the agentic inference pipelines, start Ollama locally and execute the
notebook for the model and pipeline you want to reproduce. The notebooks read
the gold-set records and write CSV/JSONL results under
`notebooks/agents/results/full/`.

Direct English dialogue → Chinese summary:

| Model | Notebook | Expected CSV | Expected JSONL |
|---|---|---|---|
| aya-expanse:32b | `notebooks/agents/pipeline/direct/aya_expanse/direct_aya_full.ipynb` | `notebooks/agents/results/full/direct/aya/direct_aya32b_50samples.csv` | `notebooks/agents/results/full/direct/aya/direct_aya32b_50samples.jsonl` |
| gemma3:27b | `notebooks/agents/pipeline/direct/gemma3/direct_gemma_full.ipynb` | `notebooks/agents/results/full/direct/gemma/direct_gemma27b_50samples.csv` | `notebooks/agents/results/full/direct/gemma/direct_gemma27b_50samples.jsonl` |
| qwen3.5:27b | `notebooks/agents/pipeline/direct/qwen3.5/direct_qwen_full.ipynb` | `notebooks/agents/results/full/direct/qwen/direct_qwen_50samples.csv` | `notebooks/agents/results/full/direct/qwen/direct_qwen_50samples.jsonl` |

Summarize-then-translate:

| Model | Notebook | Expected CSV | Expected JSONL |
|---|---|---|---|
| aya-expanse:32b | `notebooks/agents/pipeline/summarize-then-translate/aya_expanse/st_aya_full.ipynb` | `notebooks/agents/results/full/summarize_then_translate/aya/st_aya32b_50samples.csv` | `notebooks/agents/results/full/summarize_then_translate/aya/st_aya32b_50samples.jsonl` |
| gemma3:27b | `notebooks/agents/pipeline/summarize-then-translate/gemma3/ts_gemma_full.ipynb` | `notebooks/agents/results/full/summarize_then_translate/gemma/st_gemma27b_50samples.csv` | `notebooks/agents/results/full/summarize_then_translate/gemma/st_gemma27b_50samples.jsonl` |
| qwen3.5:27b | `notebooks/agents/pipeline/summarize-then-translate/qwen3.5/st_qwen_full.ipynb` | `notebooks/agents/results/full/summarize_then_translate/qwen/st_qwen_50samples.csv` | `notebooks/agents/results/full/summarize_then_translate/qwen/st_qwen_50samples.jsonl` |

Translate-then-summarize:

| Model | Notebook | Expected CSV | Expected JSONL |
|---|---|---|---|
| aya-expanse:32b | `notebooks/agents/pipeline/translate-then-summarize/aya_expanse/ts_aya_full.ipynb` | `notebooks/agents/results/full/translate_then_summarize/aya/ts_aya32b_50samples.csv` | `notebooks/agents/results/full/translate_then_summarize/aya/ts_aya32b_50samples.jsonl` |
| gemma3:27b | `notebooks/agents/pipeline/translate-then-summarize/gemma3/st_gemma_full.ipynb` | `notebooks/agents/results/full/translate_then_summarize/gemma/ts_gemma27b_50samples.csv` | `notebooks/agents/results/full/translate_then_summarize/gemma/ts_gemma27b_50samples.jsonl` |
| qwen3.5:27b | `notebooks/agents/pipeline/translate-then-summarize/qwen3.5/ts_qwen_full.ipynb` | `notebooks/agents/results/full/translate_then_summarize/qwen/ts_qwen_50samples.csv` | `notebooks/agents/results/full/translate_then_summarize/qwen/ts_qwen_50samples.jsonl` |

Semantic multi-agent pipeline:

| Model | Notebook | Expected CSV | Expected JSONL |
|---|---|---|---|
| aya-expanse:32b | `notebooks/agents/pipeline/semantic pipeline/aya_expanse/sem_aya_full.ipynb` | `notebooks/agents/results/full/semantic/aya/semantic_aya32b_50samples.csv` | `notebooks/agents/results/full/semantic/aya/semantic_aya32b_50samples.jsonl` |
| gemma3:27b | `notebooks/agents/pipeline/semantic pipeline/gemma3/sem_gemma_full.ipynb` | `notebooks/agents/results/full/semantic/gemma/semantic_gemma27b_50samples.csv` | `notebooks/agents/results/full/semantic/gemma/semantic_gemma27b_50samples.jsonl` |
| qwen3.5:27b | `notebooks/agents/pipeline/semantic pipeline/qwen3.5/sem_qwen_full.ipynb` | `notebooks/agents/results/full/semantic/qwen/semantic_qwen27b_50samples.csv` | `notebooks/agents/results/full/semantic/qwen/semantic_qwen27b_50samples.jsonl` |


---

## 10. Evaluate the Outputs

The main evaluation metrics are:

| Metric | Purpose |
|---|---|
| ROUGE-1 | Unigram overlap |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence overlap |
| BERTScore F1 | Semantic similarity |
| OmniScore | Informativeness, clarity, plausibility, faithfulness |

For Chinese ROUGE evaluation, Chinese text should be segmented before score calculation. This project uses `jieba` for Chinese segmentation.

---

### 10.1 Evaluate Baseline Chinese Predictions

```bash
python scripts/evaluate/evaluate_rouge_bertscore.py \
  --results_path data/gold_results/mbart/mbart_gold_50_results.csv \
  --reference_col summary_zh \
  --prediction_col predicted_zh
```

```bash
pip uninstall -y torchvision torchaudio # Needed to resolve dependency issues with OmniScore

python scripts/evaluate/evaluate_omniscore.py \
  --results_path data/gold_results/mbart/mbart_gold_50_results.json \
  --results_type baseline
```

To evaluate BART+Helsinki instead, replace the paths with:

```text
data/gold_results/bart_helsinki/bart_helsinki_gold_50_results.csv
data/gold_results/bart_helsinki/bart_helsinki_gold_50_results.json
```

### 10.2 Evaluate Agentic Chinese Predictions

```bash
python scripts/evaluate/evaluate_rouge_bertscore.py \
  --results_path notebooks/agents/results/full/direct/aya/direct_aya32b_50samples.csv \
  --reference_col reference_chinese_summary \
  --prediction_col final_summary
```

```bash
pip uninstall -y torchvision torchaudio # Needed to resolve dependency issues with OmniScore

python scripts/evaluate/evaluate_omniscore.py \
  --results_path notebooks/agents/results/full/direct/aya/direct_aya32b_50samples.jsonl \
  --results_type agentic
```

Expected evaluation outputs:

```text
rouge_bertscore_results/rouge_bertscore_corpus_pair_scores_zh_XSAMSum.csv
rouge_bertscore_results/rouge_bertscore_corpus_scores_zh_XSAMSum.csv
omniscore_results/omniscore_per_example.csv
omniscore_results/omniscore_summary.csv
omniscore_results/omniscore_run_meta.json
```

To keep outputs for multiple systems separate, move or rename the generated
evaluation directory after each run. The tracked examples in this repository use
paths such as:

```text
data/gold_results/mbart/eval/
data/gold_results/bart_helsinki/eval/
```

---

## 11. Full Reproducible Workflow

To reproduce the project from scratch:

```text
1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies.
4. Obtain XSAMSum by following the ClidSum README instructions.
5. Place the XSAMSum train, validation, and test files under data/raw/.
6. Rename the files as train.json, val.json, and test.json if necessary.
7. Run dataset profiling to verify the data files.
8. Train BART and/or mBART, or load the Hugging Face checkpoints directly.
9. Run baseline inference on the test set or sampled gold set.
10. Build baseline evaluation CSV/JSON files from the prediction text files.
11. Install Ollama and pull the agentic models if reproducing agentic systems.
12. Run the relevant agentic notebooks and verify their CSV/JSONL outputs.
13. Evaluate baseline Chinese predictions with ROUGE/BERTScore and OmniScore.
14. Evaluate agentic Chinese predictions with ROUGE/BERTScore and OmniScore.
```
