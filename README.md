# Team XLS

Group 4 team project repository for **LING 573**.

## Project Overview

This project investigates **English-to-Chinese cross-lingual dialogue summarization**.

The goal is to build a system that takes **multi-turn English dialogues** as input and generates **concise Chinese summaries**.

The task requires the system to handle two challenges:

- **Dialogue summarization**: identifying important information across speaker turns and conversational context.
- **Cross-lingual generation**: producing the final summary in Chinese while the source dialogue is in English.

Our current implementation uses a **pipeline approach**:

```text
English dialogue → English summary → Chinese summary
```

This design separates summarization from translation, making the system easier to debug and allowing us to evaluate each component separately.

---

## Task

The main task is:

```text
English dialogue → Chinese summary
```

Our pipeline performs this task in two stages:

1. **Summarization**
   - English dialogue → English summary

2. **Translation**
   - English summary → Chinese summary

The intermediate English summary allows us to analyze whether errors come from the summarization stage or the translation stage.

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

We recommend using Python 3.10 or later.

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

---

## 4. Models

The project uses two fine-tuned summarization models and one pretrained translation model.

| Component | Model |
|---|---|
| BART summarizer | `yunu919/bart-large-dialogue-summarization` |
| mBART summarizer | `yunu919/mbart-large-dialogue-summarization` |
| English-to-Chinese translator | `Helsinki-NLP/opus-mt-en-zh` |

The summarization models can either be loaded directly from Hugging Face or trained locally using the scripts in this repository.

Hugging Face model links:

```text
https://huggingface.co/yunu919/bart-large-dialogue-summarization
https://huggingface.co/yunu919/mbart-large-dialogue-summarization
https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
```

---

## 5. Train the Summarization Models

The summarization models are trained for:

```text
English dialogue → English summary
```

### 5.1 Train BART

```bash
python scripts/train_bart.py \
  --train_path data/raw/train.json \
  --val_path data/raw/val.json \
  --test_path data/raw/test.json \
  --output_dir outputs/bart_model
```

The trained BART model will be saved to:

```text
outputs/bart_model/
```

### 5.2 Train mBART

```bash
python scripts/train_mbart.py \
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

## 6. Run the Inference Pipeline

The full inference pipeline performs:

```text
English dialogue → English summary → Chinese summary
```

It first generates intermediate English summaries using a fine-tuned summarization model. Then, it translates those English summaries into Chinese using `Helsinki-NLP/opus-mt-en-zh`.

---

### 6.1 Run Pipeline with BART

Using the Hugging Face BART checkpoint:

```bash
python scripts/run_inference_pipeline.py \
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

---

### 6.2 Run Pipeline with mBART

Using the Hugging Face mBART checkpoint:

```bash
python scripts/run_inference_pipeline.py \
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

---

### 6.3 Run Pipeline with a Locally Trained Model

If the model was trained locally, use the local model directory instead of the Hugging Face model ID.

Example:

```bash
python scripts/run_inference_pipeline.py \
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

---

## 7. Evaluate the Outputs

The system is evaluated at two stages:

| Stage | Prediction File | Reference Field | Purpose |
|---|---|---|---|
| English intermediate summary | `*_predictions_en.txt` | `summary` | Evaluate summarization quality before translation |
| Chinese final summary | `*_predictions_zh.txt` | `summary_zh` | Evaluate final cross-lingual summarization quality |

The main evaluation metrics are:

| Metric | Purpose |
|---|---|
| ROUGE-1 | Unigram overlap |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence overlap |
| BERTScore F1 | Semantic similarity |

For Chinese ROUGE evaluation, Chinese text should be segmented before score calculation. This project uses `jieba` for Chinese segmentation.

---

### 7.1 Evaluate BART English Predictions

```bash
python scripts/evaluate_outputs.py \
  --pred_path outputs/bart_predictions_en.txt \
  --ref_path data/raw/test.json \
  --ref_field summary \
  --lang en \
  --output_path outputs/bart_eval_en.json
```

---

### 7.2 Evaluate BART Chinese Predictions

```bash
python scripts/evaluate_outputs.py \
  --pred_path outputs/bart_predictions_zh.txt \
  --ref_path data/raw/test.json \
  --ref_field summary_zh \
  --lang zh \
  --output_path outputs/bart_eval_zh.json
```

---

### 7.3 Evaluate mBART English Predictions

```bash
python scripts/evaluate_outputs.py \
  --pred_path outputs/mbart_predictions_en.txt \
  --ref_path data/raw/test.json \
  --ref_field summary \
  --lang en \
  --output_path outputs/mbart_eval_en.json
```

---

### 7.4 Evaluate mBART Chinese Predictions

```bash
python scripts/evaluate_outputs.py \
  --pred_path outputs/mbart_predictions_zh.txt \
  --ref_path data/raw/test.json \
  --ref_field summary_zh \
  --lang zh \
  --output_path outputs/mbart_eval_zh.json
```

Expected evaluation outputs:

```text
outputs/bart_eval_en.json
outputs/bart_eval_zh.json
outputs/mbart_eval_en.json
outputs/mbart_eval_zh.json
```

---

## 8. Full Reproducible Workflow

To reproduce the project from scratch:

```text
1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies.
4. Obtain XSAMSum by following the ClidSum README instructions.
5. Place the XSAMSum train, validation, and test files under data/raw/.
6. Rename the files as train.json, val.json, and test.json if necessary.
7. Train BART and/or mBART, or load the Hugging Face checkpoints.
8. Run the inference pipeline.
9. Generate English and Chinese prediction files.
10. Evaluate English predictions against summary.
11. Evaluate Chinese predictions against summary_zh.
12. Save evaluation results under outputs/.
```
