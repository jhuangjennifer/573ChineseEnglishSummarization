# 573ChineseEnglishSummarization

Team project repository for **LING 573**.

## Project Overview

This project investigates **English-to-Chinese cross-lingual dialogue summarization**.  
Our goal is to build a system that takes **multi-turn English dialogues** as input and generates **concise Chinese summaries**.

The task requires the model to solve two problems at the same time:

- **Dialogue summarization**: understanding speaker turns, context flow, and important information across multiple utterances.
- **Cross-lingual generation**: producing the final summary in Chinese while the source dialogue is in English.

Our current implementation focuses on a **pipeline baseline**:

**English dialogue → English summary → Chinese summary**

This design separates the task into two stages: summarization and translation.

## Task

Our main task is:

**English dialogue → Chinese summary**

We consider two approaches:

1. **Pipeline approach**
   - English dialogue → English summary
   - English summary → Chinese summary

2. **End-to-end approach**
   - English dialogue → Chinese summary

At the current stage, our primary focus is the **pipeline approach**, because it allows easier debugging and clearer comparison between components.

## Dataset

We use datasets from the **ClidSum** benchmark for cross-lingual dialogue summarization.

Dataset repository:  
https://github.com/krystalan/ClidSum?tab=readme-ov-file

The dataset includes the following fields:

- `dialogue`
- `summary`
- `summary_zh`

For our pipeline baseline:

- `dialogue` = English source dialogue
- `summary` = English intermediate summary
- `summary_zh` = Chinese reference summary

Raw dataset files are **not included in this repository** because of file size limits.

Place files locally under:

- `data/raw/train.json`
- `data/raw/val.json`
- `data/raw/test.json`

The raw `.json` files are ignored by Git, while `data/raw/.gitkeep` preserves the folder structure.

## Reproducibility Instructions

This section explains how to source the required materials and reproduce the full project pipeline, including data preparation, model training, inference, and evaluation.

---

### 1. Source the Required Materials

#### 1.1 Dataset

This project uses the **XSAMSum** dataset from the **ClidSum** benchmark.

Dataset repository:

```text
https://github.com/krystalan/ClidSum
```

Download the XSAMSum data from the ClidSum repository and place the files locally under `data/raw/`.

Expected file structure:

```text
data/raw/train.json
data/raw/val.json
data/raw/test.json
```

Each file should contain the following fields:

```text
dialogue
summary
summary_zh
```

Field usage:

| Field | Description |
|---|---|
| `dialogue` | English multi-turn dialogue |
| `summary` | English reference summary |
| `summary_zh` | Chinese reference summary |

Raw dataset files are not included in this repository and should not be committed to GitHub.

---

#### 1.2 Models

The project uses two fine-tuned summarization models and one pretrained translation model.

| Component | Model |
|---|---|
| BART summarizer | `yunu919/bart-large-dialogue-summarization` |
| mBART summarizer | `yunu919/mbart-large-dialogue-summarization` |
| English-to-Chinese translator | `Helsinki-NLP/opus-mt-en-zh` |

The summarization models can either be loaded directly from Hugging Face or trained locally using the scripts in this repository.

Hugging Face links:

```text
https://huggingface.co/yunu919/bart-large-dialogue-summarization
https://huggingface.co/yunu919/mbart-large-dialogue-summarization
https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
```

---

### 2. Set Up the Environment

Clone the repository:

```bash
git clone <your-repo-url>
cd 573ChineseEnglishSummarization
```

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

If `requirements.txt` is unavailable or incomplete, install the main dependencies manually:

```bash
pip install torch transformers accelerate datasets evaluate
pip install sentencepiece protobuf sacrebleu rouge-score bert-score
pip install pandas numpy scikit-learn tqdm jieba nltk
```

---

### 3. Prepare the Data

Place the downloaded XSAMSum files under:

```text
data/raw/
```

The expected structure is:

```text
data/raw/train.json
data/raw/val.json
data/raw/test.json
```

No heavy preprocessing is required for the current pipeline. The system uses:

```text
Input:  dialogue
Target: summary
Final reference: summary_zh
```

The dialogue format is preserved as much as possible because speaker names, turn boundaries, informal language, and emojis may contain useful dialogue information.

---

### 4. Train the Summarization Models

The summarization models are trained for:

```text
English dialogue → English summary
```

#### 4.1 Train BART

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

#### 4.2 Train mBART

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

For mBART, the source and target language should both be set to English:

```text
source language = en_XX
target language = en_XX
```

This ensures that mBART generates intermediate English summaries rather than summaries in another language.

---

### 5. Run the Full Inference Pipeline

The full pipeline performs:

```text
English dialogue → English summary → Chinese summary
```

It first generates intermediate English summaries using a fine-tuned summarization model.  
Then, it translates those English summaries into Chinese using `Helsinki-NLP/opus-mt-en-zh`.

---

#### 5.1 Run Pipeline with BART

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

#### 5.2 Run Pipeline with mBART

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

#### 5.3 Run Pipeline with a Locally Trained Model

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

### 6. Evaluate the Outputs

The system should be evaluated at two stages:

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

For Chinese evaluation, Chinese text should be segmented before ROUGE calculation.  
This project uses `jieba` for Chinese segmentation.

---

#### 6.1 Evaluate BART English Predictions

```bash
python scripts/evaluate_outputs.py \
  --pred_path outputs/bart_predictions_en.txt \
  --ref_path data/raw/test.json \
  --ref_field summary \
  --lang en \
  --output_path outputs/bart_eval_en.json
```

---

#### 6.2 Evaluate BART Chinese Predictions

```bash
python scripts/evaluate_outputs.py \
  --pred_path outputs/bart_predictions_zh.txt \
  --ref_path data/raw/test.json \
  --ref_field summary_zh \
  --lang zh \
  --output_path outputs/bart_eval_zh.json
```

---

#### 6.3 Evaluate mBART English Predictions

```bash
python scripts/evaluate_outputs.py \
  --pred_path outputs/mbart_predictions_en.txt \
  --ref_path data/raw/test.json \
  --ref_field summary \
  --lang en \
  --output_path outputs/mbart_eval_en.json
```

---

#### 6.4 Evaluate mBART Chinese Predictions

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

### 7. Expected Reproducible Workflow

To reproduce the full project from scratch, run the following steps in order:

```text
1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies.
4. Download XSAMSum from the ClidSum repository.
5. Place train.json, val.json, and test.json under data/raw/.
6. Train BART and/or mBART, or load the uploaded Hugging Face checkpoints.
7. Run the inference pipeline.
8. Generate English and Chinese prediction files.
9. Evaluate English predictions against summary.
10. Evaluate Chinese predictions against summary_zh.
11. Save evaluation results under outputs/.
```
