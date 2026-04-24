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

---

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

---

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

data/raw/
├── train.json
├── val.json
└── test.json

The raw `.json` files are ignored by Git, while `data/raw/.gitkeep` preserves the folder structure.

---

## Current Project Direction

We are following **Option 1** using an existing dataset.

Current workflow:

1. Inspect dataset structure and token lengths  
2. Prepare local raw data under `data/raw/`  
3. Fine-tune summarization models  
4. Upload trained models to Hugging Face  
5. Run full pipeline inference  
6. Generate English and Chinese prediction files  
7. Evaluate outputs using ROUGE and BERTScore  
8. Analyze model behavior and error patterns

---

## Baseline System

Our primary baseline is:

**English dialogue → English summary → Chinese summary**

### Stage 1: Summarizer

Input: English dialogue  
Output: English summary

Models:

- BART
- mBART

### Stage 2: Translator

Input: English summary  
Output: Chinese summary

The translation stage uses an English-to-Chinese machine translation model.

This design helps us:

- isolate summarization errors
- isolate translation errors
- compare summarization models
- identify where pipeline errors are introduced

---

## Models

### BART

Base model:

facebook/bart-large

Fine-tuned model:

yunu919/bart-large-dialogue-summarization

Task:

English dialogue → English summary

### mBART

Base model:

facebook/mbart-large-50-many-to-many-mmt

Fine-tuned model:

yunu919/mbart-large-dialogue-summarization

Task:

English dialogue → English summary

Language setting:

- source language: `en_XX`
- target language: `en_XX`

### PEGASUS

PEGASUS was tested during experimentation, but it is currently secondary because the available run was not a full training run due to GPU limitations.

Current focus remains on BART and mBART.

---

## Implementation

The project was initially developed in Jupyter notebooks and later converted into standalone Python scripts for reproducibility.

Main scripts:

scripts/
├── train_bart.py
├── train_mbart.py
└── run_inference_pipeline.py

### train_bart.py

Fine-tunes BART for:

English dialogue → English summary

### train_mbart.py

Fine-tunes mBART for:

English dialogue → English summary

### run_inference_pipeline.py

Runs the full pipeline:

English dialogue → English summary → Chinese summary

It loads a summarization model, generates English summaries, translates them into Chinese, and saves prediction files.

---

## Source Code Organization

Reusable helper functions are stored under `src/`.

src/
├── data/
│   ├── load_data.py
│   └── preprocess.py
├── models/
│   ├── bart_model.py
│   └── mbart_model.py
├── pipeline/
│   └── inference.py
└── utils/
    └── io_utils.py

### src/data/

Utilities for:

- loading JSON dataset files
- creating Hugging Face `DatasetDict`
- preparing train / validation / test splits

### src/models/

Utilities for:

- loading BART model and tokenizer
- loading mBART model and tokenizer
- setting language codes

### src/pipeline/

Reusable inference functions:

- summary generation
- translation
- device selection

### src/utils/

General utilities:

- saving files
- reading files
- creating output directories

---

## Run Instructions

### 1. Clone Repository

git clone <your-repo-url>  
cd 573ChineseEnglishSummarization

### 2. Create Virtual Environment

python -m venv .venv  
source .venv/bin/activate

Windows:

.venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

Manual install if needed:

pip install torch transformers accelerate datasets evaluate sentencepiece protobuf sacrebleu rouge-score bert-score pandas numpy scikit-learn tqdm

### 4. Prepare Dataset

Place files under:

data/raw/
├── train.json
├── val.json
└── test.json

---

## Run Examples

### Train BART

python scripts/train_bart.py \
  --train_path data/raw/train.json \
  --val_path data/raw/val.json \
  --test_path data/raw/test.json \
  --output_dir outputs/bart_model

Saved to:

outputs/bart_model/

### Train mBART

python scripts/train_mbart.py \
  --train_path data/raw/train.json \
  --val_path data/raw/val.json \
  --test_path data/raw/test.json \
  --output_dir outputs/mbart_model

Saved to:

outputs/mbart_model/

### Run Pipeline with BART

python scripts/run_inference_pipeline.py \
  --summary_model yunu919/bart-large-dialogue-summarization \
  --model_tag bart \
  --input_path data/raw/test.json \
  --output_dir outputs

Outputs:

outputs/bart_predictions_en.txt  
outputs/bart_predictions_zh.txt

### Run Pipeline with mBART

python scripts/run_inference_pipeline.py \
  --summary_model yunu919/mbart-large-dialogue-summarization \
  --model_tag mbart \
  --input_path data/raw/test.json \
  --output_dir outputs

Outputs:

outputs/mbart_predictions_en.txt  
outputs/mbart_predictions_zh.txt

---

## Output Files

outputs/
├── bart_predictions_en.txt
├── bart_predictions_zh.txt
├── mbart_predictions_en.txt
└── mbart_predictions_zh.txt

These files are ignored by Git because they can be regenerated.

---

## Evaluation

We evaluate generated summaries using:

- ROUGE
- BERTScore

For Chinese summaries, evaluation compares generated Chinese outputs against Chinese references.

For English intermediate summaries, evaluation helps inspect summarization quality before translation.

We analyze:

- whether summaries capture important dialogue content
- whether translation preserves meaning
- differences between BART and mBART
- whether metric scores align with qualitative judgments

---

## Hardware Notes

GPU is recommended for training and inference.

Recommended options:

- Google Colab GPU runtime
- Colab Pro with T4 or A100
- CUDA-enabled local GPU

CPU inference is possible but slower.  
Training on CPU is not recommended.

---

## Repository Structure

project/
├── scripts/  
├── src/  
├── data/  
│   └── raw/  
├── docs/  
├── notebooks/  
├── report/  
├── requirements.txt  
└── README.md

---

## Documentation

docs/
├── weekly_standup/
├── meeting_notes/
├── slides/
└── references/

report/
├── sections/
├── figures/
└── references.bib

---

## Notes

This README reflects our current implementation and project direction.  
It may be updated as the project develops.
