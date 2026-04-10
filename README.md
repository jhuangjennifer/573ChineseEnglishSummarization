# 573ChineseEnglishSummarization

Team project repository for **LING 573**.

## Project Overview
This project investigates **English-to-Chinese cross-lingual dialogue summarization**.  
Our goal is to build a system that takes **multi-turn English dialogues** as input and generates **concise Chinese summaries**.

We are currently focusing on **XMediaSum40k**, a dataset for cross-lingual dialogue summarization with English dialogue inputs and Chinese target summaries.

## Task
Our current task is:

**English dialogue → Chinese summary**

This task is interesting because it requires the model to handle both:
- **dialogue structure**, such as speaker turns and context flow, and
- **cross-lingual generation**, where the output summary is produced in a different language from the input.

## Dataset
We currently plan to use **XMediaSum40k** as our main dataset.

Dataset repository:  
https://github.com/krystalan/ClidSum?tab=readme-ov-file

At the moment, we are reviewing:
- whether the dataset is the best fit for our project goals,
- how the English–Chinese portion should be extracted and preprocessed, and
- how to format the source and target data for training.

## Current Plan
We are following **Option 1** and using an **existing dataset**.

Our current plan is to:
1. inspect and preprocess the dataset,
2. build a **pipeline baseline system**,
3. evaluate the baseline, and
4. explore an improved model.

## Baseline Direction
Our **primary baseline** is a **pipeline approach**:

**English dialogue → English summary → Chinese summary**

More specifically, the pipeline baseline consists of two stages:
1. **Summarizer**
   - input: **English dialogue**
   - output: **English summary**
2. **Translator**
   - input: **English summary**
   - output: **Chinese summary**

We chose this as our primary baseline because it is easier to analyze errors at each stage and provides a clear point of comparison for later improvements.

If time allows, we may also implement an **end-to-end comparison system**:

**English dialogue → Chinese summary**

## Preprocessing
Our current preprocessing goals include:
- identifying the fields needed for each stage of the pipeline,
- checking dialogue length, summary length, and speaker-turn formatting,
- cleaning and standardizing dialogue text where needed, and
- preparing the data for model-specific tokenization and training.

For the pipeline baseline, the expected training pairs are:
- **Summarizer**: `dialogue` → `summary`
- **Translator**: `summary` → `summary_zh`

## Repository Structure
- `src/` — source code
- `data/` — dataset files and processed outputs
- `docs/` — notes, references, and project materials
- `report/` — report drafts and course deliverables

## Notes
This README reflects our **current working plan** and may be updated as the project develops.
