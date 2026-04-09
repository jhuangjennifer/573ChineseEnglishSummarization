# 573ChineseEnglishSummarization

Team project repository for **LING 573**.

## Project Overview
This project investigates **English-to-Chinese cross-lingual dialogue summarization**.  
Our goal is to build a system that takes **multi-turn English dialogues** as input and generates **concise Chinese summaries**.

We are currently focusing on **XMediaSum40k**, a dataset for cross-lingual dialogue summarization based on English dialogue inputs and Chinese target summaries.

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
2. build a baseline system,
3. evaluate the baseline, and
4. explore an improved model.

## Baseline Direction
We are currently considering a **pipeline baseline**, motivated by the dataset structure:

**English dialogue → English summary → Chinese translation**

Based on this, one possible starting point is a **summarize-then-translate** baseline.  
We may also compare this with a more direct **end-to-end English-to-Chinese summarization model**.

## Preprocessing
The current preprocessing goals include:
- keeping only the **English-source / Chinese-target** data,
- checking dialogue length, summary length, and speaker-turn formatting,
- preparing the data for model-specific tokenization and training.

## Repository Structure
- `src/` — source code
- `data/` — dataset files and processed outputs
- `docs/` — notes, references, and project materials
- `report/` — report drafts and course deliverables

## Notes
This README reflects our **current working plan** and may be updated as the project develops.
