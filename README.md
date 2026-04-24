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

Raw dataset files are **not included in this repository** because of file size limits.  
To run the scripts locally, place the dataset files under:

```bash
data/raw/
├── train.json
├── val.json
└── test.json