import argparse
import json
import random
import re
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="data/raw/train.json")
    parser.add_argument("--val_path", type=str, default="data/raw/val.json")
    parser.add_argument("--test_path", type=str, default="data/raw/test.json")

    parser.add_argument("--model_checkpoint", type=str, default="facebook/bart-large")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/bart_large_dialogue_summarization",
    )

    parser.add_argument("--max_source_length", type=int, default=768)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_beams", type=int, default=5)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_json_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            return pd.DataFrame([data])

        raise ValueError(f"Unsupported JSON structure in {file_path}")

    except json.JSONDecodeError:
        records = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    records.append(json.loads(line))

        return pd.DataFrame(records)


def clean_dialogue_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_summary_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["dialogue", "summary"]
    df = df[required_columns].copy()

    df["dialogue"] = df["dialogue"].apply(clean_dialogue_text)
    df["summary"] = df["summary"].apply(clean_summary_text)

    df = df[(df["dialogue"] != "") & (df["summary"] != "")]
    df = df.reset_index(drop=True)

    return df


def main():
    args = parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Model checkpoint:", args.model_checkpoint)
    print("Output directory:", args.output_dir)

    train_df = prepare_dataframe(load_json_file(args.train_path))
    val_df = prepare_dataframe(load_json_file(args.val_path))
    test_df = prepare_dataframe(load_json_file(args.test_path))

    print("train_df shape:", train_df.shape)
    print("val_df shape:", val_df.shape)
    print("test_df shape:", test_df.shape)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    def preprocess_function(batch):
        model_inputs = tokenizer(
            batch["dialogue"],
            max_length=args.max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["summary"],
            max_length=args.max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return {key: round(value * 100, 4) for key, value in result.items()}

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    validation_output = trainer.predict(tokenized_datasets["validation"])
    print("Validation metrics:", validation_output.metrics)

    test_output = trainer.predict(tokenized_datasets["test"])
    print("Test metrics:", test_output.metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saved model and tokenizer to:", args.output_dir)


if __name__ == "__main__":
    main()