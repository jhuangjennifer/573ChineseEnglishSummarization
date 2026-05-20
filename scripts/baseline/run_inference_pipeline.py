import argparse
import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary_model",
        type=str,
        default="yunu919/bart-large-dialogue-summarization",
        help="Hugging Face summarization model path.",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="bart",
        help="Model name used for output file names, e.g., bart or mbart.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/raw/test.json",
        help="Path to test JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save prediction files.",
    )
    parser.add_argument(
        "--translation_model",
        type=str,
        default="Helsinki-NLP/opus-mt-en-zh",
        help="English-to-Chinese translation model.",
    )
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_summary_length", type=int, default=150)
    parser.add_argument("--num_beams", type=int, default=4)

    return parser.parse_args()


def generate_summary(text, tokenizer, model, args, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=args.max_input_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=args.max_summary_length,
            num_beams=args.num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def translate_to_chinese(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    output_en_path = os.path.join(
        args.output_dir, f"{args.model_tag}_predictions_en.txt"
    )
    output_zh_path = os.path.join(
        args.output_dir, f"{args.model_tag}_predictions_zh.txt"
    )

    print("Loading summarization model:", args.summary_model)
    summary_tokenizer = AutoTokenizer.from_pretrained(args.summary_model)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(args.summary_model).to(device)

    print("Loading translation model:", args.translation_model)
    translate_tokenizer = AutoTokenizer.from_pretrained(args.translation_model)
    translate_model = AutoModelForSeq2SeqLM.from_pretrained(args.translation_model).to(device)

    dataset = load_dataset("json", data_files={"test": args.input_path})
    test_dialogues = dataset["test"]["dialogue"]

    with open(output_en_path, "w", encoding="utf-8") as f_en, open(
        output_zh_path, "w", encoding="utf-8"
    ) as f_zh:
        for dialogue in tqdm(test_dialogues, desc="Running inference"):
            predicted_en = generate_summary(
                dialogue,
                summary_tokenizer,
                summary_model,
                args,
                device,
            )

            predicted_zh = translate_to_chinese(
                predicted_en,
                translate_tokenizer,
                translate_model,
                device,
            )

            f_en.write(predicted_en.strip() + "\n")
            f_zh.write(predicted_zh.strip() + "\n")

    print(f"Saved English predictions to: {output_en_path}")
    print(f"Saved Chinese predictions to: {output_zh_path}")


if __name__ == "__main__":
    main()