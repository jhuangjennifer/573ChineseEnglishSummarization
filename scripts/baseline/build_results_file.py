import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_data import load_json_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join dataset records with baseline prediction text files."
    )
    parser.add_argument("--input_path", required=True, help="Source test or gold-set JSON.")
    parser.add_argument("--predicted_zh_path", required=True, help="Chinese predictions text file.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--output_json", required=True, help="Output JSON path.")
    parser.add_argument("--predicted_en_path", help="Optional English predictions text file.")

    return parser.parse_args()


def read_lines(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def main():
    args = parse_args()

    records = load_json_file(args.input_path).to_dict(orient="records")
    predicted_zh = read_lines(args.predicted_zh_path)
    predicted_en = read_lines(args.predicted_en_path) if args.predicted_en_path else []

    if len(records) != len(predicted_zh):
        raise ValueError(
            f"Length mismatch: {len(records)} input records vs "
            f"{len(predicted_zh)} Chinese predictions."
        )

    if predicted_en and len(records) != len(predicted_en):
        raise ValueError(
            f"Length mismatch: {len(records)} input records vs "
            f"{len(predicted_en)} English predictions."
        )

    rows = []
    for index, record in enumerate(records):
        row = {
            "test_index": record.get("test_index", index),
            "dialogue": record["dialogue"],
            "summary": record.get("summary", ""),
            "summary_zh": record["summary_zh"],
            "predicted_zh": predicted_zh[index],
        }
        if predicted_en:
            row["predicted_en"] = predicted_en[index]
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} rows to: {output_csv}")
    print(f"Wrote {len(rows)} records to: {output_json}")


if __name__ == "__main__":
    main()
