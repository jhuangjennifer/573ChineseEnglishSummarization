import json
from pathlib import Path


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_lines(lines: list[str], output_path: str) -> None:
    """Save a list of strings as a line-by-line text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).strip() + "\n")


def read_lines(input_path: str) -> list[str]:
    """Read a line-by-line text file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def save_json(data, output_path: str) -> None:
    """Save data as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(input_path: str):
    """Load JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)