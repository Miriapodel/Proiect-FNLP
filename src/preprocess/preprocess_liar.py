import json
from pathlib import Path
from typing import List
import pandas as pd


RAW_DATA_DIR = Path("data/liar_raw")
PROCESSED_DATA_DIR = Path("data/liar_processed")

TRAIN_FILE = RAW_DATA_DIR / "train.tsv"
VALID_FILE = RAW_DATA_DIR / "valid.tsv"
TEST_FILE = RAW_DATA_DIR / "test.tsv"

LABEL2ID = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL_COL_IDX = 1
TEXT_COL_IDX = 2

def ensure_directories() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_liar_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep="\t", header=None, quoting=3)
    return df


def preprocess_split(df: pd.DataFrame, split_name: str) -> List[dict]:
    examples: List[dict] = []

    for idx, row in df.iterrows():
        raw_label = str(row[LABEL_COL_IDX]).strip()
        text = str(row[TEXT_COL_IDX]).strip()

        if raw_label not in LABEL2ID:
            raise ValueError(f"Unknown label '{raw_label}' encountered in {split_name} split.")

        if not text:
            continue

        examples.append({
                "id": f"{split_name}-{idx}",
                "text": text,
                "label": raw_label,
                "label_id": LABEL2ID[raw_label],
            })

    return examples


def save_jsonl(examples: List[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def print_label_distribution(examples: List[dict], split_name: str) -> None:
    counts = {label: 0 for label in LABEL2ID}

    for ex in examples:
        counts[ex["label"]] += 1

    print(f"\nLabel distribution for {split_name}:")
    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<13} : {count}")



def main() -> None:
    print("Starting LIAR preprocessing...")

    ensure_directories()

    train_df = load_liar_tsv(TRAIN_FILE)
    valid_df = load_liar_tsv(VALID_FILE)
    test_df = load_liar_tsv(TEST_FILE)

    train_examples = preprocess_split(train_df, "train")
    valid_examples = preprocess_split(valid_df, "valid")
    test_examples = preprocess_split(test_df, "test")


    print(f"\nNumber of examples:")
    print(f"  Train: {len(train_examples)}")
    print(f"  Valid: {len(valid_examples)}")
    print(f"  Test : {len(test_examples)}")

    print_label_distribution(train_examples, "train")
    print_label_distribution(valid_examples, "valid")
    print_label_distribution(test_examples, "test")

    save_jsonl(train_examples, PROCESSED_DATA_DIR / "train.jsonl")
    save_jsonl(valid_examples, PROCESSED_DATA_DIR / "valid.jsonl")
    save_jsonl(test_examples, PROCESSED_DATA_DIR / "test.jsonl")


if __name__ == "__main__":
    main()
