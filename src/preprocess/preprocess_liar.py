import json
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/liar_raw")
PROCESSED_DATA_DIR = Path("data/liar_processed")

TRAIN_FILE = RAW_DATA_DIR / "train.tsv"
VALID_FILE = RAW_DATA_DIR / "valid.tsv"
TEST_FILE = RAW_DATA_DIR / "test.tsv"

LABEL_TO_ID = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

LABEL_COL_IDX = 1
TEXT_COL_IDX = 2


def ensure_directories():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_liar_tsv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep="\t", header=None, quoting=3)
    
    return df


def preprocess_split(df, split_name):
    examples = []

    for idx, row in df.iterrows():
        raw_label = str(row[LABEL_COL_IDX]).strip()
        text = str(row[TEXT_COL_IDX]).strip()

        if raw_label not in LABEL_TO_ID:
            raise ValueError(f"Unknown label '{raw_label}' in {split_name} split.")

        if not text:
            continue

        examples.append({
                "id": f"{split_name}-{idx}",
                "text": text,
                "label": raw_label,
                "label_id": LABEL_TO_ID[raw_label],
            })

    return examples


def save_jsonl(examples, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def print_label_distribution(examples, split_name):
    counts = {label: 0 for label in LABEL_TO_ID}

    for ex in examples:
        counts[ex["label"]] += 1

    print(f"\nLabel distribution for {split_name}:")
    
    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<13} : {count}")



def main():
    ensure_directories()

    train_df = load_liar_tsv(TRAIN_FILE)
    valid_df = load_liar_tsv(VALID_FILE)
    test_df = load_liar_tsv(TEST_FILE)

    train_examples = preprocess_split(train_df, "train")
    valid_examples = preprocess_split(valid_df, "valid")
    test_examples = preprocess_split(test_df, "test")

    print_label_distribution(train_examples, "train")
    print_label_distribution(valid_examples, "valid")
    print_label_distribution(test_examples, "test")

    save_jsonl(train_examples, PROCESSED_DATA_DIR / "train.jsonl")
    save_jsonl(valid_examples, PROCESSED_DATA_DIR / "valid.jsonl")
    save_jsonl(test_examples, PROCESSED_DATA_DIR / "test.jsonl")


if __name__ == "__main__":
    main()
