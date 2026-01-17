import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

LABEL2ID = {"false": 0, "true": 1, "unverified": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text(tweet_json: dict) -> str:
    for key in ["full_text", "text", "body"]:
        if key in tweet_json and tweet_json[key]:
            return str(tweet_json[key])
    return ""


def process_split(raw_root: Path, key_file: Path, split_name: str):
    key = load_json(key_file)

    if "subtaskbenglish" not in key:
        raise KeyError(f"Missing 'subtaskbenglish' in {key_file}. Found keys: {list(key.keys())[:10]}")

    veracity_labels = key["subtaskbenglish"]

    rows = []
    events = [d for d in raw_root.iterdir() if d.is_dir()]

    for event_dir in tqdm(events, desc=f"Processing RumourEval VERACITY {split_name.upper()}"):
        for thread_dir in event_dir.iterdir():
            if not thread_dir.is_dir():
                continue

            source_dir = thread_dir / "source-tweet"
            source_files = list(source_dir.glob("*.json"))
            if not source_files:
                continue

            source_file = source_files[0]
            source_id = source_file.stem

            if source_id not in veracity_labels:
                continue

            label = veracity_labels[source_id]
            if label not in LABEL2ID:
                continue

            source_json = load_json(source_file)
            text = extract_text(source_json).strip()
            if not text:
                continue

            rows.append(
                {
                    "id": source_id,
                    "text": text,
                    "label": label,
                    "label_id": LABEL2ID[label],
                }
            )

    df = pd.DataFrame(rows)
    out_dir = Path("data/rumoureval_veracity_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}.jsonl"

    df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    print(f"\n{split_name.upper()} label distribution:")
    if len(df) > 0:
        print(df["label"].value_counts())
    print(f"\nSaved {len(df)} examples → {out_path}")
    return df


if __name__ == "__main__":
    train_root = Path("rumoureval2019/rumoureval-2019-training-data/twitter-english")
    train_key = Path("rumoureval2019/rumoureval-2019-training-data/train-key.json")
    process_split(train_root, train_key, "train")

    test_root = Path("rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data")
    test_key = Path("rumoureval2019/rumoureval-2019-test-data/test-key.json")
    process_split(test_root, test_key, "test")
