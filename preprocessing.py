import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

LABEL2ID = {
    "support": 0,
    "deny": 1,
    "query": 2,
    "comment": 3,
}

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text(tweet_json):
    for key in ["full_text", "text", "body"]:
        if key in tweet_json:
            return tweet_json[key]
    return ""

def build_parent_map(structure):
    parent_map = {}

    def walk(node, parent_id=None):
        if isinstance(node, dict):
            for key, value in node.items():
                if parent_id is not None:
                    parent_map[key] = parent_id
                walk(value, key)

        elif isinstance(node, list):
            for item in node:
                walk(item, parent_id)

        elif isinstance(node, str):
            if parent_id is not None:
                parent_map[node] = parent_id

    walk(structure)
    return parent_map

def process_split(raw_root: Path, key_file: Path, split_name: str):
    stance_labels = load_json(key_file)["subtaskaenglish"]
    rows = []

    events = [d for d in raw_root.iterdir() if d.is_dir()]

    for event_dir in tqdm(events, desc=f"Processing RumourEval {split_name.upper()}"):
        for thread_dir in event_dir.iterdir():
            if not thread_dir.is_dir():
                continue

            source_dir = thread_dir / "source-tweet"
            source_files = list(source_dir.glob("*.json"))
            if not source_files:
                continue

            source_id = source_files[0].stem
            source_json = load_json(source_files[0])
            source_text = extract_text(source_json)

            replies = {}
            replies_dir = thread_dir / "replies"
            if replies_dir.exists():
                for rf in replies_dir.glob("*.json"):
                    replies[rf.stem] = extract_text(load_json(rf))

            structure_file = thread_dir / "structure.json"
            if not structure_file.exists():
                continue

            structure = load_json(structure_file)
            parent_map = build_parent_map(structure)

            for reply_id, reply_text in replies.items():
                if reply_id not in stance_labels:
                    continue

                label = stance_labels[reply_id]
                if label not in LABEL2ID:
                    continue

                parent_id = parent_map.get(reply_id)

                if parent_id == source_id:
                    parent_text = source_text
                elif parent_id in replies:
                    parent_text = replies[parent_id]
                else:
                    parent_text = ""

                if parent_text:
                    text = f"[PARENT] {parent_text} [REPLY] {reply_text}"
                else:
                    text = reply_text

                rows.append(
                    {
                        "id": reply_id,
                        "text": text,
                        "label": label,
                        "label_id": LABEL2ID[label],
                    }
                )

    df = pd.DataFrame(rows)
    out_dir = Path("data/rumoureval_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    print(f"\n{split_name.upper()} label distribution:")
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

