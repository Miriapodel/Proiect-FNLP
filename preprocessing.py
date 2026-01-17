import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

SEED = 42

def read_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_json(path, lines=True)

    if "text" not in df.columns or "label_id" not in df.columns:
        raise ValueError(f"{path} must contain 'text' and 'label_id'. Found: {list(df.columns)}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)
    return df

def eval_split(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }

def save_confusion_matrix(cm: np.ndarray, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    classes = [str(i) for i in range(cm.shape[0])]
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def run_baseline(train_path: str, test_path: str, out_prefix: str, max_features: int, ngram_max: int):
    train_df = read_jsonl(train_path)
    test_df = read_jsonl(test_path)

    X_train = train_df["text"].tolist()
    y_train = train_df["label_id"].to_numpy()

    X_test = test_df["text"].tolist()
    y_test = test_df["label_id"].to_numpy()

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, ngram_max),
                max_features=max_features,
            )),
            ("lr", LogisticRegression(
                max_iter=2000,
                random_state=SEED,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {"test": eval_split(y_test, y_pred)}

    os.makedirs("experiments", exist_ok=True)
    json_path = os.path.join("experiments", f"{out_prefix}.json")
    txt_path = os.path.join("experiments", f"{out_prefix}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Baseline: TF-IDF(1,{ngram_max}), max_features={max_features} + LogisticRegression\n")
        f.write(f"SEED={SEED}\n\n")
        f.write("[TEST]\n")
        f.write(f"Accuracy: {results['test']['accuracy']:.4f}\n")
        f.write(f"Macro-F1:  {results['test']['macro_f1']:.4f}\n\n")
        f.write(results["test"]["classification_report"])
        f.write("\n")

    cm = np.array(results["test"]["confusion_matrix"], dtype=int)
    fig_path = os.path.join("report", "figures", f"{out_prefix}_confusion.png")
    save_confusion_matrix(cm, fig_path, title=f"{out_prefix.replace('_', ' ').title()} - Confusion Matrix")

    print(f"[OK] {out_prefix}: acc={results['test']['accuracy']:.4f} macroF1={results['test']['macro_f1']:.4f}")
    print(f"[OK] Saved: {json_path}")
    print(f"[OK] Saved: {txt_path}")
    print(f"[OK] Saved: {fig_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["liar", "rumour", "rumour_veracity"])
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--ngram_max", type=int, default=2)
    args = ap.parse_args()

    if args.task == "liar":
        run_baseline(
            train_path="data/liar_processed/train.jsonl",
            test_path="data/liar_processed/test.jsonl",
            out_prefix="liar_baseline",
            max_features=args.max_features,
            ngram_max=args.ngram_max,
        )

    if args.task == "rumour":
        run_baseline(
            train_path="data/rumoureval_processed/train.jsonl",
            test_path="data/rumoureval_processed/test.jsonl",
            out_prefix="rumour_baseline",
            max_features=args.max_features,
            ngram_max=args.ngram_max,
        )

    if args.task == "rumour_veracity":
        run_baseline(
            train_path="data/rumoureval_veracity_processed/train.jsonl",
            test_path="data/rumoureval_veracity_processed/test.jsonl",
            out_prefix="rumour_veracity_baseline",
            max_features=args.max_features,
            ngram_max=args.ngram_max,
        )

if __name__ == "__main__":
    main()
