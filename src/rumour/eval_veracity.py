import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding

LABEL2ID = {"false": 0, "true": 1, "unverified": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def main():
    model_dir = Path("checkpoints/rumour_veracity_roberta/best_model")
    test_path = Path("data/rumoureval_veracity_processed/test.jsonl")

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model: {model_dir}. Train first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test set: {test_path}. Preprocess first.")

    print("Loading test data...")
    test_df = pd.read_json(test_path, lines=True).dropna(subset=["text", "label_id"])
    ds = Dataset.from_pandas(test_df[["id", "text", "label_id"]])

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=96)

    tok_ds = ds.map(tok, batched=True)
    tok_ds = tok_ds.rename_column("label_id", "labels")
    tok_ds = tok_ds.remove_columns(["text"])

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        args=None,
    )

    print("Predicting...")
    out = trainer.predict(tok_ds)
    logits = out.predictions
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    y_true = out.label_ids

    pred_labels = [ID2LABEL[int(i)] for i in y_pred]

    confidences = []
    for i, lab in enumerate(pred_labels):
        if lab == "unverified":
            confidences.append(0.0)
        else:
            confidences.append(float(probs[i, int(y_pred[i])]))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro-F1: {macro_f1:.4f}")

    report_txt = classification_report(
        y_true, y_pred,
        target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
        digits=4
    )
    print(report_txt)

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)

    (Path("experiments") / "rumour_veracity_report.txt").write_text(
        f"RumourEval 2019 - Subtask B (Veracity)\n"
        f"Model: {model_dir}\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report_txt}\n",
        encoding="utf-8"
    )

    preds_out = Path("experiments") / "rumour_veracity_predictions.jsonl"
    with preds_out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(test_df.itertuples(index=False)):
            rec = {
                "id": getattr(row, "id"),
                "true_label": ID2LABEL[int(getattr(row, "label_id"))],
                "pred_label": pred_labels[i],
                "confidence": float(confidences[i]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {preds_out}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("RumourEval Veracity (Subtask B) - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1, 2], ["false", "true", "unverified"], rotation=45, ha="right")
    plt.yticks([0, 1, 2], ["false", "true", "unverified"])
    plt.colorbar()

    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center")

    plt.tight_layout()
    fig_path = Path("report/figures") / "rumour_veracity_confusion.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to: {fig_path}")


if __name__ == "__main__":
    main()
