import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

SEED = 42
MODEL_NAME = "roberta-base"

MAX_LENGTH = 96
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05

OUTPUT_DIR = "checkpoints/rumour_veracity_roberta"
EXPERIMENTS_DIR = "experiments"

LABEL2ID = {"false": 0, "true": 1, "unverified": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        w = self.class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=w)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def main():
    set_seed(SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_path = Path("data/rumoureval_veracity_processed/train.jsonl")
    test_path = Path("data/rumoureval_veracity_processed/test.jsonl")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}. Run preprocess_rumoureval_veracity.py first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path}. Run preprocess_rumoureval_veracity.py first.")

    print("Loading data...")
    train_df = pd.read_json(train_path, lines=True).dropna(subset=["text", "label_id"])
    test_df = pd.read_json(test_path, lines=True).dropna(subset=["text", "label_id"])

    print("\nTrain label distribution:")
    print(train_df["label_id"].value_counts().sort_index().rename(index=ID2LABEL).to_dict())
    print("Test label distribution:")
    print(test_df["label_id"].value_counts().sort_index().rename(index=ID2LABEL).to_dict())

    tr_df, va_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df["label_id"],
    )

    counts = train_df["label_id"].value_counts().sort_index()
    counts = counts.reindex(range(NUM_LABELS), fill_value=0)

    if (counts == 0).any():
        print("\n[WARN] A class has 0 samples in train. Using uniform class weights.")
        class_weights = torch.ones(NUM_LABELS, dtype=torch.float)
    else:
        inv = 1.0 / counts.to_numpy(dtype=np.float64)
        inv = inv / inv.sum() * NUM_LABELS
        class_weights = torch.tensor(inv, dtype=torch.float)

    print("\nClass counts:", counts.to_dict())
    print("Class weights:", class_weights.tolist())

    train_ds = Dataset.from_pandas(tr_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    valid_ds = Dataset.from_pandas(va_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    test_ds = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_tok = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
    valid_tok = valid_ds.map(tokenize, batched=True, desc="Tokenizing valid")
    test_tok = test_ds.map(tokenize, batched=True, desc="Tokenizing test")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
        seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    best_dir = Path(OUTPUT_DIR) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved best model to: {best_dir}")

    print("\nEvaluating on TEST...")
    test_metrics = trainer.evaluate(test_tok)
    print(test_metrics)

    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    out_json = Path(EXPERIMENTS_DIR) / "rumour_veracity_roberta.json"
    out_json.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {out_json}")


if __name__ == "__main__":
    main()
