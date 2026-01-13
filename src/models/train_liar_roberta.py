import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed


LABEL_TO_ID: Dict[str, int] = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}



@dataclass(frozen=True)
class RunConfig:
    model_name: str
    max_length: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    num_epochs: int
    weight_decay: float
    seed: int
    fp16: bool
    debug: bool
    run_name: str


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Fine-tune a model on LIAR.")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision (recommended on GPU).")
    parser.add_argument("--debug", action="store_true", help="Smoke test: tiny subset + 1 epoch.")
    parser.add_argument("--run_name", type=str, default="liar_roberta", help="Name for outputs (checkpoints + metrics file).")

    args = parser.parse_args()

    if args.debug:
        num_epochs = 1
        train_batch = min(args.train_batch_size, 8)
        eval_batch = min(args.eval_batch_size, 8)
        max_len = min(args.max_length, 128)
    else:
        num_epochs = args.num_epochs
        train_batch = args.train_batch_size
        eval_batch = args.eval_batch_size
        max_len = args.max_length

    fp16 = bool(args.fp16)


    return RunConfig(
        model_name=args.model_name,
        max_length=max_len,
        learning_rate=args.learning_rate,
        train_batch_size=train_batch,
        eval_batch_size=eval_batch,
        num_epochs=num_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fp16=fp16,
        debug=bool(args.debug),
        run_name=args.run_name,
    )


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def load_liar_dataset() -> Any:
    data_files = {
        "train": "data/liar_processed/train.jsonl",
        "validation": "data/liar_processed/valid.jsonl",
        "test": "data/liar_processed/test.jsonl",
    }

    for split, path in data_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {split} file: {path}")

    return load_dataset("json", data_files=data_files)


def prepare_tokenized_dataset(dataset: Any, tokenizer: Any, max_length: int, debug: bool) -> Any:    
    if debug:
        dataset["train"] = dataset["train"].select(range(min(200, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(100, len(dataset["validation"]))))
        dataset["test"] = dataset["test"].select(range(min(100, len(dataset["test"]))))

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, desc="Tokenizing")

    if "label_id" not in tokenized["train"].column_names:
        raise KeyError("Expected column 'label_id' in processed dataset.")

    tokenized = tokenized.rename_column("label_id", "labels")

    remove_cols = [c for c in ["id", "text", "label"] if c in tokenized["train"].column_names]
    if remove_cols:
        tokenized = tokenized.remove_columns(remove_cols)

    return tokenized


def save_run_results(out_path: Path, cfg: RunConfig, test_metrics: Dict[str, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "test_metrics": test_metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def main() -> None:
    cfg = parse_args()

    set_seed(cfg.seed)

    checkpoints_dir = Path("checkpoints") / cfg.run_name
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Run name: {cfg.run_name}")
    print(f"Model: {cfg.model_name}")
    print(f"Debug: {cfg.debug}")
    print(f"Max length: {cfg.max_length}")
    print(f"Batch size (train/eval): {cfg.train_batch_size}/{cfg.eval_batch_size}")
    print(f"LR: {cfg.learning_rate}, Epochs: {cfg.num_epochs}, Weight decay: {cfg.weight_decay}")
    print(f"FP16: {cfg.fp16}")

    dataset = load_liar_dataset()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=6,
        LABEL_TO_ID=LABEL_TO_ID,
        ID_TO_LABEL=ID_TO_LABEL,
    )

    tokenized = prepare_tokenized_dataset(dataset, tokenizer, cfg.max_length, cfg.debug)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        run_name=cfg.run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50 if not cfg.debug else 10,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=cfg.fp16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    trainer.train()

    best_dir = checkpoints_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved best model to: {best_dir.resolve()}")


    test_metrics = trainer.evaluate(tokenized["test"])

    results_path = Path("experiments") / f"{cfg.run_name}.json"

    
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    results_path = experiments_dir / f"{cfg.run_name}.json"
    save_run_results(results_path, cfg, test_metrics)


if __name__ == "__main__":
    main()
