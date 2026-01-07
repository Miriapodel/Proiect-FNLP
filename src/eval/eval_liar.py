import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

LABEL2ID: Dict[str, int] = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

@dataclass(frozen=True)
class EvalConfig:
    run_name: Optional[str]
    model_dir: Optional[Path]
    max_length: int
    out_dir: Path
    test_path: Path
    batch_size: int


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned LIAR model on the test split.")

    p.add_argument("--run_name",type=str)
    p.add_argument("--model_dir", type=str)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", type=str, default="report/figures")
    p.add_argument("--test_path", type=str, default="data/liar_processed/test.jsonl")

    args = p.parse_args()

    run_name = getattr(args, "run_name", None)
    model_dir = Path(args.model_dir) if getattr(args, "model_dir", None) else None

    out_dir = Path(args.out_dir)
    test_path = Path(args.test_path)

    return EvalConfig(
        run_name=run_name,
        model_dir=model_dir,
        max_length=int(args.max_length),
        out_dir=out_dir,
        test_path=test_path,
        batch_size=int(args.batch_size),
    )


def resolve_model_dir(cfg: EvalConfig) -> Path:
    if cfg.model_dir is not None:
        model_dir = cfg.model_dir
    else:
        assert cfg.run_name is not None
        model_dir = Path("checkpoints") / cfg.run_name / "best_model"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Expected HuggingFace model files in {model_dir}, but config.json is missing."
        )

    return model_dir


def load_test_dataset(test_path: Path) -> Dataset:
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    ds = load_dataset("json", data_files={"test": str(test_path)})["test"]

    required_cols = {"text", "label_id"}
    missing = required_cols - set(ds.column_names)
    
    if missing:
        raise ValueError(f"Processed test data missing required columns: {missing}")

    return ds


def tokenize_dataset(ds: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = ds.map(tokenize_batch, batched=True, desc="Tokenizing test set")
    tokenized = tokenized.rename_column("label_id", "labels")

    remove_cols = [c for c in ["text", "label"] if c in tokenized.column_names]

    if remove_cols:
        tokenized = tokenized.remove_columns(remove_cols)

    return tokenized


def predict(trainer: Trainer, tokenized_test: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    out = trainer.predict(tokenized_test)

    y_true = out.label_ids
    y_pred = np.argmax(out.predictions, axis=-1)

    return y_true, y_pred


def compute_summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    print("\nPer-class classification report (test):")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


def save_confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    labels = list(range(len(ID2LABEL)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    class_names = [ID2LABEL[i] for i in labels]

    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(labels, class_names, rotation=45, ha="right")
    plt.yticks(labels, class_names)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_eval_summary_json(summary: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    model_dir = resolve_model_dir(cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating model from: {model_dir.resolve()}")
    print(f"Test set: {cfg.test_path.resolve()}")
    print(f"Max length: {cfg.max_length} | Batch size: {cfg.batch_size}")

    test_ds = load_test_dataset(cfg.test_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    tokenized_test = tokenize_dataset(test_ds, tokenizer, cfg.max_length)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        args=None
    )

    y_true, y_pred = predict(trainer, tokenized_test)

    summary_metrics = compute_summary_metrics(y_true, y_pred)
    print("\nSummary metrics (test):")
    print(f"  accuracy : {summary_metrics['accuracy']:.4f}")
    print(f"  macro_f1 : {summary_metrics['macro_f1']:.4f}")

    print_classification_report(y_true, y_pred)

    run_tag = cfg.run_name if cfg.run_name else model_dir.name
    cm_path = cfg.out_dir / f"liar_confusion_{run_tag}.png"
    save_confusion_matrix_png(
        y_true,
        y_pred,
        out_path=cm_path,
        title=f"LIAR Confusion Matrix ({run_tag})",
    )
    print(f"\nSaved confusion matrix to: {cm_path.resolve()}")

    summary_path = cfg.out_dir / f"liar_eval_summary_{run_tag}.json"
    save_eval_summary_json(
        summary={
            "run_tag": run_tag,
            "model_dir": str(model_dir),
            "max_length": cfg.max_length,
            "metrics": summary_metrics,
        },
        out_path=summary_path,
    )
    print(f"Saved eval summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
