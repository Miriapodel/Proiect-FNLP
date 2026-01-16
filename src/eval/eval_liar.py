import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer
from scipy.special import softmax
import spacy
import json

LABEL_TO_ID = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

@dataclass(frozen=True)
class EvalConfig:
    run_name: Optional[str]
    model_dir: Optional[Path]
    max_length: int
    out_dir: Path
    test_path: Path
    batch_size: int


def parse_args():
    p = argparse.ArgumentParser()

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


def resolve_model_dir(cfg):
    if cfg.model_dir is not None:
        model_dir = cfg.model_dir
    elif cfg.run_name is not None:
        model_dir = Path("checkpoints") / cfg.run_name / "best_model"
    else:
        raise ValueError("Either --model_dir or --run_name must be provided to locate the model directory.")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Expected HuggingFace model files in {model_dir}, but config.json is missing."
        )

    return model_dir

def mask_NER_tokens(test_path, output_path):
    nlp = spacy.load("en_core_web_sm")
    pref = 'NER'
    counter = 0
    output_json_list = []

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    with open(test_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        sample = json.loads(json_str)
        doc = nlp(sample["text"])
        text = []
        for tok in doc:
            if tok.ent_type != 0:
                counter += 1
                text.append(pref + str(counter))
            else:
                text.append(tok.text)

        sample["text"] = " ".join(text)
        output_json_list.append(sample)

    with output_path.open("w", encoding="utf-8") as f:
        for example in output_json_list:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    #
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)

def load_test_dataset(test_path):
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    ds = load_dataset("json", data_files={"test": str(test_path)})["test"]

    required_cols = {"text", "label_id"}
    missing = required_cols - set(ds.column_names)
    
    if missing:
        raise ValueError(f"Processed test data missing required columns: {missing}")

    return ds



def tokenize_dataset(ds, tokenizer, max_length):
    def tokenize_batch(batch):
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


def predict(trainer, tokenized_test):
    out = trainer.predict(tokenized_test)

    y_true = out.label_ids
    y_pred = np.argmax(out.predictions, axis=-1)

    return y_true, y_pred

def predict_with_confidence(trainer, tokenized_test):
    out = trainer.predict(tokenized_test)

    y_true = out.label_ids
    y_pred = np.argmax(out.predictions, axis=-1)
    probs = softmax(out.predictions, axis=-1)
    confidence = probs.max(axis=-1)

    return y_true, y_pred, confidence

def compute_summary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def print_classification_report(y_true, y_pred):
    target_names = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]
    
    print("\nPer-class classification report (test):")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


def save_confusion_matrix_png(y_true, y_pred, out_path, title):
    labels = list(range(len(ID_TO_LABEL)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    class_names = [ID_TO_LABEL[i] for i in labels]

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


def save_eval_summary_json(summary, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_calibration_curves(y_true, y_prob, out_path):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(5, 2)
    colors = plt.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i, (label_id, label_name) in enumerate(ID_TO_LABEL.items()):
        labels_binary = y_true == label_id
        prob_true, prob_pred = calibration_curve(labels_binary, y_prob, pos_label=1, n_bins=10)
        display = CalibrationDisplay.from_predictions(labels_binary, y_prob, ax=ax_calibration_curve, color=colors(i), name=label_name)
        calibration_displays[label_name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4,1)]
    for i, (label_id, label_name) in enumerate(ID_TO_LABEL.items()):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[label_name].y_prob,
            range=(0, 1),
            bins=10,
            label=label_name,
            color=colors(i),
        )
        ax.set(title=label_name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_calibration_curves_binary(y_true, y_prob, out_path):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2)
    colors = plt.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    labels = ['true-ish']
    labels_binary = y_true > 2
    for i, label_name in enumerate(labels):
        display = CalibrationDisplay.from_predictions(labels_binary, y_prob, ax=ax_calibration_curve, color=colors(i), name=label_name)
        calibration_displays[label_name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    cfg = parse_args()
    model_dir = resolve_model_dir(cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    masked_ds_path = 'data/liar_masked_ner/test_masked_ner.jsonl'

    # mask_NER_tokens(cfg.test_path, Path(masked_ds_path))

    # test_ds = load_test_dataset(Path(masked_ds_path))
    test_ds = load_test_dataset(cfg.test_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=len(LABEL_TO_ID),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )

    tokenized_test = tokenize_dataset(test_ds, tokenizer, cfg.max_length)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        args=None
    )

    # y_true, y_pred = predict(trainer, tokenized_test)
    #

    y_true, y_pred, confidence = predict_with_confidence(trainer, tokenized_test)

    summary_metrics = compute_summary_metrics(y_true, y_pred)

    print_classification_report(y_true, y_pred)

    run_tag = cfg.run_name if cfg.run_name else model_dir.name
    cm_path = cfg.out_dir / f"liar_confusion_{run_tag}_masked.png"

    save_confusion_matrix_png(y_true, y_pred, out_path=cm_path, title=f"LIAR Confusion Matrix ({run_tag})",)

    summary_path = cfg.out_dir / f"liar_eval_summary_{run_tag}_masked.json"

    save_eval_summary_json(
        summary={
            "run_tag": run_tag,
            "model_dir": str(model_dir),
            "max_length": cfg.max_length,
            "metrics": summary_metrics,
        },
        out_path=summary_path,
    )



    run_tag = cfg.run_name if cfg.run_name else model_dir.name
    cc_path = cfg.out_dir / f"liar_calibration_{run_tag}.png"

    save_calibration_curves(y_true, confidence, cc_path)

    # save_calibration_curves_binary(y_true, confidence, cc_path)

    top_confidence_ids = np.argsort(confidence)[::-1]
    misclassified_ids = y_true != y_pred

    top_misclassified = []
    top_correct = []
    for top_confidence_id in top_confidence_ids:
        print(top_confidence_id, top_confidence_id in misclassified_ids, len(top_misclassified), len(top_correct))
        if top_confidence_id in misclassified_ids and len(top_misclassified) < 10:
            top_misclassified.append(test_ds[top_confidence_id]['text'])
        elif top_confidence_id not in misclassified_ids and len(top_correct) < 10:
            top_correct.append(test_ds[top_confidence_id]['text'])

    with open('D:\\univ_subj\\M_An_I\\FNLP\\Proiect-FNLP\\report\\files\\misclassified_samples.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(top_misclassified))

    with open('D:\\univ_subj\\M_An_I\\FNLP\\Proiect-FNLP\\report\\files\\correct_samples.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(top_correct))

if __name__ == "__main__":
    main()
