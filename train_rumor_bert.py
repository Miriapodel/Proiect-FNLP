import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_NAME = "roberta-base"
LEARNING_RATE = 2e-5

# MAX_LENGTH = 128
# BATCH_SIZE = 8
# NUM_EPOCHS = 4

MAX_LENGTH = 96
BATCH_SIZE = 4
NUM_EPOCHS = 3

OUTPUT_DIR = "checkpoints/rumour_roberta"

LABEL2ID = {"support": 0, "deny": 1, "query": 2, "comment": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 4

print("Loading data...")
train_df = pd.read_json("data/rumoureval_processed/train.jsonl", lines=True)
test_df = pd.read_json("data/rumoureval_processed/test.jsonl", lines=True)

train_df = train_df.dropna(subset=["label_id"])
#test_df = test_df.dropna(subset=["label_id"])

train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        #return_tensors="pt"
    )

train_tokenized = train_dataset.map(tokenize, batched=True, batch_size=1000)
test_tokenized = test_dataset.map(tokenize, batched=True, batch_size=1000)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

print("Starting training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=False,
    logging_steps=100,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\n" + "="*50)
print("FINAL TEST EVALUATION")
results = trainer.evaluate()
print(results)

os.makedirs("experiments", exist_ok=True)
with open("experiments/rumour_roberta.txt", "w") as f:
    f.write("RumourEval Stance Classification - RoBERTa-base\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy: {results['eval_accuracy']:.4f}\n")
    f.write(f"Macro F1: {results['eval_macro_f1']:.4f}\n")

print("\nDetailed classification report:")
test_preds = trainer.predict(test_tokenized)
y_true = test_preds.label_ids
y_pred = np.argmax(test_preds.predictions, axis=1)

report = classification_report(
    y_true, y_pred,
    target_names=["support", "deny", "query", "comment"],
    digits=4
)
print(report)

trainer.save_model("checkpoints/rumour_roberta/final")

with open("experiments/rumour_roberta_detailed.txt", "w") as f:
    f.write(report)

print("\nResults saved to:")
print(" - experiments/rumour_roberta.txt")
print(" - experiments/rumour_roberta_detailed.txt")