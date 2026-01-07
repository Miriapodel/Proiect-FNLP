import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import pipeline
import torch
import os

# Generate full test predictions
print("Loading test data and model...")
test_df = pd.read_json("data/rumoureval_processed/test.jsonl", lines=True)

classifier = pipeline(
    "text-classification",
    model="checkpoints/rumour_roberta/final",
    tokenizer="checkpoints/rumour_roberta/final",
    device=0 if torch.cuda.is_available() else -1
)

print("Running predictions...")
results = []
for _, row in test_df.iterrows():
    pred = classifier(row["text"])[0]
    results.append({
        "id": row["id"],
        "text": row["text"],
        "true_label": row["label"],
        "pred_label": pred["label"],
        "confidence": pred["score"]
    })

output_df = pd.DataFrame(results)
output_df.to_json("experiments/rumour_test_predictions.jsonl", lines=True, orient="records")
print(f"Saved {len(output_df)} predictions to experiments/rumour_test_predictions.jsonl")

# Generate confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(
    output_df["true_label"],
    output_df["pred_label"],
    labels=["support", "deny", "query", "comment"]
)

plt.figure(figsize=(6, 5))
sns.set(font_scale=1.1)
heatmap = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["support", "deny", "query", "comment"],
    yticklabels=["support", "deny", "query", "comment"]
)
plt.title("RumourEval Stance Classification\nConfusion Matrix (RoBERTa-base)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
os.makedirs("report/figures", exist_ok=True)
plt.savefig("report/figures/rumour_confusion_matrix.png", dpi=200, bbox_inches='tight')
print("Confusion matrix saved to report/figures/rumour_confusion_matrix.png")

# Extract & save error examples
print("Extracting misclassified examples...")

support_errors = output_df[
    (output_df["true_label"] == "support") &
    (output_df["pred_label"] != "support")
    ].head(3)

deny_errors = output_df[
    (output_df["true_label"] == "deny") &
    (output_df["pred_label"] != "deny")
    ].head(2)

notes_path = "report/notes.md"
with open(notes_path, "a") as f:
    f.write("\n\n## RumourEval Error Analysis (Person C)\n")
    f.write("### Misclassified `support` examples (true=support, pred≠support):\n")
    for _, row in support_errors.iterrows():
        f.write(f"- **Text**: `{row['text'][:100]}...`\n")
        f.write(f"  - Predicted: `{row['pred_label']}` (confidence: {row['confidence']:.2f})\n")

    f.write("\n### Misclassified `deny` examples:\n")
    for _, row in deny_errors.iterrows():
        f.write(f"- **Text**: `{row['text'][:100]}...`\n")
        f.write(f"  - Predicted: `{row['pred_label']}` (confidence: {row['confidence']:.2f})\n")

print("Error examples appended to report/notes.md")

# Final summary
accuracy = (output_df["true_label"] == output_df["pred_label"]).mean()
print("\n" + "=" * 50)
print("FINAL SUMMARY")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Predictions saved: experiments/rumour_test_predictions.jsonl")
print(f"Confusion matrix: report/figures/rumour_confusion_matrix.png")
print(f"Error examples added to: report/notes.md")
print("=" * 50)