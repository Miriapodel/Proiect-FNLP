from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings

warnings.filterwarnings("ignore")

MODEL_DIR = "checkpoints/rumour_roberta/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

stance_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

texts = [
    "[PARENT] #Breaking: Pentagon releases video of the “mother of all bombs” being dropped in Afghanistan https:\/\/t.co\/GaXwhpWDmb [REPLY] @TODAYshow Blowed up real good. https:\/\/t.co\/Tr9pOpfIhE",
]

results = stance_classifier(texts)

for text, res in zip(texts, results):
    print(f"\nTweet: {text}")
    for label_score in res:
        print(f"{label_score['label']}: {label_score['score']:.4f}")
