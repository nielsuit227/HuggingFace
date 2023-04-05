import evaluate
import torch
from dataloader import load_data
from transformers import AutoModelForSequenceClassification

_, test_loader = load_data("data/first.csv")

# Load metric
metric = evaluate.load("accuracy")
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFace/whatsapp_classifier/models/distilbert_v1"
)

# Iteratively evaluate the model and compute metrics
model.eval()
for batch in test_loader:
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Get model accuracy and F1 score
result = metric.compute()
print(result)
